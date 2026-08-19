#!/usr/bin/env python
"""Import Qt sidecar tags into a WebAPI index.

The Qt application stores tags next to its index in ``<indexBase>.json``, keyed
by the **absolute path** of the CAD file::

    {"version":1,"parts":{"C:/cad/bracket.stp":{"tags":["bracket","v2"], ...}}}

The WebAPI keys everything by ``file_id``, the SHA-256 of the file contents.
The two are not interchangeable, so this tool re-reads each CAD file to compute
its hash and writes ``indexes/<name>/tags.json`` in the server's format.

This is the one place where confusing an id with a path is easy and expensive,
so the conversion is explicit and every unmatched entry is listed rather than
dropped: a path that no longer exists, or a file whose hash is not registered
in the target index, is reported instead of silently vanishing.

Usage
-----
    python tools/import_qt_tags.py <qt_sidecar.json> --index <name> [--dry-run]
    python tools/import_qt_tags.py <qt_sidecar.json> --index <name> --merge

By default the tool refuses to overwrite existing tags; pass ``--merge`` to add
to them or ``--force`` to replace the file. Run it with the server stopped: it
writes tags.json directly rather than through the API, so a concurrent edit
made through the API could be lost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import core  # noqa: E402  (path set up above)

_HASH_CHUNK = 1024 * 1024


def sha256_of(path: pathlib.Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_qt_sidecar(path: pathlib.Path) -> dict[str, list[str]]:
    """Return ``{absolute_path: [tag, ...]}`` from a Qt sidecar file."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} is not a JSON object.")
    parts = raw.get("parts")
    if not isinstance(parts, dict):
        raise ValueError(f"{path} has no 'parts' object.")

    out: dict[str, list[str]] = {}
    for cad_path, entry in parts.items():
        if not isinstance(entry, dict):
            continue
        tags = entry.get("tags")
        if not isinstance(tags, list):
            continue
        cleaned = [t for t in tags if isinstance(t, str) and t.strip()]
        if cleaned:
            out[str(cad_path)] = cleaned
    return out


def convert(
    sidecar: dict[str, list[str]],
    registered: set[str],
) -> tuple[dict[str, list[str]], list[dict[str, str]]]:
    """Map Qt paths onto file_ids, returning ``(by_file_id, skipped)``.

    Several paths can hash to the same id (the same CAD stored twice), in which
    case their tags are merged -- the ids are what the server indexes, so
    keeping them apart is not possible and dropping one would lose tags.
    """
    by_file_id: dict[str, list[str]] = {}
    skipped: list[dict[str, str]] = []

    for cad_path, tags in sorted(sidecar.items()):
        path = pathlib.Path(cad_path)
        if not path.is_file():
            skipped.append({"path": cad_path, "reason": "file_not_found"})
            continue
        try:
            file_id = sha256_of(path)
        except OSError as exc:
            skipped.append({"path": cad_path, "reason": f"unreadable: {exc}"})
            continue
        if file_id not in registered:
            skipped.append({"path": cad_path, "reason": "not_registered"})
            continue

        cleaned: list[str] = []
        for tag in tags:
            try:
                cleaned.append(core.normalize_tag(tag))
            except ValueError as exc:
                # A Qt tag may legitimately contain characters this server
                # rejects (notably '/', which would break the tag URL), so one
                # bad tag must not abort the import or discard the good ones.
                skipped.append(
                    {"path": cad_path, "reason": f"invalid tag {tag!r}: {exc}"}
                )
        if not cleaned:
            continue

        existing = by_file_id.setdefault(file_id, [])
        for tag in cleaned:
            if tag not in existing:
                existing.append(tag)

    return by_file_id, skipped


def build_document(
    by_file_id: dict[str, list[str]],
    existing: dict[str, Any],
    merge: bool,
    client_id: str,
) -> dict[str, Any]:
    """Return the tags.json document to write."""
    doc = {"version": 1, "updated_at": None, "parts": {}}
    if merge:
        doc["parts"] = {
            pid: dict(entry) for pid, entry in existing.get("parts", {}).items()
        }

    limit = core.max_tags_per_part()
    for file_id, tags in by_file_id.items():
        entry = doc["parts"].setdefault(file_id, {})
        current = list(entry.get("tags", []))
        for tag in tags:
            if tag not in current:
                current.append(tag)
        if len(current) > limit:
            print(
                f"  warning: {file_id[:12]} would hold {len(current)} tags; "
                f"keeping the first {limit}",
                file=sys.stderr,
            )
            current = current[:limit]
        entry["tags"] = current
        entry["updated_by"] = client_id
    return doc


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sidecar", help="Path to the Qt <indexBase>.json file")
    parser.add_argument("--index", required=True, help="Target WebAPI index name")
    parser.add_argument(
        "--dry-run", action="store_true", help="Report what would change and write nothing"
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Add to the existing tags.json instead of refusing to overwrite it",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Replace an existing tags.json outright",
    )
    parser.add_argument(
        "--client-id", default="import_qt_tags",
        help="Value recorded as updated_by (default: import_qt_tags)",
    )
    args = parser.parse_args(argv)

    sidecar_path = pathlib.Path(args.sidecar).expanduser()
    if not sidecar_path.is_file():
        print(f"Sidecar not found: {sidecar_path}", file=sys.stderr)
        return 2

    try:
        core._validate_index_name(args.index)
    except (ValueError, PermissionError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    tags_path = core._index_tags_path(args.index)
    if tags_path.exists() and not (args.merge or args.force or args.dry_run):
        print(
            f"{tags_path} already exists. Re-run with --merge to add to it or "
            "--force to replace it.",
            file=sys.stderr,
        )
        return 2

    try:
        core.load_env_file()
        registered = core._registered_index_ids(args.index)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    sidecar = load_qt_sidecar(sidecar_path)
    print(f"Qt sidecar:      {len(sidecar)} tagged paths")
    print(f"Target index:    {args.index} ({len(registered)} registered files)")

    by_file_id, skipped = convert(sidecar, registered)
    print(f"Converted:       {len(by_file_id)} file_ids")
    print(f"Skipped:         {len(skipped)}")
    for item in skipped:
        print(f"  - {item['reason']}: {item['path']}")

    if args.dry_run:
        print("\n--dry-run: nothing was written.")
        return 0
    if not by_file_id:
        print("\nNothing to import.")
        return 0

    existing, _ = core._read_tags_document(args.index)
    doc = build_document(by_file_id, existing, args.merge, args.client_id)
    with core._get_tags_lock(args.index):
        core._write_tags_document(args.index, doc)
    print(f"\nWrote {tags_path} ({len(doc['parts'])} parts).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

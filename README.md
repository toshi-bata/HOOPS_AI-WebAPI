# HOOPS AI WebAPI

A FastAPI-based REST API that exposes [HOOPS AI](https://www.techsoft3d.com/developers/products/hoops-ai/) (Tech Soft 3D) capabilities as HTTP endpoints.

---

## Requirements

- Python 3.12
- A valid **HOOPS AI license key**
- HOOPS AI (CPU or GPU version) installed in the environment
- **HOOPS AI Tutorials**  Ethe notebooks folder and its contents (ML datasets and pre-trained models) are required to run this server.  
  The tutorials are available at [github.com/techsoft3d/HOOPS-AI-tutorials](https://github.com/techsoft3d/HOOPS-AI-tutorials/tree/main).  
  Data packages (datasets and trained model checkpoints) must be obtained from the Tech Soft 3D File Transfer service by following the HOOPS AI installation instructions.

  **Directory layout**  E`notebooks/` and `packages/` must both reside directly under `HOOPS_AI_SDK_DIR`:

  ```
  <HOOPS_AI_SDK_DIR>/
  ├── notebooks/
  └── packages/
      ├── flows/
      ├── trained_ml_models/
      └── vectorstores/
          └── tmcad/
              ├── TMCAD_SIGNAL.faiss
              ├── TMCAD_SIGNAL.meta
              └── images_tmcad/
  ```

  **Pre-run requirements**  Esome endpoints require notebook output or downloaded files to be present in advance:

  | Endpoint | What to do | Required files |
  |---|---|---|
  | MFR endpoints | Run `3b_workflow_for_MFR_cadsynth.ipynb` | `notebooks/out/flows/ETL_CADSYNTH_training_b2/`<br>`.dataset` / `.infoset` / `.attribset` <br>`stream_cache/*.png` |
  | `/similarity/search` (`signal` preset, **default**) | Download `TMCAD_SIGNAL.faiss` bundle from Tech Soft 3D File Transfer and place under `packages/vectorstores/tmcad/`  E**no notebook run needed** | `packages/vectorstores/tmcad/TMCAD_SIGNAL.faiss` / `.meta`<br>`packages/vectorstores/tmcad/images_tmcad/` |
  | `/similarity/search` (`legacy` preset) | Run `5b_cad_search_using_HOOPS_embeddings.ipynb` (up to **Saving an Index**) | `notebooks/fabwave_embeddings_store.faiss` / `.meta` |
  | `/part-classification/dataset/*` | Run `3c_workflow_for_Part_classification_fabwave.ipynb`<br>(up to **Pipeline execution**) | `notebooks/out/flows/ETL_Fabwave_training_b2/` <br> `.dataset` / `.infoset` / `.attribset`<br>`stream_cache/*.png` |

  > **Tip:** Pre-generated dataset files are also available for download from the Tech Soft 3D File Transfer service  Eno need to run the notebooks yourself:  
  > URL: https://transfer.techsoft3d.com/link/mb9c3d8eTHhVHFpnI0FFaD  
  > Password: `HOOPS-AI-RELEASE`

---

## Setup

### 1. Install dependencies

Install HOOPS AI (CPU or GPU version) separately according to your HOOPS AI distribution instructions.
Then install the WebAPI dependencies into the **HOOPS AI virtual environment**:

**Windows:**
```bat
<Path\to\HOOPS_AI\install\dir>\.venv\Scripts\pip.exe install -r requirements.txt
```

**Linux:**
```bash
/path/to/HOOPS_AI/install/dir/.venv/bin/pip install -r requirements.txt
```

> On Ubuntu 22.04+ the system Python is externally managed (PEP 668) and will reject bare `pip install`.
> Using the HOOPS AI venv's pip avoids this restriction and ensures the same Python that runs the server has all required packages.

#### Additional steps for headless Linux (Ubuntu)

For running HOOPS AI on a headless Ubuntu server (no GPU, no monitor), refer to the following community forum post for step-by-step instructions:

**[I tried running HOOPS AI v1.1 headless on Ubuntu 24.04 EC2  ETech Soft 3D Forum](https://forum.techsoft3d.com/t/i-tried-running-hoops-ai-v1-1-headless-on-ubuntu-24-04-ec2/5165)**

### 2. Place the web viewer JS file

The 3D viewer uses the HOOPS Web Viewer monolith JS (not tracked in git). Copy it manually:

**Windows:**
```bat
copy "<HOOPS_AI_INSTALL_DIR>\.venv\Lib\site-packages\hoops_viewer\static\javascript\communicator\web-viewer-monolith\hoops-web-viewer-monolith.mjs" "static\hoops-web-viewer-monolith.mjs"
```

**Linux:**
```bash
cp "<hoops_ai_install_dir>/.venv/lib/python3.12/site-packages/hoops_viewer/static/javascript/communicator/web-viewer-monolith/hoops-web-viewer-monolith.mjs" "static/hoops-web-viewer-monolith.mjs"
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
# Windows
copy .env.example .env

# Linux
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `HOOPS_AI_LICENSE` | ✁E| Your HOOPS AI license key |
| `HOOPS_AI_SDK_DIR` | ✁E| Absolute path to your HOOPS AI SDK install directory (must contain `notebooks/` and `packages/`) |
| `HOOPS_AI_MFR_FLOW_NAME` | optional | MFR flow name (dataset files are resolved relative to this) |
| `HOOPS_AI_MFR_MODEL_NAME` | optional | MFR trained model checkpoint filename (e.g. `ts3d_162k_mfr.ckpt`) |
| `HOOPS_AI_EMBEDDINGS_MODEL_NAME` | optional | Embeddings trained model checkpoint filename (e.g. `ts3d_1M_hoops_embeddings.ckpt`). Used by the **`legacy` default-index preset** and when `PUT /similarity/default-model/setting?model=legacy`. |
| `HOOPS_AI_FAISS_INDEX_PATH` | optional | FAISS index file for the **`legacy` preset** of similarity search (e.g. `fabwave_embeddings_store.faiss`), located directly under `<HOOPS_AI_SDK_DIR>/notebooks`. Not used when the active default-index is `signal`. |
| `HOOPS_AI_EMBEDDINGS_MODEL_NAME_SIGNAL` | optional | SIGNAL architecture embeddings model checkpoint (e.g. `ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt`). Used as the **default active model** for `/compare`, `/map`, and `/index/create`, and also by the **`signal` default-index preset** (`TMCAD_SIGNAL.faiss`). Change the active model at runtime via `PUT /similarity/default-model/setting`. |
| `HOOPS_AI_FAISS_INDEX_PATH_SIGNAL` | optional | FAISS index filename for the **`signal` preset** (default) of similarity search (e.g. `TMCAD_SIGNAL.faiss`), located under `<HOOPS_AI_SDK_DIR>/packages/vectorstores/tmcad/`. Defaults to `TMCAD_SIGNAL.faiss` when unset. |
| `HOOPS_AI_PART_CLASS_MODEL_NAME` | optional | Filename of the trained GraphClassification checkpoint under `packages/trained_ml_models/` (e.g. `ts3d_graphclassification_5k_10epochs.ckpt`) |
| `HOOPS_AI_PART_CLASS_FLOW_NAME` | optional | Part Classification flow name (required for `/part-classification/dataset/*` endpoints). The server automatically prefers `<HOOPS_AI_SDK_DIR>/notebooks/out/flows/<name>` (notebook output, includes thumbnails) and falls back to `<HOOPS_AI_SDK_DIR>/packages/flows/<name>` (pre-packaged). |
| `HOOPS_AI_PART_CLASS_LABEL_KEY` | optional | Label array key for dataset queries (default: `part_label`; use `task_A` for custom ETL) |
| `HOOPS_AI_ENABLE_DEMO_FEATURES` | optional | Set to `true` to expose the demo-only endpoints listed below. Defaults to `false` (disabled) so a public deployment never serves them by accident. |
| `HOOPS_AI_ASSEMBLY_SEARCH_JOBS` | optional | Thread-pool size for assembly-to-assembly scoring (`POST /similarity/index/{name}/search-assembly`). Defaults to `8`, matching `hoops_ai_native_bridge`. Lower it when assembly searches starve the server's request workers. |
| `HOOPS_AI_LOG_LEVEL` | optional | Root log level applied at startup (`DEBUG`, `INFO`, `WARNING`, ...). Defaults to `INFO`; an unrecognised value falls back to `INFO`. uvicorn configures only its own loggers, so without this the application's `logger.info()` diagnostics — such as the assembly matcher build time — are dropped. |
| `HOOPS_AI_MAX_UPLOAD_BYTES` | optional | Per-file upload cap in bytes (default `4294967296`, i.e. 4 GB). Exceeding it returns **413**. |
| `HOOPS_AI_EXISTS_MAX_IDS` | optional | Maximum ids per `POST /files/exists` request (default `1000`). Exceeding it returns **422**. |
| `HOOPS_AI_ZIP_MAX_TOTAL_BYTES` | optional | Uncompressed-size cap for a ZIP archive (default `524288000`, i.e. 500 MB). |
| `HOOPS_AI_ZIP_MAX_FILES` | optional | CAD-file-count cap for a ZIP archive (default `50`). |
| `HOOPS_AI_CAD_SHARED_DIR` | optional | Location of the **CAD store** (default: `uploads/`). This directory holds the only copy of the payload behind every registered index and is never cleared by the server. |
| `HOOPS_AI_OUT_TTL_HOURS` | optional | Age at which files under `out/` (viewer streams, result images, shape maps) are swept at startup (default `24`). `0` disables the sweep. |
| `HOOPS_AI_EMBEDDINGS_CACHE_TTL_DAYS` | optional | Age at which cached embeddings under `embeddings_cache/` are swept at startup (default `0`, i.e. keep forever). Recomputing one entry costs a full CAD load. |
| `HOOPS_AI_JOB_MAX_CONCURRENCY` | optional | How many registration jobs run at once (default `1`). Serialised because of worker memory, because hoops_ai's `error_summary.json` is per-process, and because the live-progress shim replaces the process-wide `sys.stderr`. |
| `HOOPS_AI_JOB_TTL_DAYS` | optional | Retention of finished job records under `jobs/` (default `7`). `0` keeps them until the record cap evicts them. |
| `HOOPS_AI_JOB_MAX_RECORDS` | optional | Hard cap on retained job records (default `1000`). |
| `HOOPS_AI_MAX_TAGS_PER_PART` | optional | Maximum number of index tags one part may carry (default `32`). |
| `HOOPS_AI_MAX_WORKERS` | optional | Cap used **only when `workers` is omitted** (default `8`). Each worker is a spawned interpreter holding its own ~2 GB copy of the checkpoint, so peak RSS grows roughly linearly with this value. An explicit `workers` in the request is passed to the SDK unchanged. |
| `HOOPS_AI_MODEL_FOOTPRINT_MB` | optional | Assumed per-worker memory footprint used to bound the automatic worker count by free RAM (default `2048`). |
| `HOOPS_AI_MIN_FILES_PARALLEL` | optional | Force a single worker below this many files. Defaults to `1`, i.e. disabled: a file-count threshold ignores how heavy each file is, and a few heavy assemblies are faster on several workers. Set it if your corpus is uniformly light. |
| `HOOPS_AI_EMBED_TIME_LIMIT` | optional | Per-file embedding budget in seconds. Unset (the default) leaves the SDK's own 120 s. Raise it to let heavy assemblies finish instead of failing with `Timeout`. |
| `HOOPS_AI_ALLOW_SERVER_PATHS` | optional | Set to `true` to let jobs read CAD from `server_paths` on the server machine. Defaults to `false`: it reads any file the server process can read and bypasses the upload size and extension checks. |

> **Note:** `HOOPS_AI_LICENSE` is read **only** from the `.env` file, not from system environment variables.

#### Demo-only endpoints (`HOOPS_AI_ENABLE_DEMO_FEATURES`)

These endpoints mutate server-wide state or run long/heavy jobs. They back the tools
in the private `HOOPS_AI-MCPServer-demo` companion repo and are hidden (return `404`)
unless `HOOPS_AI_ENABLE_DEMO_FEATURES=true`:

- `GET`/`PUT /similarity/default-model/setting` (embedding-model switch)
- `GET`/`PUT /similarity/default-index/setting` (default FAISS index preset switch)
- `POST /similarity/index/create`, `GET /similarity/index/list`, `POST /similarity/index/add`,
  `DELETE /similarity/index/{name}/parts`, `DELETE /similarity/index/{name}` (named index management)- `POST /similarity/map`, `GET /similarity/map/job/{job_id}`, `POST /similarity/map/{map_id}/query`,
  `POST /similarity/map/{map_id}/add-to-index` (Shape Space Map)
- `GET /MFR/dataset/table-of-contents`, `GET /MFR/labels/description`, `GET /MFR/files/search`,
  `GET /MFR/files/{file_id}/thumbnail` (MFR dataset browsing)
- `GET /part-classification/dataset/table-of-contents`, `GET /part-classification/dataset/label-distribution`
  (Part Classification dataset browsing)
- `POST /context/predict` (context-layer prediction)

All other endpoints (upload, viewer, B-Rep analysis, MFR/Part-Classification inference,
`compare`/`embed`/`search` similarity, searching an *existing* named index via
`POST /similarity/index/{name}/search` and
`POST /similarity/index/{name}/search-assembly`, and the read-only named-index
reporting/asset endpoints `GET /similarity/index/{name}/stats`,
`GET /similarity/index/{name}/parts`,
`GET /similarity/index/{name}/parts/{part_id}/thumbnail`, `.../scs`, and the bulk
registration job endpoints `POST`/`GET`/`DELETE /similarity/index/{name}/jobs[...]`)
remain available regardless of this flag.

Example `.env`:

```
HOOPS_AI_LICENSE=XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
HOOPS_AI_SDK_DIR=C:\hoops_ai
HOOPS_AI_MFR_FLOW_NAME=ETL_CADSYNTH_training_b2
HOOPS_AI_MFR_MODEL_NAME=ts3d_162k_mfr.ckpt
HOOPS_AI_EMBEDDINGS_MODEL_NAME=ts3d_1M_hoops_embeddings.ckpt
HOOPS_AI_FAISS_INDEX_PATH=fabwave_embeddings_store.faiss
HOOPS_AI_EMBEDDINGS_MODEL_NAME_SIGNAL=ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt
HOOPS_AI_FAISS_INDEX_PATH_SIGNAL=TMCAD_SIGNAL.faiss
HOOPS_AI_PART_CLASS_MODEL_NAME=ts3d_graphclassification_5k_10epochs.ckpt
HOOPS_AI_PART_CLASS_FLOW_NAME=ETL_Fabwave_training_b2
```

### 4. Start the server

Run the following command from the repository root, using the Python executable from your HOOPS AI virtual environment.

**Windows:**

```bat
<Path\to\HOOPS_AI\install\dir>\.venv\Scripts\python.exe main.py --host 0.0.0.0 --port 8000
```

**Linux:**

```bash
/path/to/HOOPS_AI/install/dir/.venv/bin/python main.py --host 0.0.0.0 --port 8000
```


> **Note:** Port `8000` is the default. If port 8000 is already in use, the server will print an error and exit  Esimply retry with a different port (e.g. `--port 8001`).
> When using the HOOPS AI MCP server, the WebAPI URL defaults to `http://127.0.0.1:8000` and no configuration is needed for local use on the default port.
> If you change the port or run the WebAPI on a different machine, add `HOOPS_WEBAPI_URL` to the `"env"` section of your MCP client config (e.g. `claude_desktop_config.json`):
> ```json
> "env": { "HOOPS_WEBAPI_URL": "http://127.0.0.1:8001" }
> ```

> **Note (Linux):** If the server is not reachable from other machines, check the firewall. On Ubuntu with `ufw` enabled, open the port with:
> ```bash
> sudo ufw allow 8000/tcp
> ```

Once the server is running, you can access it at:

- API base URL: `http://<server-ip>:8000`
- Interactive docs (Swagger UI): `http://<server-ip>:8000/docs`

> **`<server-ip>` substitution:**  
> - **Same machine**  Euse `127.0.0.1` (e.g. `http://127.0.0.1:8000`). No IP lookup needed.  
> - **Different machine**  Euse the LAN IP of the server machine (e.g. `http://192.168.0.6:8000`).  
>   On Windows, run `ipconfig` on the server to find its IP address.

---

## API Endpoints

### File Management

#### Upload CAD file

Upload a local CAD file to the server. Returns a `file_id` derived from the file's SHA-256 hash.
Uploading the same file again returns the same `file_id` without re-storing the file.

```
POST /files/upload
```

**Windows (PowerShell):**
```powershell
curl.exe -X POST "http://127.0.0.1:8000/files/upload" -F "file=@C:\path\to\model.stp"
```

**Linux:**
```bash
curl -X POST "http://127.0.0.1:8000/files/upload" -F "file=@/path/to/model.stp"
```

**Response:**

```json
{ "file_id": "a3f8c2...", "filename": "model.stp", "already_existed": false }
```

Pass the returned `file_id` to any processing endpoint instead of re-uploading the same file.

The body is streamed to disk and hashed in one pass, so large assemblies never
have to fit in memory. A single file is capped by `HOOPS_AI_MAX_UPLOAD_BYTES`
(default 4 GB); exceeding it returns **413**.

---

#### Upload several CAD files at once

Send many files in one request. Use this with `POST /files/exists` to register a
whole folder: hash locally, ask which ids the server already has, and upload only
the missing ones.

```
POST /files/upload-batch
```

**Windows (PowerShell):**
```powershell
curl.exe -X POST "http://127.0.0.1:8000/files/upload-batch" `
  -F "files=@C:\path\to\a.stp" -F "files=@C:\path\to\b.stp"
```

**Response:**

```json
{
  "files": [ { "file_id": "a3f8c2...", "filename": "a.stp", "already_existed": false } ],
  "errors": [ { "filename": "b.stp", "detail": "..." } ],
  "count": 1
}
```

One bad file does not abort the request: every part sent appears in either
`files` or `errors`, so the two lists always add up to the number of parts.
`POST /files/upload` keeps its single-object response and is unchanged.

---

#### Check which files the server already has

```
POST /files/exists
```

**Request / response:**

```json
{ "file_ids": ["a3f8c2...", "ffffff..."] }
```
```json
{ "known": ["a3f8c2..."], "unknown": ["ffffff..."], "invalid": [] }
```

Ids must be lower-case 64-character SHA-256 hex digests; anything else comes
back under `invalid` rather than `unknown`, so a client that hashes incorrectly
gets a distinct signal instead of silently re-uploading everything on every run.
Duplicates are collapsed. At most `HOOPS_AI_EXISTS_MAX_IDS` (default 1000) ids
per request; more returns **422**.

---

#### Upload CAD file or ZIP from server-side path

Register a CAD file or ZIP archive that already exists on the server's filesystem by providing
its path directly. The server reads the file itself — no binary upload is needed. This is the
recommended approach for **MCP / scripted clients** when the file path is known (e.g. a path
the user pasted into the chat).

```
POST /files/upload-from-path?file_path=<path>
```

- **Single CAD file** — uploaded and a single `file_id` is returned.
- **ZIP archive** — every recognised CAD file inside is extracted and uploaded;
  all resulting `file_id` values are returned together.

Absolute paths (e.g. `C:\temp\parts.zip`) are accepted as-is.
Relative paths are resolved under `HOOPS_AI_CAD_SHARED_DIR`.

**Windows (PowerShell) — single file:**
```powershell
curl.exe -X POST "http://127.0.0.1:8000/files/upload-from-path?file_path=C:\temp\bracket.stp"
```

**Windows (PowerShell) — ZIP archive:**
```powershell
curl.exe -X POST "http://127.0.0.1:8000/files/upload-from-path?file_path=C:\temp\parts.zip"
```

**Linux:**
```bash
curl -X POST "http://127.0.0.1:8000/files/upload-from-path?file_path=/tmp/parts.zip"
```

**Response:**

```json
{
  "files": [
    { "file_id": "a3f8c2...", "filename": "bracket_a.step" },
    { "file_id": "cd34ef...", "filename": "bracket_b.step" }
  ],
  "errors": []
}
```

Pass the returned `file_id` values to `POST /similarity/compare`, `POST /similarity/map`,
`POST /similarity/index/add`, etc.

ZIP limits: 500 MB total uncompressed size, 50 CAD files per archive (HTTP 413 if exceeded).
Both are raised with `HOOPS_AI_ZIP_MAX_TOTAL_BYTES` / `HOOPS_AI_ZIP_MAX_FILES`.

> **Note:** Requires the WebAPI server to be able to access the given path.
> For the typical local setup (MCP and WebAPI on the same machine) this works out of the box.

---

### 3D CAD Viewer

#### Launch viewer  EUpload file

Upload a local CAD file and open an interactive browser viewer.

```
POST /CAD/viewer
```

**Windows (PowerShell):**
```powershell
curl.exe -X POST "http://127.0.0.1:8000/CAD/viewer" -F "file=@C:\path\to\model.stp"
```

**Linux:**
```bash
curl -X POST "http://127.0.0.1:8000/CAD/viewer" -F "file=@/path/to/model.stp"
```

**Response:**

```json
{ "viewer_url": "http://<server-ip>:<viewer_port>/index.html", "image_url": "http://127.0.0.1:8000/out/<stem>.png" }
```

Open the returned `viewer_url` in your browser to view the model. `image_url` is a PNG preview of the model.

> **Note:** `uploads/` is the persistent CAD store and is **never** cleared —
> index records hold only the SHA-256 `file_id`, so deleting it would destroy the
> payload of every registered index. `out/` is transient: files older than
> `HOOPS_AI_OUT_TTL_HOURS` (default 24 h) are swept at startup, so a `image_url`
> or `viewer_url` handed out earlier is not guaranteed to stay valid forever.

#### Launch viewer  EShared folder path

Open a CAD file already present in the shared folder (`HOOPS_AI_CAD_SHARED_DIR`).

```
POST /CAD/viewer/from-path
```

**Windows (PowerShell):**
```powershell
curl.exe -X POST "http://127.0.0.1:8000/CAD/viewer/from-path" -d "cad_file_path=model.stp"
```

**Linux:**
```bash
curl -X POST "http://127.0.0.1:8000/CAD/viewer/from-path" -d "cad_file_path=model.stp"
```

**Response:** same as above.

> This endpoint is also used by the browser UI at `http://<server-ip>:8000/CAD/viewer`.

#### Terminate viewer

```
DELETE /CAD/viewer          # terminate last active viewer
DELETE /CAD/viewer?all=true # terminate all viewers
```

**Windows (PowerShell):**
```powershell
curl.exe -X DELETE "http://127.0.0.1:8000/CAD/viewer"
curl.exe -X DELETE "http://127.0.0.1:8000/CAD/viewer?all=true"
```

**Linux:**
```bash
curl -X DELETE "http://127.0.0.1:8000/CAD/viewer"
curl -X DELETE "http://127.0.0.1:8000/CAD/viewer?all=true"
```

**Response:** `{ "terminated": 1 }`

---

### B-Rep Analysis

#### Face adjacency graph

Build a face adjacency graph from the B-Rep model. Returns graph data and a PNG visualization URL.

```
POST /BRep/adjacency-graph
```

**Windows (PowerShell):**
```powershell
curl.exe -X POST "http://127.0.0.1:8000/BRep/adjacency-graph" -F "file=@C:\path\to\model.SLDPRT"
```

**Linux:**
```bash
curl -X POST "http://127.0.0.1:8000/BRep/adjacency-graph" -F "file=@/path/to/model.SLDPRT"
```

**Response:**

```json
{
  "graph": {
    "nodes": [0, 1, 2, ...],
    "edges": [[0, 1], [1, 2], ...],
    "num_nodes": 144,
    "num_edges": 210
  },
  "image_url": "http://127.0.0.1:8000/out/<uuid>.png"
}
```

#### Face and edge attributes

Extract face and edge attributes from the B-Rep model.

```
POST /BRep/attributes
```

**Windows (PowerShell):**
```powershell
curl.exe -X POST "http://127.0.0.1:8000/BRep/attributes" -F "file=@C:\path\to\model.SLDPRT"
```

**Linux:**
```bash
curl -X POST "http://127.0.0.1:8000/BRep/attributes" -F "file=@/path/to/model.SLDPRT"
```

**Response:**

```json
{
  "faces": {
    "types": [...], "areas": [...], "centroids": [...],
    "loops": [...], "types_description": {...}
  },
  "edges": {
    "types": [...], "lengths": [...], "dihedrals": [...],
    "convexities": [...], "types_description": {...}
  }
}
```

#### Face and edge type counts

Return face and edge counts grouped by type, aggregated server-side. Prefer this
endpoint over `/BRep/attributes` for any counting question (e.g. "how many faces",
"faces by type") — an AI/MCP client asked to tally a raw array of a few hundred
per-face/per-edge entries by hand is prone to reporting a wrong, non-reproducible
count; this endpoint returns the counts already aggregated.

```
POST /BRep/type-counts
```

**Windows (PowerShell):**
```powershell
curl.exe -X POST "http://127.0.0.1:8000/BRep/type-counts" -F "file=@C:\path\to\model.SLDPRT"
```

**Linux:**
```bash
curl -X POST "http://127.0.0.1:8000/BRep/type-counts" -F "file=@/path/to/model.SLDPRT"
```

**Response:**

```json
{
  "faces": {
    "total": 448,
    "by_type": {"Plane": 8, "Cylinder": 343, "Cone": 49, "Torus": 48}
  },
  "edges": {
    "total": 1222,
    "by_type": {"Line": 392, "Circle": 494, "Nurbs": 336}
  }
}
```

---

### Manufacturing Feature Recognition (MFR)

#### Dataset table of contents

Returns a summary of the loaded MFR dataset.

```
GET /MFR/dataset/table-of-contents
```

**Windows (PowerShell):**
```powershell
curl.exe "http://127.0.0.1:8000/MFR/dataset/table-of-contents"
```

**Linux:**
```bash
curl "http://127.0.0.1:8000/MFR/dataset/table-of-contents"
```

#### List label descriptions

Returns all MFR label IDs with their names and descriptions.

```
GET /MFR/labels/description
```

**Windows (PowerShell):**
```powershell
curl.exe "http://127.0.0.1:8000/MFR/labels/description"
```

**Linux:**
```bash
curl "http://127.0.0.1:8000/MFR/labels/description"
```

#### Search files by feature

Returns CAD file names and IDs that contain a given manufacturing feature.

```
GET /MFR/files/search?feature_name=<name>
```

**Windows (PowerShell):**
```powershell
curl.exe "http://127.0.0.1:8000/MFR/files/search?feature_name=through%20hole"
```

**Linux:**
```bash
curl "http://127.0.0.1:8000/MFR/files/search?feature_name=through%20hole"
```

**Response:**

```json
{
  "file_names": ["bracket_a.stp", "housing_b.stp"],
  "file_list": [1, 3]
}
```

#### File thumbnail

Returns the thumbnail PNG image for a given file ID.

```
GET /MFR/files/{file_id}/thumbnail
```

**Windows (PowerShell):**
```powershell
curl.exe "http://127.0.0.1:8000/MFR/files/1/thumbnail" -o thumbnail.png
```

**Linux:**
```bash
curl "http://127.0.0.1:8000/MFR/files/1/thumbnail" -o thumbnail.png
```

**Response:** PNG image (`image/png`)

#### Run inference

Upload a CAD file and run MFR inference. Launches the CAD viewer and returns predictions, probabilities, and viewer URL.
Note: no `image_url` preview is returned — the static PNG snapshot has no prediction colors baked in (colorization is applied client-side in the viewer only), so open `viewer_url` to see the colorized result.

```
POST /MFR/inference
```

**Windows (PowerShell):**
```powershell
curl.exe -X POST "http://127.0.0.1:8000/MFR/inference" -F "file=@C:\path\to\model.SLDPRT"
```

**Linux:**
```bash
curl -X POST "http://127.0.0.1:8000/MFR/inference" -F "file=@/path/to/model.SLDPRT"
```

**Response:**

```json
{
  "predictions": [...],
  "probabilities": [...],
  "viewer_url": "http://<server-ip>:<viewer_port>/index.html"
}
```


---

### Shape Similarity Search

Shape similarity endpoints fall into two groups:

- **Embedding model (index-free)**  E`default-model/setting`, `embed`, `compare`, `map`  Euse the embedding model only; no FAISS index is needed.
- **Index-based search**  E`default-index/setting`, `search`, `index-info`, and all `index/*` endpoints  Equery a FAISS index (built-in presets or a user-created named index).

---

#### Embedding model (index-free)

These endpoints use the embedding model only and do **not** require a FAISS index.

##### Default embedding model setting

Read or change the server-wide active embedding model used by `/embed`, `/compare`, `/map`, and `/index/create`.
The default is `'signal'` (HOOPS AI SIGNAL model).

```
GET  /similarity/default-model/setting
PUT  /similarity/default-model/setting?model=<model>
```

| Parameter | Values | Description |
|---|---|---|
| `model` | `signal` *(default)*, `legacy` | Embeddings model: `'signal'` = SIGNAL model (`HOOPS_AI_EMBEDDINGS_MODEL_NAME_SIGNAL`); `'legacy'` = 1M model (`HOOPS_AI_EMBEDDINGS_MODEL_NAME`) |

**Windows (PowerShell)  Eread current setting:**
```powershell
curl.exe "http://127.0.0.1:8000/similarity/default-model/setting"
```

**Windows (PowerShell)  Eswitch to 1M (legacy) model:**
```powershell
curl.exe -X PUT "http://127.0.0.1:8000/similarity/default-model/setting?model=legacy"
```

**Linux:**
```bash
curl "http://127.0.0.1:8000/similarity/default-model/setting"
curl -X PUT "http://127.0.0.1:8000/similarity/default-model/setting?model=signal"
```

**Response:**
```json
{ "model": "signal" }
```

---

##### Compute shape embedding (index-free)

Compute (or retrieve from cache) the shape embedding vector for a single CAD part.
This endpoint does **not** require a FAISS index  Ethe embedding model alone is sufficient.

```
POST /similarity/embed
```

Supply **either** a file upload or a `file_id` from a previous `POST /files/upload`.

**Windows (PowerShell):**
```powershell
# Upload a file and get its embedding
curl.exe -X POST "http://127.0.0.1:8000/similarity/embed" -F "file=@C:\path\to\bracket.step"

# Or use an already-uploaded file_id
curl.exe -X POST "http://127.0.0.1:8000/similarity/embed?file_id=a3f8c2..."
```

**Linux:**
```bash
curl -X POST "http://127.0.0.1:8000/similarity/embed" -F "file=@/path/to/bracket.step"
```

**Response:**

```json
{
  "file_id": "a3f8c2...",
  "filename": "bracket.step",
  "dim": 512,
  "model_name": "hoops_embeddings_model",
  "num_bodies": 1,
  "cached": false
}
```

Add `?include_vector=true` to include the raw float array in the response (omitted by default to save bandwidth).

| Field | Description |
|---|---|
| `file_id` | SHA-256 content hash of the uploaded file |
| `filename` | Original filename |
| `dim` | Embedding vector dimension |
| `model_name` | Name of the embedding model used |
| `num_bodies` | Number of solid bodies detected in the CAD file |
| `cached` | `true` if the vector was returned from cache |
| `vector` | Raw float array (only present when `include_vector=true`) |

---

##### Compare parts by shape similarity (index-free)

Compare multiple CAD parts and return a pairwise cosine similarity matrix.
Input sources can be combined freely.
This endpoint does **not** require a FAISS index.

```
POST /similarity/compare
```

| Input | How to supply |
|---|---|
| Existing file IDs | `?file_ids=<id1>,<id2>,...` query parameter |
| CAD file uploads | `files` multipart field (one or more) |
| ZIP archive | `zip_file` multipart field (auto-extracted, Zip Slip protected) |

The embeddings model is taken from the server-wide setting (`PUT /similarity/default-model/setting`).
Default is `'signal'` (HOOPS AI SIGNAL model).

At least **two** valid parts are required.  Per-file failures are collected in `errors`
and do not abort the request (unless fewer than two parts succeed).

**Windows (PowerShell)  Eupload two files directly:**
```powershell
curl.exe -X POST "http://127.0.0.1:8000/similarity/compare" -F "files=@C:\path\to\bracket_a.step" -F "files=@C:\path\to\bracket_b.step"
```

**Linux  Eupload two files directly:**
```bash
curl -X POST "http://127.0.0.1:8000/similarity/compare" -F "files=@/path/to/bracket_a.step" -F "files=@/path/to/bracket_b.step"
```

**Linux  Ecompare using already-uploaded file_ids:**
```bash
curl -X POST "http://127.0.0.1:8000/similarity/compare?file_ids=a3f8c2...,cd34ef..."
```

**Linux  Ecompare files inside a ZIP archive:**
```bash
curl -X POST "http://127.0.0.1:8000/similarity/compare" -F "zip_file=@/path/to/parts.zip"
```

**Response:**

```json
{
  "count": 3,
  "model_name": "hoops_embeddings_model",
  "files": [
    {"index": 0, "file_id": "ab12...", "filename": "bracket_a.step", "num_bodies": 1},
    {"index": 1, "file_id": "cd34...", "filename": "bracket_b.step", "num_bodies": 1},
    {"index": 2, "file_id": "ef56...", "filename": "gear.step",      "num_bodies": 2}
  ],
  "matrix": [
    [1.0,    0.9532, 0.6821],
    [0.9532, 1.0,    0.7015],
    [0.6821, 0.7015, 1.0   ]
  ],
  "pairs": [
    {"a": 0, "b": 1, "score": 0.9532},
    {"a": 1, "b": 2, "score": 0.7015},
    {"a": 0, "b": 2, "score": 0.6821}
  ],
  "errors": []
}
```

| Field | Description |
|---|---|
| `count` | Number of parts compared |
| `model_name` | Embedding model used |
| `files` | Metadata for each part in index order |
| `matrix` | N×N cosine similarity matrix (diagonal = 1.0) |
| `pairs` | All i < j pairs sorted by similarity score descending |
| `errors` | Per-file failures that were skipped (empty on full success) |

ZIP archives are filtered to recognised CAD extensions (66 formats, mirroring the
HOOPS Exchange "All Supported Files" list used by `hoops_ai_qt_sandbox`).
Paths that escape the extraction directory (Zip Slip) are rejected with HTTP 400.
Uncompressed size is capped at 500 MB and file count at 50 (HTTP 413 if exceeded);
raise them with `HOOPS_AI_ZIP_MAX_TOTAL_BYTES` / `HOOPS_AI_ZIP_MAX_FILES`.

---

##### Shape space map (index-free)

Arrange a set of CAD parts in an interactive 3D scene so that shape-similar parts are
placed closer together.  Embeddings are compared by cosine similarity and laid out with
classical MDS (multidimensional scaling), then rendered together in the HOOPS Web Viewer.

```
POST /similarity/map
GET  /similarity/map/show?map=<map_id>
```

| Input | How to supply |
|---|---|
| Existing file IDs | `?file_ids=<id1>,<id2>,...` query parameter |
| CAD file uploads | `files` multipart field (one or more) |
| ZIP archive | `zip_file` multipart field (auto-extracted, Zip Slip protected) |
| Sync mode | `?sync=true` — block until done and return full result (HTTP 200). Optional `?timeout=<seconds>` (default 300). Recommended for MCP/scripted clients that cannot poll. |

At least **two** valid parts are required.  Accepts the same three input sources as
`POST /similarity/compare`.  The embeddings model is taken from the server-wide setting
(`PUT /similarity/default-model/setting`; default `'signal'`).  The response includes a 3D `position`
for each part, the similarity `matrix`, a Kruskal `stress` value (layout accuracy:
`0.0` is exact), and an absolute `viewer_url` that opens the interactive map.
The viewer page fetches its layout data from `/out/shape_map_<map_id>.json`.

**Linux  Egenerate a shape map from uploaded files:**
```bash
# Upload parts
curl -s -X POST http://localhost:8000/files/upload -F "file=@part_a.step"
curl -s -X POST http://localhost:8000/files/upload -F "file=@part_b.step"

# Generate shape map (async — returns job_id immediately)
curl -s -X POST "http://localhost:8000/similarity/map?file_ids=<id_a>,<id_b>" | python -m json.tool

# Generate shape map (sync — blocks until done, returns full result)
curl -s -X POST "http://localhost:8000/similarity/map?file_ids=<id_a>,<id_b>&sync=true" | python -m json.tool

# Open the viewer_url from the response in a browser
```

**Windows (PowerShell)  Eupload parts directly (sync mode):**
```powershell
curl.exe -X POST "http://127.0.0.1:8000/similarity/map?sync=true" -F "files=@C:\path\to\bracket_a.step" -F "files=@C:\path\to\bracket_b.step"
```

**Response (abridged):**

```json
{
  "map_id": "a1b2c3d4",
  "viewer_url": "http://localhost:8000/similarity/map/show?map=a1b2c3d4",
  "count": 2,
  "parts": [
    {"index": 0, "file_id": "ab12...", "filename": "bracket_a.step",
     "scs_url": "http://localhost:8000/out/xxxx_bracket_a.scs", "position": [0.5, 0.0, 0.0]},
    {"index": 1, "file_id": "cd34...", "filename": "bracket_b.step",
     "scs_url": "http://localhost:8000/out/yyyy_bracket_b.scs", "position": [-0.5, 0.0, 0.0]}
  ],
  "matrix": [[1.0, 0.9532], [0.9532, 1.0]],
  "stress": 0.0,
  "errors": []
}
```

| Field | Description |
|---|---|
| `map_id` | Identifier for the generated layout |
| `viewer_url` | Absolute URL of the interactive 3D viewer page |
| `count` | Number of parts placed |
| `parts` | Per-part metadata, absolute `scs_url`, and centred 3D `position` |
| `matrix` | N×N cosine similarity matrix (diagonal = 1.0) |
| `stress` | Kruskal stress-1 layout accuracy (`< 0.01` = exact, higher = approximate) |
| `errors` | Per-file upload/embed/SCS failures that were skipped (non-fatal) |

The viewer overlays a scale slider (to spread parts apart or pack them together), a layout
accuracy indicator, and per-part filename labels that track the camera.

---

##### Shape space map  Equery overlay (index-free)

Highlight a single query CAD part inside an **existing** shape-space map.  The query
part is embedded with the same pipeline used to build the map and projected into the
existing 3D coordinate space using the out-of-sample MDS extension formula, so it appears
near its most similar parts.  It is rendered in **magenta** so it is clearly distinguishable.

```
POST /similarity/map/{map_id}/query
```

| Parameter | Where | Description |
|---|---|---|
| `map_id` | path | `map_id` returned by `POST /similarity/map` |
| `file_id` | query | `file_id` of an already-uploaded part |
| `file` | multipart | CAD file upload (alternative to `file_id`) |
| `persist` | query | `false` (default)  Eoverlay only; `true`  Eadd to original map permanently |

Supply **either** `file_id` **or** a `file` upload.

**Windows (PowerShell)  Edirect upload:**
```powershell
curl.exe -X POST "http://localhost:8000/similarity/map/d2a7f205/query" -F "file=@C:\temp\Sprocket.step"
```

**Linux  Euse an already-uploaded file:**
```bash
curl -s -X POST "http://localhost:8000/similarity/map/d2a7f205/query?file_id=<id>" | python -m json.tool
```

**Response (abridged):**
```json
{
  "overlay_map_id": "e5f6a7b8",
  "viewer_url": "http://localhost:8000/similarity/map/show?map=e5f6a7b8",
  "query_part": {
    "index": 4, "file_id": "ab12...", "filename": "Sprocket.step",
    "scs_url": "http://localhost:8000/out/xxxx_Sprocket.scs",
    "position": [0.12, -0.05, 0.0], "is_query": true
  },
  "nearest_parts": [
    {"index": 2, "file_id": "cd34...", "filename": "gear.step", "score": 0.9741},
    {"index": 0, "file_id": "ef56...", "filename": "sprocket_v2.step", "score": 0.9312}
  ],
  "persisted": false,
  "errors": []
}
```

| Field | Description |
|---|---|
| `overlay_map_id` | New temporary map that includes the query part |
| `viewer_url` | Absolute URL  Eopen in browser to see the query highlighted in magenta |
| `query_part` | Query part metadata, position, and `is_query: true` flag |
| `nearest_parts` | Top-5 most similar existing parts sorted by cosine similarity |
| `persisted` | `true` when `persist=true` was used and the query was added to the original map |

The overlay map is independent of the original  Eby default it exists only until the
server restarts.  Use `persist=true` to permanently add the query part to the source map.

---

#### Index-based search

These endpoints query the active default FAISS index to find similar parts.  The active preset can be switched between `signal` (TMCAD, default) and `legacy` (fabwave) via `PUT /similarity/default-index/setting`.

##### Default index setting

Read or switch the active default-index preset used by `/search`, `/part-image`, and `/index-info`.

```
GET  /similarity/default-index/setting
PUT  /similarity/default-index/setting?index=<preset>
```

| Parameter | Values | Description |
|---|---|---|
| `index` | `signal` *(default)*, `legacy` | `'signal'` = `HOOPS_AI_FAISS_INDEX_PATH_SIGNAL` (TMCAD_SIGNAL.faiss, 39 k parts, SIGNAL model); `'legacy'` = `HOOPS_AI_FAISS_INDEX_PATH` (1M model, notebook-generated) |

**Windows (PowerShell)  Eread current setting:**
```powershell
curl.exe "http://127.0.0.1:8000/similarity/default-index/setting"
```

**Windows (PowerShell)  Eswitch to legacy (fabwave) index:**
```powershell
curl.exe -X PUT "http://127.0.0.1:8000/similarity/default-index/setting?index=legacy"
```

**Linux:**
```bash
curl "http://127.0.0.1:8000/similarity/default-index/setting"
curl -X PUT "http://127.0.0.1:8000/similarity/default-index/setting?index=signal"
```

**Response:**
```json
{ "index": "signal" }
```

---

##### Default index info

Return metadata about the currently active FAISS similarity-search index.
This endpoint is read-only and never triggers index construction
or model training.  When the index has not been loaded yet, a
``"not_loaded"`` status is returned instead of an error.

```
GET /similarity/index-info
```

**Windows (PowerShell):**
```powershell
curl.exe "http://127.0.0.1:8000/similarity/index-info"
```

**Linux:**
```bash
curl "http://127.0.0.1:8000/similarity/index-info"
```

**Response (index loaded):**

```json
{
  "preset": "signal",
  "status": "loaded",
  "index_last_modified": "2025-06-01T12:00:00Z",
  "index_count": 39736,
  "model_name": "CUSTOM:hoops_embeddings_signal",
  "embedding_dim": 2048,
  "metadata": null
}
```

**Response (index not yet loaded):**

```json
{
  "preset": "signal",
  "status": "not_loaded",
  "index_last_modified": null,
  "index_count": null,
  "model_name": null,
  "embedding_dim": null,
  "metadata": null
}
```

| Field | Description |
|---|---|
| `preset` | Active preset: `"signal"` or `"legacy"` |
| `status` | `"loaded"` or `"not_loaded"` |
| `index_last_modified` | UTC last-modified timestamp of the index file (`null` if file not found) |
| `index_count` | Number of embeddings stored in the index |
| `model_name` | Name of the embedding model used to build the index |
| `embedding_dim` | Dimension of each embedding vector |
| `metadata` | Auxiliary metadata stored in the index (e.g. `failed_count`) |

---

##### Search default index

Upload a CAD file and retrieve the most similar parts from the active FAISS index.
The active index is controlled by `PUT /similarity/default-index/setting` (default: `signal` = TMCAD_SIGNAL.faiss).

```
POST /similarity/search?top_k=<n>
```

**Windows (PowerShell):**
```powershell
curl.exe -X POST "http://127.0.0.1:8000/similarity/search?top_k=10" -F "file=@C:\path\to\model.step"
```

**Linux:**
```bash
curl -X POST "http://127.0.0.1:8000/similarity/search?top_k=10" -F "file=@/path/to/model.step"
```

**Response:**

```json
{
  "results": [
    {"id": "part_042", "score": 0.997},
    {"id": "part_018", "score": 0.991}
  ],
  "image_url": "http://127.0.0.1:8000/out/<uuid>.png"
}
```

- `results`  Etop-k matches sorted by similarity score (higher = more similar)
- `image_url`  EURL to a PNG grid image of the search results

---

##### Part thumbnail image

Return the pre-generated PNG thumbnail for a trained part by filename.

> **Note:** Requires thumbnail images pre-generated by the embeddings notebook (`5b_cad_search_using_HOOPS_embeddings.ipynb`). Images are expected at `notebooks/out/images/STEP/<stem>.png` (or `<stem>_white.png`). Returns 404 if the images have not been generated.

```
GET /similarity/part-image?filename=<name>
```

**Windows (PowerShell):**
```powershell
curl.exe "http://127.0.0.1:8000/similarity/part-image?filename=part_042.stp" -o part_042.png
```

**Linux:**
```bash
curl "http://127.0.0.1:8000/similarity/part-image?filename=part_042.stp" -o part_042.png
```

**Response:** PNG image (`image/png`)

---

#### Named index management

Manage user-created similarity indexes that grow over time.  Unlike the built-in
read-only ``default`` index (backed by ``HOOPS_AI_FAISS_INDEX_PATH``), named indexes
are fully writable: create an empty index, register new parts whenever they arrive,
and query immediately  Eall via Web API with no notebook re-runs.

Each named index is bound to the embeddings model used when it was created (stored in
an `index.json` sidecar together with a `schema_version`).  The model for new indexes is
taken from the server-wide setting (`PUT /similarity/default-model/setting`; default
`'signal'`).  Indexes with different models can coexist; the correct embedder is applied
automatically at search and add time.

**Body-level indexing (schema v2).**  New indexes store **one FAISS row per body**
(not one averaged vector per file).  This unlocks per-index statistics (files / bodies /
assemblies / single-part), `part` vs `assembly` classification, and per-body matching for
assembly-to-assembly search (`POST /index/{name}/search-assembly`).  Registering a file
also produces a thumbnail
(`thumbnails/<file_id>.png`) and a stream cache (`scs/<file_id>.scs`) from the same CAD
load.  Search runs through `CADSearch.search_by_shape()`, which embeds the query at body
granularity and applies the geometric reranker; the per-body ranked lists are then
concatenated in order and de-duplicated by file so each file appears once — the
`POST /index/{name}/search` response schema is unchanged.

**Search filtering.**  `POST /index/{name}/search` accepts `kind` (`any` | `part` |
`assembly`, default `any`) and `include_self` (default `false`).  `kind=any` passes no
filter at all, preserving the previous behaviour for existing clients; use **`kind=part`**
to reproduce the results of the sibling `hoops_ai_native_bridge` project, which always
filters to parts.  A query that is itself registered in the index is returned by the SDK
as a perfect self match; it is **removed by default** so that `top_k` hits are genuine
neighbours.  `include_self=true` keeps it instead, pinned first with score `1.0` (useful
for tagging a whole displayed cluster); the total still respects `top_k`.

Hit `metadata` contains only the keys stored at registration time (`file_id`, `filename`,
`registered_at`, `kind`, `bodies`, `thumbnail`, `scs`, `obb`).  Undocumented
underscore-prefixed fields added by the SDK are stripped from the response.

Legacy indexes created before this change are **schema v1** (one averaged vector per
file).  They remain fully readable (search / list / stats), but **writes**
(`POST /index/add`, `DELETE /index/{name}/parts`) return **409** with an instruction to
rebuild the index.  Migration tooling is provided separately.

Indexes are stored under ``APP_ROOT/indexes/<name>/``
(`index.faiss`, `index.meta`, `index.json`, `thumbnails/`, `scs/`).
Index names must match ``^[a-z0-9_-]{1,64}$``; ``default`` is reserved.

##### Incremental workflow example

```
# 0. (Optional) Switch active embedding model  Edefault is already 'signal'
PUT /similarity/default-model/setting?model=signal

# 1. Create an empty index  Euses the active model (signal by default)
POST /similarity/index/create?name=my-parts

# 2. Register parts (repeat as new parts arrive)  Emodel taken from index's model.json
POST /similarity/index/add?name=my-parts
     + files=@bracket_v1.step

# 3. Search the growing index
POST /similarity/index/my-parts/search
     + files=@new_bracket.step

# 4. Update a part (re-registering overwrites the old entry)
POST /similarity/index/add?name=my-parts
     + file_ids=<existing_file_id>

# 5. Remove a part
DELETE /similarity/index/my-parts/parts?part_ids=<file_id>

# 6. Delete the whole index (destructive  Erequires confirm=true)
DELETE /similarity/index/my-parts?confirm=true
```

##### Create a named index

```
POST /similarity/index/create?name=<name>
```

Returns **201** on success, **409** if the name already exists, **422** for invalid/reserved names.
The embeddings model is taken from the server-wide setting (`PUT /similarity/default-model/setting`).

| Parameter | Values | Description |
|---|---|---|
| `name` | `^[a-z0-9_-]{1,64}$` | Index name (required) |

**Windows (PowerShell):**
```powershell
curl.exe -X POST "http://127.0.0.1:8000/similarity/index/create?name=my-parts"
```

**Linux:**
```bash
curl -X POST "http://127.0.0.1:8000/similarity/index/create?name=my-parts"
```

**Response:**
```json
{ "name": "my-parts", "count": 0, "dim": 512, "model": "signal" }
```

---

##### List all indexes

```
GET /similarity/index/list
```

Returns all named indexes plus the built-in ``default`` index (``is_readonly: true``).

**Windows (PowerShell):**
```powershell
curl.exe "http://127.0.0.1:8000/similarity/index/list"
```

**Linux:**
```bash
curl "http://127.0.0.1:8000/similarity/index/list"
```

**Response:**
```json
[
  { "name": "default",          "count": 5000, "last_modified": "2025-06-01T12:00:00Z", "is_readonly": true,  "model": null      },
  { "name": "my-parts",         "count": 3,    "last_modified": "2026-07-01T08:30:00Z", "is_readonly": false, "model": "default" },
  { "name": "my-parts-signal",  "count": 3,    "last_modified": "2026-07-06T10:00:00Z", "is_readonly": false, "model": "signal"  }
]
```

---

##### Register parts in a named index

```
POST /similarity/index/add?name=<name>
```

Accepts the same three input sources as ``POST /similarity/compare``:

| Input | How to supply |
|---|---|
| Existing file IDs | `?file_ids=<id1>,<id2>,...` |
| CAD file uploads | `files` multipart field |
| ZIP archive | `zip_file` multipart field |

Re-registering a part ID overwrites the existing entry (``updated`` counter).
Embedding results are cached on disk  Ere-adding the same file is fast.

The embedder is always the one recorded in the index's `model.json` sidecar (set at creation time).

**Windows (PowerShell):**
```powershell
# Upload a new part directly
curl.exe -X POST "http://127.0.0.1:8000/similarity/index/add?name=my-parts" -F "files=@C:\path\to\new_bracket.step"

# Add an already-uploaded part by file_id
curl.exe -X POST "http://127.0.0.1:8000/similarity/index/add?name=my-parts" -F "" --data-urlencode "file_ids=a3f8c2..."
```

**Linux:**
```bash
curl -X POST "http://127.0.0.1:8000/similarity/index/add?name=my-parts" -F "files=@/path/to/new_bracket.step"
```

**Response:**
```json
{ "name": "my-parts", "added": 1, "updated": 0, "index_count": 4, "errors": [] }
```

---

##### Bulk registration as a background job

`POST /similarity/index/add` above is synchronous: the client holds a connection
open for the whole run and gets no progress and no way to stop. For hundreds or
thousands of files, use the job form instead.

```
POST   /similarity/index/{name}/jobs/add            → 202 {"job_id": ...}
GET    /similarity/index/{name}/jobs/{job_id}       → status + progress + summary
GET    /similarity/index/{name}/jobs/{job_id}/report → text/plain audit report
GET    /similarity/index/{name}/jobs                → list, newest first (paginated)
DELETE /similarity/index/{name}/jobs/{job_id}       → request cancellation
```

Request body (supply at least one input):

| Field | Meaning |
|---|---|
| `file_ids` | `file_id`s already in the CAD store. **The standard route** — upload with `POST /files/upload-batch` and skip what `POST /files/exists` already knows. |
| `zip_file_id` | `file_id` of an uploaded ZIP archive, expanded server-side. |
| `server_paths` | Paths on the server machine. Disabled by default; returns **403** unless `HOOPS_AI_ALLOW_SERVER_PATHS=true`. Administrator seeding only, since it reads any file the server process can read and bypasses the upload checks. |
| `workers` | Parallel embedding workers. Omit to size automatically from CPU cores and free RAM. |
| `time_limit` | Per-file embedding budget in seconds. Omit to use `HOOPS_AI_EMBED_TIME_LIMIT`, or the SDK's 120 s when that is unset. Files that exceed it come back with `retryable: true`. |

Job records are JSON files under `jobs/`, so they survive a restart. Each job
also gets a `jobs/<job_id>/` directory holding its report and the SDK's
`too_heavy_files.log`; it is deleted together with the record. Finished
records are swept after `HOOPS_AI_JOB_TTL_DAYS` (default 7) and capped at
`HOOPS_AI_JOB_MAX_RECORDS` (default 1000). Duplicate `file_id`s in one request
are collapsed; every rejected input is reported in `errors` rather than dropped.
A job that was still `queued` or `running` when the server stopped is marked
`failed` at the next startup — nothing resumes it, but everything it had already
registered is kept, so resubmitting the remainder is safe.

> **Jobs run one at a time by default.** Three reasons: each job spawns its own
> pool of embedding workers, every one of which loads the ~2 GB model
> checkpoint, so concurrent jobs multiply peak memory and end up slower in
> total; hoops_ai writes `error_summary.json` to the process working directory
> with no way to redirect it, so concurrent jobs would read each other's failure
> reasons; and the live-progress shim replaces the process-wide `sys.stderr`.
> Raise `HOOPS_AI_JOB_MAX_CONCURRENCY` only if you accept those.

> **Cancellation only takes effect before a job starts embedding.** The whole
> input goes to a single `embed_shape_batch` call, which cannot be interrupted,
> so cancelling a running job leaves it to finish. Split a large corpus across
> several jobs to keep the commit and cancellation granularity under your
> control.

### Live progress

`progress.done`, `progress.errors` and `progress.heavy` advance while the batch
runs, so polling the job shows real movement rather than `0/N` until the end.
The numbers come from hoops_ai's own tqdm bar: the server temporarily replaces
`sys.stderr` with a shim that mirrors every write to the server log and parses
the bar line, the same technique `hoops_ai_native_bridge` uses for its native
callers. The shim reports `isatty() == True`, because tqdm only emits
incremental updates when it believes it is writing to a terminal.
`progress.phase` is `embedding` while the batch runs, `heavy` while the SDK
works through the files it deferred to its single-worker RAM fallback, and
`done` at the end.

`progress.total` always stays the number of files you submitted, even if the
bar counts something else.

Because `sys.stderr` is process-global, only one job can be parsed at a time.
Jobs are serialised by default, so this is normally moot; if you raise
`HOOPS_AI_JOB_MAX_CONCURRENCY`, the extra jobs simply report no live progress
(a warning is logged) rather than reporting each other's counters.

> **One SDK call per job, deliberately.** Chunking a job into fixed-size batches
> was measured to cost ~55-60 s per chunk, because every chunk builds a fresh
> spawn-based worker pool in which each worker reloads the ~2 GB checkpoint. At
> 100 files per chunk that was about 30% of total runtime.

### Tuning bulk registration

Both knobs matter more than anything else here, and the right values depend on
the machine and on how heavy the CAD is.

`workers` — every worker keeps its own copy of the model, so peak memory grows
roughly linearly with the count while throughput reaches a plateau and then
declines once RAM-bound. Measured on `hoops_ai_native_bridge`: light single-body
parts plateau near the physical core count, heavy assemblies plateau lower (on a
16-core machine 12 was the peak, and 16/20/24 were each slower than 8). Aim for
the plateau, not a single "best" value — run-to-run noise exceeds the difference
between neighbouring counts. **An explicit `workers` is sent to the SDK
unchanged**; `HOOPS_AI_MAX_WORKERS` bounds only the automatic choice, which is
`min(HOOPS_AI_MAX_WORKERS, logical_cpus / 2, usable_RAM / HOOPS_AI_MODEL_FOOTPRINT_MB, file_count)`.

`time_limit` — a file that exceeds the budget is dropped with `Timeout` and
reported in `errors`. The default of 120 s is right for the many light files and
too short for heavy assemblies, so registering a corpus is a **two-pass client
loop**:

1. Submit everything with many `workers` and the default `time_limit`.
2. When the job finishes, collect the `file_id`s whose error carries
   `retryable: true`, and submit those as a second job with a much larger
   `time_limit` and a small `workers`.

Each `errors` entry carries `detail` (the reason as hoops_ai reported it) and
`retryable`. A failure is retryable when it timed out, or when no reason could
be determined at all — the safe direction, since a file that is merely slow gets
its second chance. Everything else is a deterministic CAD error
(`NoRootInModel`, a corrupt file, an unsupported entity) that will fail again
however long the budget, so retrying it only wastes the budget.

`detail` is taken from `batch.metadata["errors"]` where that names the file, and
otherwise from the `error_summary.json` hoops_ai writes next to the running
process, which lists each failed path with its reason. Both are needed: a
parallel run that times out reports reasons with no path in them, and in
completion rather than submission order, so only the summary file can attribute
them. This is the same source `hoops_ai_native_bridge`'s front end reads.

Use a small `workers` for the second job: a single heavy file is not made faster
by more workers, only across-file parallelism helps, and heavy assemblies need
far more memory per worker. `hoops_ai_native_bridge`'s front end sizes its heavy
pass as `round(free RAM / 4 GB)`.

The retry decision is deliberately the client's, not the server's: it controls
the commit granularity, and one job stays one embedding pass.

**Windows (PowerShell):**
```powershell
$body = @{ file_ids = @("a3f8c2...", "b71e94...") } | ConvertTo-Json -Compress
$body | Set-Content -Path "$env:TEMP\add.json" -Encoding utf8
$job = curl.exe -s -X POST "http://127.0.0.1:8000/similarity/index/my-parts/jobs/add" `
  -H "Content-Type: application/json" --data-binary "@$env:TEMP\add.json" | ConvertFrom-Json

# Poll until the job settles
do {
  Start-Sleep -Seconds 2
  $s = curl.exe -s "http://127.0.0.1:8000/similarity/index/my-parts/jobs/$($job.job_id)" | ConvertFrom-Json
  "{0} {1}/{2} errors={3}" -f $s.status, $s.progress.done, $s.progress.total, $s.progress.errors
} while ($s.status -in @("queued", "running"))

# Pass 2: resubmit only what a longer budget can still save
$retry = $s.errors | Where-Object { $_.retryable } | ForEach-Object { $_.file_id }
if ($retry) {
  @{ file_ids = [string[]]$retry; workers = 4; time_limit = 1200 } |
    ConvertTo-Json -Compress | Set-Content -Path "$env:TEMP\retry.json" -Encoding utf8
  curl.exe -s -X POST "http://127.0.0.1:8000/similarity/index/my-parts/jobs/add" `
    -H "Content-Type: application/json" --data-binary "@$env:TEMP\retry.json"
}
```

**Response (status):**
```json
{
  "job_id": "…", "kind": "index_add", "index_name": "my-parts", "status": "done",
  "progress": {"phase": "done", "done": 240, "total": 240, "errors": 3, "heavy": 5},
  "summary": {"added": 235, "updated": 2, "failed": 3, "skipped": 0,
              "index_count": 1042, "retryable": 2, "heavy_flagged": 5, "report": true},
  "errors": [{"file_id": "…", "detail": "Timeout (CUMULATIVE)", "retryable": true}],
  "timings": {"num_workers": 8, "time_limit": 0, "embed_seconds": 812.4},
  "report_url": "/similarity/index/my-parts/jobs/…/report"
}
```

`heavy_flagged` counts the files hoops_ai deferred to its own single-worker RAM
fallback. They are usually embedded successfully, just slowly — this is what
tells you which parts made a run expensive.

`report_url` appears once the job has written its report and serves a plain-text
audit in three groups: `[ADDED]`, `[FAILED]` (each line marked `RETRYABLE` or
`PERMANENT`, with the reason), and `[HEAVY-FLAGGED]`.

`timings` is recorded so tuning decisions can be based on this server's own
measurements rather than guesswork.

---

##### Search a named index

```
POST /similarity/index/{name}/search?top_k=<n>&kind=<any|part|assembly>&include_self=<bool>&include_image=<bool>&tags=<a,b>&tag_mode=<any|all>&tagged=<bool>
```

Supply **either** a file upload or a `file_id`.  Returns an empty ``hits`` list
when the index contains zero entries (no error).

| Query param | Default | Description |
|---|---|---|
| `top_k` | `10` | Maximum number of hits. |
| `kind` | `any` | Restrict hits to `part` or `assembly`. `any` passes no filter. Use `part` to match `hoops_ai_native_bridge`. Any other value is **422**. |
| `include_self` | `false` | Keep the query itself in the results when it is registered in this index, pinned first with score `1.0`. By default the self match is dropped. Never exceeds `top_k`. |
| `include_image` | `true` | When `false`, skip the result-grid PNG and return `image_url: null`. Hits are unaffected. |
| `tags` / `tag_mode` / `tagged` | — | Keep only hits carrying the given tags. See [Filtering searches and listings by tag](#filtering-searches-and-listings-by-tag) — a filtered search may return fewer than `top_k` hits. |

The result-grid PNG is a preview: it shows at most **12 tiles** regardless of
`top_k`, captioned `top 12 of 300`.  The full ranking is always in `hits`.
Rendering one tile costs a PNG decode plus a matplotlib subplot, so an uncapped
sheet made the endpoint scale linearly with `top_k` (39 s of a 40 s request at
`top_k=300`).  Clients that draw their own gallery should pass
`include_image=false`, which also skips the CAD reload used to render the query
thumbnail.

**Windows (PowerShell):**
```powershell
curl.exe -X POST "http://127.0.0.1:8000/similarity/index/my-parts/search?top_k=5" -F "file=@C:\path\to\query.step"

# Parts only (bridge-compatible), including the query itself
curl.exe -X POST "http://127.0.0.1:8000/similarity/index/my-parts/search?top_k=5&kind=part&include_self=true" -F "file=@C:\path\to\query.step"

# Large result set without the preview image (fastest)
curl.exe -X POST "http://127.0.0.1:8000/similarity/index/my-parts/search?top_k=300&kind=part&include_image=false" -F "file=@C:\path\to\query.step"
```

**Linux:**
```bash
curl -X POST "http://127.0.0.1:8000/similarity/index/my-parts/search?top_k=5" -F "file=@/path/to/query.step"
```

**Response:**
```json
{
  "hits": [
    { "id": "<file_id>", "score": 0.987, "metadata": { "filename": "bracket_v1.step", "registered_at": "2026-07-01T08:30:00Z" } }
  ],
  "count": 1
}
```

---

##### Assembly-to-assembly search

```
POST /similarity/index/{name}/search-assembly?top_k=<n>&coverage_mode=<symmetric|containment|jaccard>&tags=<a,b>&tag_mode=<any|all>&tagged=<bool>
```

Finds **assemblies similar to a query assembly**, as opposed to
`/index/{name}/search`, which ranks individual parts.  Query bodies are paired
one-to-one with candidate bodies by Hungarian assignment over the per-body
embeddings, and the geometric result is blended with a bag-of-parts composition
score.  Requires a **schema v2** (body-level) index.

The algorithm is the tutorials' `AssemblyMatcher`, vendored verbatim into
`vendor/assembly_matcher.py` (see *Vendored assembly matcher* below).  The fixed
arguments (`method="hungarian"`, `candidate_mode="search"`,
`assemblies_only=True`, `reuse_index_vectors=True`) and every default match
`hoops_ai_native_bridge`, so both projects rank the same corpus identically.

Supply **either** a file upload or a `file_id`.  Returns an empty `hits` list
when the index contains zero entries.  A **schema v1** (legacy, one averaged
vector per file) index returns **409**: it has no per-body vectors, so there is
nothing to match one-to-one — rebuild it as v2 first.

| Query param | Default | Description |
|---|---|---|
| `top_k` | `10` | Maximum number of hits. |
| `candidate_k` | `30` | Per-body shortlist size used to gather candidates (1–1000). Higher = better recall, slower. |
| `sim_thresh` | `0.80` | Minimum body-to-body cosine similarity for a pair to count as matched (0.0–1.0). |
| `bop_weight` | `0.30` | Blend weight of the bag-of-parts composition score against the geometric score (0.0–1.0). `0` = geometry only. |
| `coverage_mode` | `symmetric` | Coverage denominator: `symmetric` (larger side), `containment` (query side — "which assemblies contain this one?"), `jaccard` (union). |
| `use_idf` | `true` | Weight parts by cluster rarity so common fasteners contribute less. |
| `include_self` | `false` | Keep the query itself, pinned first with score `1.0`. Dropped by default. Never exceeds `top_k`. |
| `include_image` | `true` | When `false`, skip the result-grid PNG and return `image_url: null`. Hits are unaffected. |
| `tags` / `tag_mode` / `tagged` | — | Keep only hits carrying the given tags. See [Filtering searches and listings by tag](#filtering-searches-and-listings-by-tag) — a filtered search may return fewer than `top_k` hits. |

Out-of-range values return **422**.

Each hit reports the diagnostics behind its score, which is what makes the
ranking explainable:

| Field | Meaning |
|---|---|
| `score` | Final blended score (geometry + composition). |
| `geom_score` | Match quality × rarity-aware coverage. |
| `coverage` | Matched mass over the `coverage_mode` denominator. |
| `matched_parts` | Number of matched body pairs. |
| `candidate_parts` | Bodies in the candidate assembly. |
| `query_parts` | Bodies in the query assembly. |

**Cost.**  The first request against an index builds an `AssemblyMatcher`,
which runs a FAISS k-means over *every* body vector in the corpus to derive the
rarity weights.  Measured at **19.3 s** for 42,098 bodies clustered into 4,096
centroids (`k = min(max(64, N/8), 4096)`); the build time is logged as
`[ASSEMBLY] built matcher for index '<name>' in <n>s`.  The instance is cached
by (index path, mtime) — a single entry, because it holds a normalised copy of
the whole corpus — and is rebuilt automatically after any write to the index.

That cost is paid by whichever request arrives first after a restart or an
index write, so **the first assembly search on a large index takes ~20 s while
later ones take well under a second**.  Warm it deliberately after indexing if
that latency is user-visible.  Stage-2 scoring runs on a thread pool sized by
`HOOPS_AI_ASSEMBLY_SEARCH_JOBS` (default `8`); lower it if assembly searches
starve the server's request workers.

**Registered queries skip re-embedding.**  A query that is already in the index
is handed to the matcher by its record id (the `file_id`) rather than by path,
so `reuse_index_vectors=True` reuses the stored per-body vectors instead of
re-reading and re-embedding the CAD file — a warm search drops from ~1.16 s to
~0.16 s.  It also lets the matcher's own `candidates.discard(query_path)` work.
A query that is *not* indexed is passed as a path and embedded on the fly, as
before.

**On matching `hoops_ai_native_bridge` exactly.**  Assembly scores are close to
but not bit-identical with the bridge on the same corpus (0.6499 vs 0.6514 for
the reference query, 100 vs 103 candidates out of 2,467 assemblies).  The
residual sits in the IDF term: with `use_idf=false&bop_weight=0` the top hits
collapse to an exact tie and the hit count does not move.  The corpus and the
`.faiss` file are identical, so the k-means input is the same; the likely cause
is the FAISS OpenMP thread count changing the float reduction order during
clustering, which differs between a desktop process and a uvicorn worker.  The
top set and the meaningful ordering agree, which is the useful guarantee for a
similarity search — unlike part search, assembly search has a clustering step in
the middle and exact reproducibility across processes is not expected.

**Windows (PowerShell):**
```powershell
curl.exe -X POST "http://127.0.0.1:8000/similarity/index/my-parts/search-assembly?top_k=5" -F "file=@C:\path\to\assembly.step"

# "Which assemblies contain this sub-assembly?"
curl.exe -X POST "http://127.0.0.1:8000/similarity/index/my-parts/search-assembly?top_k=5&coverage_mode=containment" -F "file=@C:\path\to\assembly.step"

# Geometry only, no preview image
curl.exe -X POST "http://127.0.0.1:8000/similarity/index/my-parts/search-assembly?top_k=20&bop_weight=0&include_image=false" -F "file=@C:\path\to\assembly.step"
```

**Linux:**
```bash
curl -X POST "http://127.0.0.1:8000/similarity/index/my-parts/search-assembly?top_k=5" -F "file=@/path/to/assembly.step"
```

**Response:**
```json
{
  "hits": [
    {
      "id": "<file_id>",
      "score": 0.8123,
      "geom_score": 0.8642,
      "coverage": 0.75,
      "matched_parts": 7,
      "candidate_parts": 9,
      "query_parts": 8,
      "metadata": { "filename": "gearbox_v2.step", "kind": "assembly", "bodies": 9 }
    }
  ],
  "count": 1,
  "image_url": "http://127.0.0.1:8000/out/2f1c....png"
}
```

###### Vendored assembly matcher

`vendor/assembly_matcher.py` is a **verbatim copy** of
`HOOPS-AI-tutorials/embeddings_pipeline/assembly_matcher.py` — the same source
`hoops_ai_native_bridge` embeds in `assembly_matcher_py.h`.  Keeping a copy
(rather than a re-implementation) guarantees both projects run identical
scoring.  It carries a generated header with the source path, sync timestamp and
source SHA-256, and **must not be hand-edited**: fix the tutorial copy and
re-sync.

```bash
python tools/sync_assembly_matcher.py                 # copy + refresh the header
python tools/sync_assembly_matcher.py --check         # verify only; exit 1 on drift
python tools/sync_assembly_matcher.py --source <path> # non-default source location
```

The default source is `../HOOPS-AI-tutorials/embeddings_pipeline/assembly_matcher.py`,
i.e. the tutorials repository cloned next to this one.  The matcher needs
**scipy** (`scipy.optimize.linear_sum_assignment`), which is listed in
`requirements.txt`.

---

##### Remove parts from a named index

```
DELETE /similarity/index/{name}/parts?part_ids=<id1>,<id2>,...
```

**Windows (PowerShell):**
```powershell
curl.exe -X DELETE "http://127.0.0.1:8000/similarity/index/my-parts/parts?part_ids=a3f8c2...,cd34ef..."
```

**Linux:**
```bash
curl -X DELETE "http://127.0.0.1:8000/similarity/index/my-parts/parts?part_ids=a3f8c2...,cd34ef..."
```

**Response:**
```json
{ "name": "my-parts", "removed": 2, "index_count": 2 }
```

---

##### Delete a named index

```
DELETE /similarity/index/{name}?confirm=true
```

Destructive and irreversible.  Requires ``?confirm=true``; without it returns **409** with
an instruction.  Returns **403** for the read-only ``default`` index.

**Windows (PowerShell):**
```powershell
curl.exe -X DELETE "http://127.0.0.1:8000/similarity/index/my-parts?confirm=true"
```

**Linux:**
```bash
curl -X DELETE "http://127.0.0.1:8000/similarity/index/my-parts?confirm=true"
```

**Response:**
```json
{ "name": "my-parts", "deleted": true }
```

##### Index statistics

```
GET /similarity/index/{name}/stats
```

Body-level counts for a named index.  Works for both schema versions (a legacy index
reports one body per file, so `bodies == files` and `assemblies == 0`).

**Windows (PowerShell):**
```powershell
curl.exe "http://127.0.0.1:8000/similarity/index/my-parts/stats"
```

**Linux:**
```bash
curl "http://127.0.0.1:8000/similarity/index/my-parts/stats"
```

**Response:**
```json
{
  "name": "my-parts",
  "files": 9,
  "bodies": 47,
  "assemblies": 7,
  "single_part": 2,
  "dim": 2048,
  "model": "signal",
  "schema_version": 2,
  "last_modified": "2026-08-17T06:00:00Z"
}
```

##### List registered parts (paginated)

```
GET /similarity/index/{name}/parts?offset=0&limit=100&kind=part|assembly&tags=a,b&tag_mode=any|all&tagged=true|false
```

One item per file.  `limit` is clamped to `1..2000` (default `100`); `offset` is `>= 0`.
Omit `kind` for all items, or filter by `part` / `assembly`.  `thumbnail_url` / `scs_url`
are absolute URLs to the asset endpoints below, or `null` when the asset is missing.
Every item carries its `tags` (an empty list when untagged); the tag filters are
described under [Index tags](#index-tags) and are applied **before** paging, so
`total` is the size of the filtered set.

**Linux:**
```bash
curl "http://127.0.0.1:8000/similarity/index/my-parts/parts?limit=50&kind=assembly"
```

**Response:**
```json
{
  "total": 9,
  "offset": 0,
  "limit": 50,
  "items": [
    {
      "id": "<file_id>",
      "filename": "1005.stp",
      "kind": "assembly",
      "bodies": 12,
      "thumbnail_url": "http://127.0.0.1:8000/similarity/index/my-parts/parts/<file_id>/thumbnail",
      "scs_url": "http://127.0.0.1:8000/similarity/index/my-parts/parts/<file_id>/scs",
      "registered_at": "2026-08-17T06:00:00Z",
      "tags": ["bracket", "revision-b"]
    }
  ]
}
```

##### Index tags

Free-form labels attached to registered parts, so a shared index can be
organised by hand: `bracket`, `revision-b`, `do-not-reuse`.  Tags are metadata
only — they never change an embedding, a score or a ranking, they only decide
which results you are shown.

```
GET    /similarity/index/{name}/tags                     → every tag with its part count
GET    /similarity/index/{name}/tags/{tag}/parts         → the part ids carrying one tag
GET    /similarity/index/{name}/parts/{part_id}/tags     → the tags of one part
PUT    /similarity/index/{name}/parts/{part_id}/tags     → replace the tags of one part
POST   /similarity/index/{name}/tags/{tag}/parts         → add one tag to many parts
DELETE /similarity/index/{name}/tags/{tag}/parts         → remove one tag from many parts
DELETE /similarity/index/{name}/tags/{tag}?confirm=true  → remove one tag from every part
```

Tags live in `indexes/<name>/tags.json`, beside the index rather than inside it.
That is a measured decision, not a preference: `FaissVectorStore` does accept an
arbitrary metadata key and does keep it across `save()`/`load()`, but persisting
one costs a **full rewrite of `.faiss`** (0.22 s / 84 MB at 42 k rows) and moves
its mtime.  Both the part searcher and the assembly matcher are cached on
`(path, mtime)`, so storing tags in the index would force a **19.3 s matcher
rebuild after every single tag edit** — unusable for an interactive "tag the
whole cluster" gesture.  The sidecar costs a few kB per write and touches
neither cache.

Because the key is the `file_id` — the SHA-256 of the CAD file's contents — tags
survive rebuilding the index from scratch, and the same file re-registered under
a different name keeps its tags.  Removing a part from the index prunes its
entry, so the sidecar cannot outgrow the index.

**Rules.** A tag is trimmed, must be non-empty, at most 64 characters, must not
contain control characters, and must not contain `/` or `\` because it appears
as a URL path segment.  Tags are case-sensitive (`Bracket` and `bracket` are two
tags); a case-only collision within an index is logged as a warning rather than
merged, since guessing which spelling was meant would silently lose one of them.
A part may carry `HOOPS_AI_MAX_TAGS_PER_PART` tags (default 32).  A violation is
**422**, an unknown index is **404**, and an unregistered `part_id` is rejected
rather than tagged.

**Concurrent edits.** Every read returns an `ETag` for the whole tag document.
Send it back as `If-Match` on a write and a **412** tells you someone edited the
file since you read it; re-read and retry.  Sending no `If-Match` is allowed and
means last-write-wins, which will discard a colleague's concurrent edit without
saying so — pass it whenever a human is driving.  `X-Client-Id` is recorded as
`updated_by` so the audit trail names a workstation instead of nobody.  Bulk
calls report every id they could not apply in `skipped`, with a reason, instead
of failing the whole request.

**Windows (PowerShell):**
```powershell
$base = "http://127.0.0.1:8000/similarity/index/my-parts"

# Read the current tags and the ETag that goes with them
$r = curl.exe -s -D - "$base/tags" -o "$env:TEMP\tags.json"
$etag = ($r | Select-String '^ETag:').Line.Split(' ')[1].Trim()

# Replace one part's tags, refusing to clobber a concurrent edit
'{"tags":["bracket","revision-b"]}' | Set-Content "$env:TEMP\t.json" -Encoding utf8
curl.exe -s -X PUT "$base/parts/<file_id>/tags" -H "Content-Type: application/json" `
  -H "If-Match: $etag" -H "X-Client-Id: $env:COMPUTERNAME" --data-binary "@$env:TEMP\t.json"

# Tag a whole cluster at once
'{"part_ids":["<id1>","<id2>","<id3>"]}' | Set-Content "$env:TEMP\b.json" -Encoding utf8
curl.exe -s -X POST "$base/tags/bracket/parts" -H "Content-Type: application/json" `
  --data-binary "@$env:TEMP\b.json"

# Retire a tag everywhere (409 without confirm=true)
curl.exe -s -X DELETE "$base/tags/bracket?confirm=true"
```

**Linux:**
```bash
curl "http://127.0.0.1:8000/similarity/index/my-parts/tags"
curl -X POST "http://127.0.0.1:8000/similarity/index/my-parts/tags/bracket/parts" \
  -H "Content-Type: application/json" -d '{"part_ids":["<id1>","<id2>"]}'
```

**Response (`GET /tags`):**
```json
{ "tags": [ { "tag": "bracket", "count": 42 }, { "tag": "revision-b", "count": 7 } ] }
```

##### Filtering searches and listings by tag

The same three parameters work on `GET .../parts`, `POST .../search` and
`POST .../search-assembly`:

| Query param | Default | Description |
|---|---|---|
| `tags` | — | Comma-separated tag list. |
| `tag_mode` | `any` | `any` keeps a part carrying at least one of them, `all` requires every one. Any other value is **422**. |
| `tagged` | — | `true` keeps only tagged parts, `false` only untagged. Combines with `tags`. |

> **A tag-filtered search can return fewer than `top_k` hits.** CADSearch knows
> nothing about tags, so the filter is applied **after** it has ranked — the
> alternative, re-ranking a tag-restricted subset, would change the scores and
> break parity with `hoops_ai_native_bridge`. To compensate, a filtered request
> asks the index for `top_k × 5` candidates (capped at 200) and keeps the ones
> that match. When a tag is rare, that pool can still run out. Ask for a larger
> `top_k`, or list by tag with `GET .../parts?tags=…` when you want completeness
> rather than ranking. The self hit added by `include_self` is filtered too, so
> a response never contains a part that fails your filter.

```powershell
curl.exe -X POST "$base/search?top_k=5&tags=bracket,revision-b&tag_mode=all" -F "file=@C:\path\to\query.step"
curl.exe "$base/parts?tags=bracket&limit=500"
curl.exe "$base/parts?tagged=false&limit=500"    # what still needs organising
```

##### Importing tags from the Qt sandbox

`tools/import_qt_tags.py` converts a `hoops_ai_qt_sandbox` tag sidecar into this
server's `tags.json`.  The Qt file keys parts by **absolute path**, this server
by content hash, so the tool re-hashes every referenced CAD file — the files
must still be reachable at the recorded paths.

```powershell
C:\SDK\HOOPS_AI\V1.1\.venv\Scripts\python.exe tools\import_qt_tags.py `
  C:\path\to\qt_index.json --index my-parts --dry-run
```

`--dry-run` prints what would change and writes nothing.  Add `--merge` to keep
existing tags instead of replacing them, and `--client-id` to stamp `updated_by`.
Files that are missing, unregistered in the target index, or carrying an invalid
tag are skipped with a reason rather than aborting the import; without `--force`
the tool refuses to overwrite a non-empty `tags.json`.

##### Part thumbnail / stream cache

```
GET /similarity/index/{name}/parts/{part_id}/thumbnail   -> image/png
GET /similarity/index/{name}/parts/{part_id}/scs         -> application/octet-stream
```

Serve the per-part PNG thumbnail or SCS stream cache generated at registration time.
Returns **404** when the asset does not exist.  Paths are validated to stay within
`indexes/<name>/`.

**Linux:**
```bash
curl -o thumb.png "http://127.0.0.1:8000/similarity/index/my-parts/parts/<file_id>/thumbnail"
```

---

Classify a CAD solid into one of 45 part categories (FabWave dataset) using a trained Graph Classification model.

#### Run inference

Upload a CAD file and classify it into one of the 45 part categories. Returns the top-k predictions with class ID, part name, and confidence (%).

```
POST /part-classification/predict?top_k=5
```

**Windows (PowerShell):**
```powershell
curl.exe -X POST "http://127.0.0.1:8000/part-classification/predict?top_k=5" -F "file=@C:\path\to\model.stp"
```

**Linux:**
```bash
curl -X POST "http://127.0.0.1:8000/part-classification/predict?top_k=5" -F "file=@/path/to/model.stp"
```

**Reuse an uploaded file by `file_id`:**
```powershell
# Windows
curl.exe -X POST "http://127.0.0.1:8000/part-classification/predict?file_id=<file_id>&top_k=5"
```
```bash
# Linux
curl -X POST "http://127.0.0.1:8000/part-classification/predict?file_id=<file_id>&top_k=5"
```

**Response:**

```json
{
  "predicted_class_id": 30,
  "predicted_part_name": "Gears",
  "top_predictions": [
    {"rank": 1, "class_id": 30, "part_name": "Gears",       "confidence": 87},
    {"rank": 2, "class_id": 32, "part_name": "Idler Sprocket", "confidence": 8},
    {"rank": 3, "class_id": 34, "part_name": "Miter Gears", "confidence": 3},
    {"rank": 4, "class_id": 29, "part_name": "Gear Rod Stock", "confidence": 1},
    {"rank": 5, "class_id": 33, "part_name": "Miter Gear Set Screw", "confidence": 1}
  ]
}
```

#### List all part labels

Returns the full 45-class label dictionary.

```
GET /part-classification/labels
```

**Windows (PowerShell):**
```powershell
curl.exe "http://127.0.0.1:8000/part-classification/labels"
```

**Linux:**
```bash
curl "http://127.0.0.1:8000/part-classification/labels"
```

#### Dataset table of contents

```
GET /part-classification/dataset/table-of-contents
```

**Windows (PowerShell):**
```powershell
curl.exe "http://127.0.0.1:8000/part-classification/dataset/table-of-contents"
```

**Linux:**
```bash
curl "http://127.0.0.1:8000/part-classification/dataset/table-of-contents"
```

#### Per-class file count distribution

```
GET /part-classification/dataset/label-distribution
```

**Windows (PowerShell):**
```powershell
curl.exe "http://127.0.0.1:8000/part-classification/dataset/label-distribution"
```

**Linux:**
```bash
curl "http://127.0.0.1:8000/part-classification/dataset/label-distribution"
```

**Response:**
```json
{
  "label_key": "part_label",
  "bins": [
    {"class_id": 0, "part_name": "Bearings", "bin_start": 0.0, "bin_end": 1.0, "file_count": 42},
    {"class_id": 1, "part_name": "Bolts",    "bin_start": 1.0, "bin_end": 2.0, "file_count": 38}
  ]
}
```

#### List files for a class

Returns the file IDs in the dataset that belong to a specific class.

```
GET /part-classification/dataset/files?label_id=<0-44>
```

**Windows (PowerShell):**
```powershell
curl.exe "http://127.0.0.1:8000/part-classification/dataset/files?label_id=30"
```

**Linux:**
```bash
curl "http://127.0.0.1:8000/part-classification/dataset/files?label_id=30"
```

#### Dataset thumbnail preview

Returns the URL of a PNG grid of dataset thumbnails for a given class (same pattern as `/similarity/search`).

```
GET /part-classification/dataset/preview?label_id=<0-44>&k=25&grid_cols=8
```

**Windows (PowerShell):**
```powershell
curl.exe "http://127.0.0.1:8000/part-classification/dataset/preview?label_id=30&k=25"
```

**Linux:**
```bash
curl "http://127.0.0.1:8000/part-classification/dataset/preview?label_id=30&k=25"
```

**Response:**

```json
{
  "label_id": 30,
  "part_name": "Gears",
  "image_url": "http://127.0.0.1:8000/out/<uuid>.png"
}
```

Open `image_url` in a browser to view the thumbnail grid.

> **Note:** Thumbnails are rendered from the `stream_cache/` folder inside the flow directory. This folder is populated when running the ETL step of `3c_workflow_for_Part_classification_fabwave.ipynb`. If `stream_cache/` is empty, the image grid will show "No Preview" placeholders.

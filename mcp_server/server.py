import httpx
from pathlib import Path
from mcp.server.fastmcp import FastMCP

API_BASE = "http://127.0.0.1:8001"

mcp = FastMCP("HOOPS AIサーバー")

@mcp.tool()
def open_cad_viewer(cad_file_path: str) -> str:
    """
    Open a CAD file in CADViewer and return the generated viewer URL.
    """
    source_path = Path(cad_file_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"CAD file not found: {source_path}")
    if not source_path.is_file():
        raise ValueError(f"Path is not a file: {source_path}")

    response = httpx.post(
        f"{API_BASE}/CAD/viewer/from-path",
        data={"cad_file_path": str(source_path)},
        timeout=120,
    )
    response.raise_for_status()

    data = response.json()
    viewer_url = data.get("viewer_url")
    if not viewer_url:
        raise RuntimeError(f"Viewer URL was not returned: {data}")

    return viewer_url

@mcp.tool()
def get_MFR_table_of_contents():
    """
    HOOPS AIサーバーのManufacturing Feature Recognition (MFR) データベースの目次を取得する。
    """

    response = httpx.get(f"{API_BASE}/MFR/dataset/table-of-contents")
    return response.json()

@mcp.tool()
def get_MFR_labels_description():
    """Return MFR label IDs, feature names, and descriptions."""

    response = httpx.get(f"{API_BASE}/MFR/labels/description")
    return response.json()

@mcp.tool()
def search_MFR_files(
        feature_name: str,
    ):
    """
    Return CAD file names and file IDs that contain the requested MFR feature name.
    Response includes file_names (list of strings) and file_list (list of int file IDs).
    """

    params = {}
    if feature_name:
        params["feature_name"] = feature_name

    response = httpx.get(f"{API_BASE}/MFR/files/search", params=params)
    return response.json()


if __name__ == "__main__":
    mcp.run(transport="stdio")
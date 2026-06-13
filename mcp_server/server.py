import httpx
from pathlib import Path
from mcp.server.fastmcp import FastMCP

API_BASE = "http://127.0.0.1:8001"

mcp = FastMCP("HOOPS AIサーバー")

@mcp.tool()
def open_cad_viewer(cad_file_path: str) -> str:
    """
    Open a local CAD file in CADViewer and return the generated viewer URL.
    The file is uploaded from the client machine to the server.
    """
    source_path = Path(cad_file_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"CAD file not found: {source_path}")
    if not source_path.is_file():
        raise ValueError(f"Path is not a file: {source_path}")

    with source_path.open("rb") as f:
        response = httpx.post(
            f"{API_BASE}/CAD/viewer",
            files={"file": (source_path.name, f, "application/octet-stream")},
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
    """Return a summary table of contents for the MFR dataset."""

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


@mcp.tool()
def get_MFR_file_thumbnail(file_id: int) -> str:
    """
    Download the thumbnail PNG image for a given file ID and return it as a base64-encoded string.
    """
    import base64

    response = httpx.get(f"{API_BASE}/MFR/files/{file_id}/thumbnail", timeout=60)
    response.raise_for_status()
    return base64.b64encode(response.content).decode("utf-8")


@mcp.tool()
def run_MFR_inference(cad_file_path: str) -> dict:
    """
    Run MFR inference on a local CAD file and launch the CAD viewer.
    Uploads the file to the server and returns predictions, probabilities, and viewer_url.
    Call colorize_MFR_viewer() after the viewer has loaded to apply face colors.
    """
    source_path = Path(cad_file_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"CAD file not found: {source_path}")

    with source_path.open("rb") as f:
        response = httpx.post(
            f"{API_BASE}/MFR/inference",
            files={"file": (source_path.name, f, "application/octet-stream")},
            timeout=300,
        )
    response.raise_for_status()
    return response.json()


@mcp.tool()
def colorize_MFR_viewer() -> dict:
    """
    Apply MFR prediction colors to the last active CAD viewer.
    Must be called after run_MFR_inference() and after the viewer has fully loaded.
    Returns color_map: {label_id: {name, color_rgb}}.
    """
    response = httpx.post(f"{API_BASE}/MFR/viewer/colorize", timeout=120)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    mcp.run(transport="stdio")
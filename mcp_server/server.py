import httpx
from pathlib import Path
from mcp.server.fastmcp import FastMCP

API_BASE = "http://127.0.0.1:8001"

mcp = FastMCP("HOOPS AI MCP Server")

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

    After calling this function:
    1. Share the viewer_url with the user and ask them to open it in a browser.
    2. Wait for the user to confirm that the 3D model has fully loaded in the viewer.
    3. Only call colorize_MFR_viewer() after receiving confirmation from the user.
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

    This function must be called only after the user has confirmed that the 3D model
    has fully loaded in the browser viewer opened by run_MFR_inference().
    Do NOT call this automatically — always wait for explicit instruction from the user.

    Returns color_map: {label_id: {name, color_rgb}}.
    """
    response = httpx.post(f"{API_BASE}/MFR/viewer/colorize", timeout=120)
    response.raise_for_status()
    return response.json()


@mcp.tool()
def terminate_CAD_viewer(terminate_all: bool = False) -> dict:
    """
    Terminate the CAD viewer.
    - terminate_all=False (default): terminate only the last active viewer.
    - terminate_all=True: terminate all active viewers.
    Returns the number of viewers terminated.
    """
    params = {"all": "true"} if terminate_all else {}
    response = httpx.delete(f"{API_BASE}/CAD/viewer", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


@mcp.tool()
def get_brep_adjacency_graph(cad_file_path: str) -> dict:
    """
    Build a face adjacency graph from the B-rep model of a local CAD file.
    Nodes are faces; edges connect adjacent faces.
    Returns graph data (nodes, edges, counts) and a base64-encoded PNG visualization.
    """
    source_path = Path(cad_file_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"CAD file not found: {source_path}")

    with source_path.open("rb") as f:
        response = httpx.post(
            f"{API_BASE}/BRep/adjacency-graph",
            files={"file": (source_path.name, f, "application/octet-stream")},
            timeout=120,
        )
    response.raise_for_status()
    return response.json()


@mcp.tool()
def get_brep_attributes(cad_file_path: str) -> dict:
    """
    Extract face and edge attributes from the B-rep model of a local CAD file.
    Returns:
    - faces: types, areas, centroids, loops, types_description
    - edges: types, lengths, dihedrals, convexities, types_description
    """
    source_path = Path(cad_file_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"CAD file not found: {source_path}")

    with source_path.open("rb") as f:
        response = httpx.post(
            f"{API_BASE}/BRep/attributes",
            files={"file": (source_path.name, f, "application/octet-stream")},
            timeout=120,
        )
    response.raise_for_status()
    return response.json()


@mcp.tool()
def search_similar_shapes(cad_file_path: str, top_k: int = 10) -> dict:
    """
    Search for similar CAD shapes using HOOPS Embeddings and a FAISS index.
    Uploads a local CAD file and returns the top-k most similar shapes from the indexed database.
    Each hit contains an id (file identifier in the database) and a similarity score.
    Also returns image_url: a URL path to a PNG grid image of the search results
    (e.g. http://127.0.0.1:8001/out/<uuid>.png).
    """
    source_path = Path(cad_file_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"CAD file not found: {source_path}")

    with source_path.open("rb") as f:
        response = httpx.post(
            f"{API_BASE}/similarity/search",
            files={"file": (source_path.name, f, "application/octet-stream")},
            params={"top_k": top_k},
            timeout=300,
        )
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    mcp.run(transport="stdio")
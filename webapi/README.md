# HOOPS AI WebAPI

A FastAPI-based REST API that exposes [HOOPS AI](https://www.techsoft3d.com/developers/products/hoops-ai/) (Tech Soft 3D) capabilities as HTTP endpoints.  
See the root [README](../README.md) for an overview of the full HOOPS AI MCP platform.

---

## Requirements

- Python 3.12
- A valid **HOOPS AI license key**
- HOOPS AI (CPU or GPU version) installed in the environment
- **HOOPS AI Tutorials**  Ethe notebooks folder and its contents (ML datasets and pre-trained models) are required to run this server.  
  The tutorials are available at [github.com/techsoft3d/HOOPS-AI-tutorials](https://github.com/techsoft3d/HOOPS-AI-tutorials/tree/main).  
  Data packages (datasets and trained model checkpoints) must be obtained from the Tech Soft 3D File Transfer service by following the HOOPS AI installation instructions.

---

## Setup

### 1. Install dependencies

```bash
cd webapi
pip install -r requirements.txt
```

> Install HOOPS AI (CPU or GPU version) separately according to your HOOPS AI distribution instructions.

### 2. Configure environment variables

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
| `HOOPS_AI_NOTEBOOK_DIR` | ✁E| Absolute path to your HOOPS AI notebooks directory |
| `HOOPS_AI_MFR_FLOW_NAME` | optional | MFR flow name (dataset files are resolved relative to this) |
| `HOOPS_AI_MFR_MODEL_NAME` | optional | MFR trained model checkpoint filename (e.g. `ts3d_162k_mfr.ckpt`) |
| `HOOPS_AI_EMBEDDINGS_MODEL_NAME` | optional | Embeddings trained model checkpoint filename (e.g. `ts3d_1M_hoops_embeddings.ckpt`) |
| `HOOPS_AI_FAISS_INDEX_PATH` | optional | FAISS index file for shape similarity search (e.g. `fabwave_embeddings_store.faiss`) |

> **Note:** `HOOPS_AI_LICENSE` is read **only** from the `.env` file, not from system environment variables.

Example `.env`:

```
HOOPS_AI_LICENSE=XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
HOOPS_AI_NOTEBOOK_DIR=C:\hoops_ai\notebooks
HOOPS_AI_MFR_FLOW_NAME=cadsynth_1000
HOOPS_AI_MFR_MODEL_NAME=ts3d_162k_mfr.ckpt
HOOPS_AI_EMBEDDINGS_MODEL_NAME=ts3d_1M_hoops_embeddings.ckpt
HOOPS_AI_FAISS_INDEX_PATH=fabwave_embeddings_store.faiss
```

### 3. Start the server

Run from the `webapi/` directory using the Python executable from your HOOPS AI virtual environment.

**Windows:**

```bat
cd webapi
<Path\to\HOOPS_AI\install\dir>\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001
```

**Linux:**

```bash
cd webapi
/path/to/HOOPS_AI/install/dir/.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8001
```

> Replace the path prefix with the actual directory where HOOPS AI is installed.
> The venv Python executable ensures HOOPS AI packages from that environment are used.

For development with auto-reload:

**Windows:**

```bat
cd webapi
<Path\to\HOOPS_AI\install\dir>\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001 --reload
```

**Linux:**

```bash
cd webapi
/path/to/HOOPS_AI/install/dir/.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8001 --reload
```

- API base URL: `http://127.0.0.1:8001`
- Interactive docs (Swagger UI): `http://127.0.0.1:8001/docs`

---

## API Endpoints

### 3D CAD Viewer

#### Launch viewer — Upload file

Upload a local CAD file and open an interactive browser viewer.

```
POST /CAD/viewer
```

**Windows (PowerShell):**
```powershell
curl.exe -X POST "http://127.0.0.1:8001/CAD/viewer" `
         -F "file=@C:\path\to\model.stp"
```

**Linux:**
```bash
curl -X POST "http://127.0.0.1:8001/CAD/viewer" \
     -F "file=@/path/to/model.stp"
```

**Response:**

```json
{ "viewer_url": "http://127.0.0.1:<viewer_port>/index.html" }
```

Open the returned `viewer_url` in your browser to view the model.

> The viewer runs on a **separate port** from the API server. Make sure that port is not blocked by a firewall.

#### Launch viewer — Shared folder path

Open a CAD file already present in the shared folder (`HOOPS_AI_CAD_SHARED_DIR`).

```
POST /CAD/viewer/from-path
```

**Windows (PowerShell):**
```powershell
curl.exe -X POST "http://127.0.0.1:8001/CAD/viewer/from-path" `
         -d "cad_file_path=model.stp"
```

**Linux:**
```bash
curl -X POST "http://127.0.0.1:8001/CAD/viewer/from-path" \
     -d "cad_file_path=model.stp"
```

**Response:** same as above.

> This endpoint is also used by the browser UI at `http://127.0.0.1:8001/CAD/viewer`.

#### Terminate viewer

```
DELETE /CAD/viewer          # terminate last active viewer
DELETE /CAD/viewer?all=true # terminate all viewers
```

**Windows (PowerShell):**
```powershell
Invoke-RestMethod -Method Delete -Uri "http://127.0.0.1:8001/CAD/viewer"
Invoke-RestMethod -Method Delete -Uri "http://127.0.0.1:8001/CAD/viewer?all=true"
```

**Linux:**
```bash
curl -X DELETE "http://127.0.0.1:8001/CAD/viewer"
curl -X DELETE "http://127.0.0.1:8001/CAD/viewer?all=true"
```

**Response:** `{ "terminated": 1 }`

---

### B-Rep Analysis

#### Face adjacency graph

Build a face adjacency graph from the B-Rep model. Returns graph data and a base64-encoded PNG visualization.

```
POST /BRep/adjacency-graph
```

**Windows (PowerShell):**
```powershell
curl.exe -X POST "http://127.0.0.1:8001/BRep/adjacency-graph" `
    -F "file=@C:\path\to\model.SLDPRT"
```

**Linux:**
```bash
curl -X POST "http://127.0.0.1:8001/BRep/adjacency-graph" \
    -F "file=@/path/to/model.SLDPRT"
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
  "graph_image": "<base64-encoded PNG>"
}
```

#### Face and edge attributes

Extract face and edge attributes from the B-Rep model.

```
POST /BRep/attributes
```

**Windows (PowerShell):**
```powershell
curl.exe -X POST "http://127.0.0.1:8001/BRep/attributes" `
    -F "file=@C:\path\to\model.SLDPRT"
```

**Linux:**
```bash
curl -X POST "http://127.0.0.1:8001/BRep/attributes" \
    -F "file=@/path/to/model.SLDPRT"
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

---

### Manufacturing Feature Recognition (MFR)

#### Dataset table of contents

Returns a summary of the loaded MFR dataset.

```
GET /MFR/dataset/table-of-contents
```

**Windows (PowerShell):** `curl.exe "http://127.0.0.1:8001/MFR/dataset/table-of-contents"`

**Linux:** `curl "http://127.0.0.1:8001/MFR/dataset/table-of-contents"`

#### List label descriptions

Returns all MFR label IDs with their names and descriptions.

```
GET /MFR/labels/description
```

**Windows (PowerShell):** `curl.exe "http://127.0.0.1:8001/MFR/labels/description"`

**Linux:** `curl "http://127.0.0.1:8001/MFR/labels/description"`

#### Search files by feature

Returns CAD file names and IDs that contain a given manufacturing feature.

```
GET /MFR/files/search?feature_name=<name>
```

**Windows (PowerShell):** `curl.exe "http://127.0.0.1:8001/MFR/files/search?feature_name=through%20hole"`

**Linux:** `curl "http://127.0.0.1:8001/MFR/files/search?feature_name=through%20hole"`

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
Invoke-RestMethod -Uri "http://127.0.0.1:8001/MFR/files/1/thumbnail" -OutFile "thumbnail.png"
```

**Linux:**
```bash
curl "http://127.0.0.1:8001/MFR/files/1/thumbnail" -o thumbnail.png
```

**Response:** PNG image (`image/png`)

#### Run inference

Upload a CAD file and run MFR inference. Launches the CAD viewer and returns predictions, probabilities, and viewer URL.

```
POST /MFR/inference
```

**Windows (PowerShell):**
```powershell
curl.exe -X POST "http://127.0.0.1:8001/MFR/inference" `
    -F "file=@C:\path\to\model.SLDPRT"
```

**Linux:**
```bash
curl -X POST "http://127.0.0.1:8001/MFR/inference" \
    -F "file=@/path/to/model.SLDPRT"
```

**Response:**

```json
{
  "predictions": [...],
  "probabilities": [...],
  "viewer_url": "http://127.0.0.1:<viewer_port>/index.html"
}
```

#### Colorize viewer

Apply MFR prediction colors to the last active CAD viewer. Call this **after** the model has fully loaded in the browser.

```
POST /MFR/viewer/colorize
```

**Windows (PowerShell):**
```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8001/MFR/viewer/colorize"
```

**Linux:**
```bash
curl -X POST "http://127.0.0.1:8001/MFR/viewer/colorize"
```

**Response:**

```json
{
  "color_map": {
    "17": {"name": "through hole", "color_rgb": [255, 0, 0]},
    "18": {"name": "circular blind step", "color_rgb": [0, 255, 0]}
  }
}
```

---

### Shape Similarity Search

> **Prerequisite:** Before using this endpoint, you must build the FAISS index by running the notebook `5b_cad_search_using_HOOPS_embeddings.ipynb` up to and including the **Saving an Index** section. This generates `fabwave_embeddings_store.faiss` in the notebooks folder. Without this file the similarity search endpoint will fail.

Upload a CAD file and retrieve the most similar parts from the indexed database using HOOPS Embeddings and a FAISS index.

```
POST /similarity/search?top_k=<n>
```

**Windows (PowerShell):**
```powershell
curl.exe -X POST "http://127.0.0.1:8001/similarity/search?top_k=10" `
    -F "file=@C:\path\to\model.step"
```

**Linux:**
```bash
curl -X POST "http://127.0.0.1:8001/similarity/search?top_k=10" \
    -F "file=@/path/to/model.step"
```

**Response:**

```json
{
  "results": [
    {"id": "part_042", "score": 0.997},
    {"id": "part_018", "score": 0.991}
  ],
  "image_url": "http://127.0.0.1:8001/out/<uuid>.png"
}
```

- `results` — top-k matches sorted by similarity score (higher = more similar)
- `image_url` — URL to a PNG grid image of the search results

---

## Running tests

```bash
cd webapi
python -m unittest discover -s tests
```

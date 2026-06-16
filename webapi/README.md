# HOOPS AI WebAPI

A FastAPI-based REST API that exposes [HOOPS AI](https://www.techsoft3d.com/developers/products/hoops-ai/) (Tech Soft 3D) capabilities as HTTP endpoints.  
See the root [README](../README.md) for an overview of the full HOOPS AI MCP platform.

---

## Requirements

- Python 3.9 (recommended: Miniconda/Anaconda environment)
- A valid **HOOPS AI license key**
- HOOPS AI Python package (`hoops_ai_cpu` or `hoops_ai_gpu`) installed in the environment

---

## Setup

### 1. Install dependencies

```bash
cd webapi
pip install -r requirements.txt
```

> Install the `hoops_ai_cpu` or `hoops_ai_gpu` package separately according to your HOOPS AI distribution instructions.

### 2. Configure environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `HOOPS_AI_LICENSE` | ✅ | Your HOOPS AI license key |
| `HOOPS_AI_NOTEBOOK_DIR` | ✅ | Absolute path to your HOOPS AI notebooks directory |
| `HOOPS_AI_MFR_FLOW_NAME` | ✅ | MFR flow name (dataset files are resolved relative to this) |
| `HOOPS_AI_MFR_MODEL_NAME` | ✅ | MFR trained model checkpoint filename (e.g. `ts3d_162k_mfr.ckpt`) |
| `HOOPS_AI_CAD_SHARED_DIR` | optional | Shared folder for CAD files (defaults to `./uploads`) |
| `HOOPS_AI_MFR_LABELS_DESCRIPTION` | optional | Custom MFR label map (Python dict literal) |

> **Note:** `HOOPS_AI_LICENSE` is read **only** from the `.env` file, not from system environment variables.

Example `.env`:

```
HOOPS_AI_LICENSE=XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
HOOPS_AI_NOTEBOOK_DIR=C:\hoops_ai\notebooks
HOOPS_AI_MFR_FLOW_NAME=cadsynth_1000
HOOPS_AI_MFR_MODEL_NAME=ts3d_162k_mfr.ckpt
```

### 3. Start the server

Run from the `webapi/` directory using the Python executable from your HOOPS AI conda environment:

```bash
cd webapi
C:\Users\<user>\miniconda3\envs\hoops_ai_cpu\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001
```

For development with auto-reload:

```bash
C:\Users\<user>\miniconda3\envs\hoops_ai_cpu\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001 --reload
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

```powershell
curl.exe -X POST "http://127.0.0.1:8001/CAD/viewer" `
         -F "file=@C:\path\to\model.stp"
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

```powershell
curl.exe -X POST "http://127.0.0.1:8001/CAD/viewer/from-path" `
         -d "cad_file_path=model.stp"
```

**Response:** same as above.

> This endpoint is also used by the browser UI at `http://127.0.0.1:8001/CAD/viewer`.

#### Terminate viewer

```
DELETE /CAD/viewer          # terminate last active viewer
DELETE /CAD/viewer?all=true # terminate all viewers
```

```powershell
Invoke-RestMethod -Method Delete -Uri "http://127.0.0.1:8001/CAD/viewer"
Invoke-RestMethod -Method Delete -Uri "http://127.0.0.1:8001/CAD/viewer?all=true"
```

**Response:** `{ "terminated": 1 }`

---

### B-Rep Analysis

#### Face adjacency graph

Build a face adjacency graph from the B-Rep model. Returns graph data and a base64-encoded PNG visualization.

```
POST /BRep/adjacency-graph
```

```powershell
curl.exe -X POST "http://127.0.0.1:8001/BRep/adjacency-graph" `
    -F "file=@C:\path\to\model.SLDPRT"
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

```powershell
curl.exe -X POST "http://127.0.0.1:8001/BRep/attributes" `
    -F "file=@C:\path\to\model.SLDPRT"
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

```bash
curl.exe "http://127.0.0.1:8001/MFR/dataset/table-of-contents"
```

#### List label descriptions

Returns all MFR label IDs with their names and descriptions.

```
GET /MFR/labels/description
```

```bash
curl.exe "http://127.0.0.1:8001/MFR/labels/description"
```

#### Search files by feature

Returns CAD file names and IDs that contain a given manufacturing feature.

```
GET /MFR/files/search?feature_name=<name>
```

```bash
curl.exe "http://127.0.0.1:8001/MFR/files/search?feature_name=through%20hole"
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

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8001/MFR/files/1/thumbnail" -OutFile "thumbnail.png"
```

**Response:** PNG image (`image/png`)

#### Run inference

Upload a CAD file and run MFR inference. Launches the CAD viewer and returns predictions, probabilities, and viewer URL.

```
POST /MFR/inference
```

```powershell
curl.exe -X POST "http://127.0.0.1:8001/MFR/inference" `
    -F "file=@C:\path\to\model.SLDPRT"
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

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8001/MFR/viewer/colorize"
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

Upload a CAD file and retrieve the most similar parts from the indexed database using HOOPS Embeddings and a FAISS index.

```
POST /similarity/search?top_k=<n>
```

```powershell
curl.exe -X POST "http://127.0.0.1:8001/similarity/search?top_k=10" `
    -F "file=@C:\path\to\model.step"
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

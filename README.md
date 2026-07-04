# HOOPS AI WebAPI

A FastAPI-based REST API that exposes [HOOPS AI](https://www.techsoft3d.com/developers/products/hoops-ai/) (Tech Soft 3D) capabilities as HTTP endpoints.

---

## Requirements

- Python 3.12
- A valid **HOOPS AI license key**
- HOOPS AI (CPU or GPU version) installed in the environment
- **HOOPS AI Tutorials** – the notebooks folder and its contents (ML datasets and pre-trained models) are required to run this server.  
  The tutorials are available at [github.com/techsoft3d/HOOPS-AI-tutorials](https://github.com/techsoft3d/HOOPS-AI-tutorials/tree/main).  
  Data packages (datasets and trained model checkpoints) must be obtained from the Tech Soft 3D File Transfer service by following the HOOPS AI installation instructions.

  **Directory layout** – `notebooks/` and `packages/` must both reside directly under the HOOPS AI install directory:

  ```
  <HOOPS_AI_INSTALL_DIR>/
  ├── notebooks/
  └── packages/
      ├── flows/
      └── trained_ml_models/
  ```

  **Pre-run requirements** – some endpoints require notebook output to be generated in advance:

  | Endpoint | Notebook to run | Generated files |
  |---|---|---|
  | MFR endpoints | `3b_workflow_for_MFR_cadsynth.ipynb` | `notebooks/out/flows/<flow_name>/`<br>`.dataset` / `.infoset` / `.attribset` |
  | `/similarity/search` | `5b_cad_search_using_HOOPS_embeddings.ipynb`<br>(up to **Saving an Index**) | `fabwave_embeddings_store.faiss` |
  | `/part-classification/dataset/*` | `3c_workflow_for_Part_classification_fabwave.ipynb`<br>(up to **Pipeline execution**) | flow `.dataset` / `.infoset` / `.attribset`<br>`stream_cache/*.png` |

  > **Tip:** Pre-generated dataset files are also available for download from the Tech Soft 3D File Transfer service — no need to run the notebooks yourself:  
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

#### Additional steps for headless Linux (Ubuntu 22.04)

HOOPS AI requires OpenGL and a display to validate its license and render CAD files.
On a headless server (no GPU, no monitor), install the following system packages:

```bash
sudo apt-get install -y libglu1-mesa libgl1 libgl1-mesa-dri libosmesa6 xvfb
```

| Package | Purpose |
|---|---|
| `libglu1-mesa` | OpenGL Utility Library |
| `libgl1` | OpenGL runtime |
| `libgl1-mesa-dri` | Mesa software renderer (swrast) |
| `libosmesa6` | Off-screen Mesa rendering |
| `xvfb` | Virtual framebuffer (headless display) |

Then start a virtual display before launching the server (see [Step 4](#4-start-the-server)).

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
| `HOOPS_AI_LICENSE` | ✓ | Your HOOPS AI license key |
| `HOOPS_AI_NOTEBOOK_DIR` | ✓ | Absolute path to your HOOPS AI notebooks directory |
| `HOOPS_AI_MFR_FLOW_NAME` | optional | MFR flow name (dataset files are resolved relative to this) |
| `HOOPS_AI_MFR_MODEL_NAME` | optional | MFR trained model checkpoint filename (e.g. `ts3d_162k_mfr.ckpt`) |
| `HOOPS_AI_EMBEDDINGS_MODEL_NAME` | optional | Embeddings trained model checkpoint filename (e.g. `ts3d_1M_hoops_embeddings.ckpt`) |
| `HOOPS_AI_FAISS_INDEX_PATH` | optional | FAISS index file for shape similarity search (e.g. `fabwave_embeddings_store.faiss`) |
| `HOOPS_AI_PART_CLASS_MODEL_NAME` | optional | Filename of the trained GraphClassification checkpoint under `packages/trained_ml_models/` (e.g. `ts3d_graphclassification_5k_10epochs.ckpt`) |
| `HOOPS_AI_PART_CLASS_FLOW_NAME` | optional | Part Classification flow name (required for `/part-classification/dataset/*` endpoints). The server automatically prefers `<HOOPS_AI_NOTEBOOK_DIR>/out/flows/<name>` (notebook output, includes thumbnails) and falls back to `../packages/flows/<name>` (pre-packaged). |
| `HOOPS_AI_PART_CLASS_LABEL_KEY` | optional | Label array key for dataset queries (default: `part_label`; use `task_A` for custom ETL) |

> **Note:** `HOOPS_AI_LICENSE` is read **only** from the `.env` file, not from system environment variables.

Example `.env`:

```
HOOPS_AI_LICENSE=XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
HOOPS_AI_NOTEBOOK_DIR=C:\hoops_ai\notebooks
HOOPS_AI_MFR_FLOW_NAME=ETL_CADSYNTH_training_b2
HOOPS_AI_MFR_MODEL_NAME=ts3d_162k_mfr.ckpt
HOOPS_AI_EMBEDDINGS_MODEL_NAME=ts3d_1M_hoops_embeddings.ckpt
HOOPS_AI_FAISS_INDEX_PATH=fabwave_embeddings_store.faiss
HOOPS_AI_PART_CLASS_MODEL_NAME=ts3d_graphclassification_5k_10epochs.ckpt
HOOPS_AI_PART_CLASS_FLOW_NAME=ETL_Fabwave_training_b2
```

### 4. Start the server

Run from the `` directory using the Python executable from your HOOPS AI virtual environment.

**Windows:**

```bat
<Path\to\HOOPS_AI\install\dir>\.venv\Scripts\python.exe main.py --host 0.0.0.0 --port 8000
```

**Linux:**

```bash
/path/to/HOOPS_AI/install/dir/.venv/bin/python main.py --host 0.0.0.0 --port 8000
```

**Linux (headless – Ubuntu 22.04 without display):**

HOOPS AI requires a display for license validation and 3D rendering. Start a virtual framebuffer first:

```bash
# Start virtual display (if not already running)
Xvfb :99 -screen 0 1280x960x24 &

# Launch the server with DISPLAY set
DISPLAY=:99 /path/to/HOOPS_AI/install/dir/.venv/bin/python /path/to/HOOPS_AI-MCP/main.py --host 0.0.0.0 --port 8000
```

> The xkbcomp warnings printed by Xvfb (`Could not resolve keysym XF86...`) are harmless and can be ignored.

> If Xvfb fails with `Server is already active for display 99`, the virtual display is already running – skip the `Xvfb` line and proceed directly to the `DISPLAY=:99 python ...` command.

Alternatively, use the provided startup script which handles Xvfb automatically:

```bash
# Make executable (first time only)
chmod +x start_server.sh

# Run (default: --host 0.0.0.0 --port 8000)
./start_server.sh

# Custom port
./start_server.sh --port 8001

# Custom HOOPS AI venv path
HOOPS_AI_VENV=/custom/path/.venv ./start_server.sh
```

> Replace the path prefix with the actual directory where HOOPS AI is installed.
> The venv Python executable ensures HOOPS AI packages from that environment are used.

> **Tip – Auto-start on boot (systemd) and file permission issues:**
>
> If you run the server as a systemd service, make sure the service runs as the **same user** you use for manual testing (typically `ubuntu`), not as `root`.  
> Running as `root` causes uploaded files to be owned by `root`, which then triggers a `PermissionError` when the server tries to clean the `uploads/` folder on the next startup by a non-root user.
>
> Use `User=` and `Group=` in the `[Service]` section of your unit file:
>
> ```ini
> [Unit]
> Description=HOOPS AI WebAPI Server
> After=network.target
>
> [Service]
> Type=simple
> User=ubuntu
> Group=ubuntu
> WorkingDirectory=/var/HOOPS_AI-WebAPI
> ExecStart=/var/HOOPS_AI-MCP/start_server.sh --host 0.0.0.0 --port 8000
> Restart=on-failure
> RestartSec=5
>
> [Install]
> WantedBy=multi-user.target
> ```
>
> If you already have files owned by `root` in `uploads/`, fix ownership once with:
> ```bash
> sudo chown -R ubuntu:ubuntu /var/HOOPS_AI-MCP/uploads
> ```

> **Note:** Port `8000` is the default. If port 8000 is already in use, the server will print an error and exit – simply retry with a different port (e.g. `--port 8001`) and update `HOOPS_WEBAPI_URL` in the MCP server config accordingly.

> **Note (Windows):** To allow connections from other machines on the LAN, add a Windows Firewall inbound rule for port 8000 (TCP).

For development with auto-reload:

**Windows:**

```bat
<Path\to\HOOPS_AI\install\dir>\.venv\Scripts\python.exe main.py --host 0.0.0.0 --port 8000 --reload
```

**Linux:**

```bash
/path/to/HOOPS_AI/install/dir/.venv/bin/python main.py --host 0.0.0.0 --port 8000 --reload
```

- API base URL: `http://<server-ip>:8000`
- Interactive docs (Swagger UI): `http://<server-ip>:8000/docs`

> **`<server-ip>` substitution:**  
> - **Same machine** – use `127.0.0.1` (e.g. `http://127.0.0.1:8000`). No IP lookup needed.  
> - **Different machine** – use the LAN IP of the server machine (e.g. `http://192.168.0.6:8000`).  
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
curl.exe -X POST "http://<server-ip>:8000/files/upload" `
         -F "file=@C:\path\to\model.stp"
```

**Linux:**
```bash
curl -X POST "http://<server-ip>:8000/files/upload" \
     -F "file=@/path/to/model.stp"
```

**Response:**

```json
{ "file_id": "a3f8c2...", "filename": "model.stp", "already_existed": false }
```

Pass the returned `file_id` to any processing endpoint instead of re-uploading the same file.

---

### 3D CAD Viewer

#### Launch viewer – Upload file

Upload a local CAD file and open an interactive browser viewer.

```
POST /CAD/viewer
```

**Windows (PowerShell):**
```powershell
curl.exe -X POST "http://<server-ip>:8000/CAD/viewer" `
         -F "file=@C:\path\to\model.stp"
```

**Linux:**
```bash
curl -X POST "http://<server-ip>:8000/CAD/viewer" \
     -F "file=@/path/to/model.stp"
```

**Response:**

```json
{ "viewer_url": "http://<server-ip>:<viewer_port>/index.html", "image_url": "http://<server-ip>:8000/out/<stem>.png" }
```

Open the returned `viewer_url` in your browser to view the model. `image_url` is a PNG preview of the model.

> The viewer runs on a **separate port** from the API server. Make sure that port is not blocked by a firewall.

> **Note:** The `out/` and `uploads/` folders are automatically cleared on server startup.

#### Launch viewer – Shared folder path

Open a CAD file already present in the shared folder (`HOOPS_AI_CAD_SHARED_DIR`).

```
POST /CAD/viewer/from-path
```

**Windows (PowerShell):**
```powershell
curl.exe -X POST "http://<server-ip>:8000/CAD/viewer/from-path" `
         -d "cad_file_path=model.stp"
```

**Linux:**
```bash
curl -X POST "http://<server-ip>:8000/CAD/viewer/from-path" \
     -d "cad_file_path=model.stp"
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
Invoke-RestMethod -Method Delete -Uri "http://<server-ip>:8000/CAD/viewer"
Invoke-RestMethod -Method Delete -Uri "http://<server-ip>:8000/CAD/viewer?all=true"
```

**Linux:**
```bash
curl -X DELETE "http://<server-ip>:8000/CAD/viewer"
curl -X DELETE "http://<server-ip>:8000/CAD/viewer?all=true"
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
curl.exe -X POST "http://<server-ip>:8000/BRep/adjacency-graph" `
    -F "file=@C:\path\to\model.SLDPRT"
```

**Linux:**
```bash
curl -X POST "http://<server-ip>:8000/BRep/adjacency-graph" \
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
  "image_url": "http://<server-ip>:8000/out/<uuid>.png"
}
```

#### Face and edge attributes

Extract face and edge attributes from the B-Rep model.

```
POST /BRep/attributes
```

**Windows (PowerShell):**
```powershell
curl.exe -X POST "http://<server-ip>:8000/BRep/attributes" `
    -F "file=@C:\path\to\model.SLDPRT"
```

**Linux:**
```bash
curl -X POST "http://<server-ip>:8000/BRep/attributes" \
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

**Windows (PowerShell):** `curl.exe "http://<server-ip>:8000/MFR/dataset/table-of-contents"`

**Linux:** `curl "http://<server-ip>:8000/MFR/dataset/table-of-contents"`

#### List label descriptions

Returns all MFR label IDs with their names and descriptions.

```
GET /MFR/labels/description
```

**Windows (PowerShell):** `curl.exe "http://<server-ip>:8000/MFR/labels/description"`

**Linux:** `curl "http://<server-ip>:8000/MFR/labels/description"`

#### Search files by feature

Returns CAD file names and IDs that contain a given manufacturing feature.

```
GET /MFR/files/search?feature_name=<name>
```

**Windows (PowerShell):** `curl.exe "http://<server-ip>:8000/MFR/files/search?feature_name=through%20hole"`

**Linux:** `curl "http://<server-ip>:8000/MFR/files/search?feature_name=through%20hole"`

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
Invoke-RestMethod -Uri "http://<server-ip>:8000/MFR/files/1/thumbnail" -OutFile "thumbnail.png"
```

**Linux:**
```bash
curl "http://<server-ip>:8000/MFR/files/1/thumbnail" -o thumbnail.png
```

**Response:** PNG image (`image/png`)

#### Run inference

Upload a CAD file and run MFR inference. Launches the CAD viewer and returns predictions, probabilities, and viewer URL.

```
POST /MFR/inference
```

**Windows (PowerShell):**
```powershell
curl.exe -X POST "http://<server-ip>:8000/MFR/inference" `
    -F "file=@C:\path\to\model.SLDPRT"
```

**Linux:**
```bash
curl -X POST "http://<server-ip>:8000/MFR/inference" \
    -F "file=@/path/to/model.SLDPRT"
```

**Response:**

```json
{
  "predictions": [...],
  "probabilities": [...],
  "viewer_url": "http://<server-ip>:<viewer_port>/index.html",
  "image_url": "http://<server-ip>:8000/out/<stem>.png"
}
```

#### Colorize viewer

Apply MFR prediction colors to the last active CAD viewer. Call this **after** the model has fully loaded in the browser.

```
POST /MFR/viewer/colorize
```

**Windows (PowerShell):**
```powershell
Invoke-RestMethod -Method Post -Uri "http://<server-ip>:8000/MFR/viewer/colorize"
```

**Linux:**
```bash
curl -X POST "http://<server-ip>:8000/MFR/viewer/colorize"
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

**Windows (PowerShell):**
```powershell
curl.exe -X POST "http://<server-ip>:8000/similarity/search?top_k=10" `
    -F "file=@C:\path\to\model.step"
```

**Linux:**
```bash
curl -X POST "http://<server-ip>:8000/similarity/search?top_k=10" \
    -F "file=@/path/to/model.step"
```

**Response:**

```json
{
  "results": [
    {"id": "part_042", "score": 0.997},
    {"id": "part_018", "score": 0.991}
  ],
  "image_url": "http://<server-ip>:8000/out/<uuid>.png"
}
```

- `results` – top-k matches sorted by similarity score (higher = more similar)
- `image_url` – URL to a PNG grid image of the search results

#### Part thumbnail image

Return the pre-generated PNG thumbnail for a trained part by filename.

```
GET /similarity/part-image?filename=<name>
```

**Windows (PowerShell):**
```powershell
curl.exe "http://<server-ip>:8000/similarity/part-image?filename=part_042.stp" -o part_042.png
```

**Linux:**
```bash
curl "http://<server-ip>:8000/similarity/part-image?filename=part_042.stp" -o part_042.png
```

**Response:** PNG image (`image/png`)

---

#### Similarity search index info

Return metadata about the FAISS similarity-search index currently loaded on
the server.  This endpoint is read-only and never triggers index construction
or model training.  When the index has not been loaded yet, a
``"not_loaded"`` status is returned instead of an error.

```
GET /similarity/index-info
```

**Windows (PowerShell):**
```powershell
curl.exe "http://<server-ip>:8000/similarity/index-info"
```

**Linux:**
```bash
curl "http://<server-ip>:8000/similarity/index-info"
```

**Response (index loaded):**

```json
{
  "status": "loaded",
  "index_path": "/path/to/notebooks/fabwave_embeddings_store.faiss",
  "index_last_modified": "2025-06-01T12:00:00Z",
  "index_count": 5000,
  "model_name": "hoops_embeddings_model",
  "embedding_dim": 512,
  "metadata": {"failed_count": 0}
}
```

**Response (index not yet loaded):**

```json
{
  "status": "not_loaded",
  "index_path": "/path/to/notebooks/fabwave_embeddings_store.faiss",
  "index_last_modified": "2025-06-01T12:00:00Z",
  "index_count": null,
  "model_name": null,
  "embedding_dim": null,
  "metadata": null
}
```

| Field | Description |
|---|---|
| `status` | `"loaded"` or `"not_loaded"` |
| `index_path` | Absolute path to the FAISS index file (from env) |
| `index_last_modified` | UTC last-modified timestamp of the index file (`null` if file not found) |
| `index_count` | Number of embeddings stored in the index |
| `model_name` | Name of the embedding model used to build the index |
| `embedding_dim` | Dimension of each embedding vector |
| `metadata` | Auxiliary metadata stored in the index (e.g. `failed_count`) |

---

#### Compute shape embedding (index-free)

Compute (or retrieve from cache) the shape embedding vector for a single CAD part.
This endpoint does **not** require a FAISS index — the embedding model alone is sufficient.

```
POST /similarity/embed
```

Supply **either** a file upload or a `file_id` from a previous `POST /files/upload`.

**Windows (PowerShell):**
```powershell
# Upload a file and get its embedding
curl.exe -X POST "http://<server-ip>:8000/similarity/embed" `
    -F "file=@C:\path\to\bracket.step"

# Or use an already-uploaded file_id
curl.exe -X POST "http://<server-ip>:8000/similarity/embed?file_id=a3f8c2..."
```

**Linux:**
```bash
curl -X POST "http://<server-ip>:8000/similarity/embed" \
    -F "file=@/path/to/bracket.step"
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

#### Compare parts by shape similarity (index-free)

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

At least **two** valid parts are required.  Per-file failures are collected in `errors`
and do not abort the request (unless fewer than two parts succeed).

**Windows (PowerShell) – upload two files directly:**
```powershell
curl.exe -X POST "http://<server-ip>:8000/similarity/compare" `
    -F "files=@C:\path\to\bracket_a.step" `
    -F "files=@C:\path\to\bracket_b.step"
```

**Linux – upload two files directly:**
```bash
curl -X POST "http://<server-ip>:8000/similarity/compare" \
    -F "files=@/path/to/bracket_a.step" \
    -F "files=@/path/to/bracket_b.step"
```

**Linux – compare using already-uploaded file_ids:**
```bash
curl -X POST "http://<server-ip>:8000/similarity/compare?file_ids=a3f8c2...,cd34ef..."
```

**Linux – compare files inside a ZIP archive:**
```bash
curl -X POST "http://<server-ip>:8000/similarity/compare" \
    -F "zip_file=@/path/to/parts.zip"
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

ZIP archives are filtered to recognised CAD extensions (`.step .stp .iges .igs .x_t .x_b .sat .ipt .prt .sldprt .catpart`).
Paths that escape the extraction directory (Zip Slip) are rejected with HTTP 400.
Uncompressed size is capped at 500 MB and file count at 50 (HTTP 413 if exceeded).

---

## Shape Space Map

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

At least **two** valid parts are required.  Accepts the same three input sources as
`POST /similarity/compare`.  The response includes a 3D `position` for each part, the
similarity `matrix`, a Kruskal `stress` value (layout accuracy: `0.0` is exact), and an
absolute `viewer_url` that opens the interactive map.  The viewer page fetches its layout
data from `/out/shape_map_<map_id>.json`.

**Linux – generate a shape map from uploaded files:**
```bash
# Upload parts
curl -s -X POST http://localhost:8000/files/upload -F "file=@part_a.step"
curl -s -X POST http://localhost:8000/files/upload -F "file=@part_b.step"

# Generate shape map
curl -s -X POST "http://localhost:8000/similarity/map?file_ids=<id_a>,<id_b>" | python -m json.tool

# Open the viewer_url from the response in a browser
```

**Windows (PowerShell) – upload parts directly:**
```powershell
curl.exe -X POST "http://<server-ip>:8000/similarity/map" `
    -F "files=@C:\path\to\bracket_a.step" `
    -F "files=@C:\path\to\bracket_b.step"
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

### Shape Space Map – Query Overlay

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
| `persist` | query | `false` (default) — overlay only; `true` — add to original map permanently |

Supply **either** `file_id` **or** a `file` upload.

**Windows (PowerShell) – direct upload:**
```powershell
curl.exe -X POST "http://localhost:8000/similarity/map/d2a7f205/query" `
    -F "file=@C:\temp\Sprocket.step"

# Or with PowerShell native:
$form = @{ file = Get-Item "C:\temp\Sprocket.step" }
Invoke-RestMethod -Uri "http://localhost:8000/similarity/map/d2a7f205/query" -Method POST -Form $form
```

**Linux – use an already-uploaded file:**
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
| `viewer_url` | Absolute URL — open in browser to see the query highlighted in magenta |
| `query_part` | Query part metadata, position, and `is_query: true` flag |
| `nearest_parts` | Top-5 most similar existing parts sorted by cosine similarity |
| `persisted` | `true` when `persist=true` was used and the query was added to the original map |

The overlay map is independent of the original — by default it exists only until the
server restarts.  Use `persist=true` to permanently add the query part to the source map.

---



Manage user-created similarity indexes that grow over time.  Unlike the built-in
read-only ``default`` index (backed by ``HOOPS_AI_FAISS_INDEX_PATH``), named indexes
are fully writable: create an empty index, register new parts whenever they arrive,
and query immediately — all via Web API with no notebook re-runs.

Indexes are stored under ``APP_ROOT/indexes/<name>.faiss`` (+ ``.meta``).
Index names must match ``^[a-z0-9_-]{1,64}$``; ``default`` is reserved.

#### Incremental workflow example

```
# 1. Create an empty index
POST /similarity/index/create?name=my-parts

# 2. Register parts (repeat as new parts arrive)
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

# 6. Delete the whole index (destructive — requires confirm=true)
DELETE /similarity/index/my-parts?confirm=true
```

#### Create a named index

```
POST /similarity/index/create?name=<name>
```

Returns **201** on success, **409** if the name already exists, **422** for invalid/reserved names.

**Windows (PowerShell):**
```powershell
curl.exe -X POST "http://<server-ip>:8000/similarity/index/create?name=my-parts"
```

**Linux:**
```bash
curl -X POST "http://<server-ip>:8000/similarity/index/create?name=my-parts"
```

**Response:**
```json
{ "name": "my-parts", "count": 0, "dim": 512 }
```

---

#### List all indexes

```
GET /similarity/index/list
```

Returns all named indexes plus the built-in ``default`` index (``is_readonly: true``).

**Windows (PowerShell):** `curl.exe "http://<server-ip>:8000/similarity/index/list"`

**Linux:** `curl "http://<server-ip>:8000/similarity/index/list"`

**Response:**
```json
[
  { "name": "default",  "count": 5000, "last_modified": "2025-06-01T12:00:00Z", "is_readonly": true  },
  { "name": "my-parts", "count": 3,    "last_modified": "2026-07-01T08:30:00Z", "is_readonly": false }
]
```

---

#### Register parts in a named index

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
Embedding results are cached on disk — re-adding the same file is fast.

**Windows (PowerShell):**
```powershell
# Upload a new part directly
curl.exe -X POST "http://<server-ip>:8000/similarity/index/add?name=my-parts" `
    -F "files=@C:\path\to\new_bracket.step"

# Add an already-uploaded part by file_id
curl.exe -X POST "http://<server-ip>:8000/similarity/index/add?name=my-parts" `
    -F "" --data-urlencode "file_ids=a3f8c2..."
```

**Linux:**
```bash
curl -X POST "http://<server-ip>:8000/similarity/index/add?name=my-parts" \
    -F "files=@/path/to/new_bracket.step"
```

**Response:**
```json
{ "name": "my-parts", "added": 1, "updated": 0, "index_count": 4, "errors": [] }
```

---

#### Search a named index

```
POST /similarity/index/{name}/search?top_k=<n>
```

Supply **either** a file upload or a `file_id`.  Returns an empty ``hits`` list
when the index contains zero entries (no error).

**Windows (PowerShell):**
```powershell
curl.exe -X POST "http://<server-ip>:8000/similarity/index/my-parts/search?top_k=5" `
    -F "file=@C:\path\to\query.step"
```

**Linux:**
```bash
curl -X POST "http://<server-ip>:8000/similarity/index/my-parts/search?top_k=5" \
    -F "file=@/path/to/query.step"
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

#### Remove parts from a named index

```
DELETE /similarity/index/{name}/parts?part_ids=<id1>,<id2>,...
```

**Windows (PowerShell):**
```powershell
Invoke-RestMethod -Method Delete `
    -Uri "http://<server-ip>:8000/similarity/index/my-parts/parts?part_ids=a3f8c2...,cd34ef..."
```

**Linux:**
```bash
curl -X DELETE "http://<server-ip>:8000/similarity/index/my-parts/parts?part_ids=a3f8c2...,cd34ef..."
```

**Response:**
```json
{ "name": "my-parts", "removed": 2, "index_count": 2 }
```

---

#### Delete a named index

```
DELETE /similarity/index/{name}?confirm=true
```

Destructive and irreversible.  Requires ``?confirm=true``; without it returns **409** with
an instruction.  Returns **403** for the read-only ``default`` index.

**Windows (PowerShell):**
```powershell
Invoke-RestMethod -Method Delete -Uri "http://<server-ip>:8000/similarity/index/my-parts?confirm=true"
```

**Linux:**
```bash
curl -X DELETE "http://<server-ip>:8000/similarity/index/my-parts?confirm=true"
```

**Response:**
```json
{ "name": "my-parts", "deleted": true }
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
curl.exe -X POST "http://<server-ip>:8000/part-classification/predict?top_k=5" `
    -F "file=@C:\path\to\model.stp"
```

**Linux:**
```bash
curl -X POST "http://<server-ip>:8000/part-classification/predict?top_k=5" \
    -F "file=@/path/to/model.stp"
```

**Reuse an uploaded file by `file_id`:**
```powershell
# Windows
curl.exe -X POST "http://<server-ip>:8000/part-classification/predict?file_id=<file_id>&top_k=5"
```
```bash
# Linux
curl -X POST "http://<server-ip>:8000/part-classification/predict?file_id=<file_id>&top_k=5"
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

**Windows (PowerShell):** `curl.exe "http://<server-ip>:8000/part-classification/labels"`

**Linux:** `curl "http://<server-ip>:8000/part-classification/labels"`

#### Dataset table of contents

```
GET /part-classification/dataset/table-of-contents
```

**Windows (PowerShell):** `curl.exe "http://<server-ip>:8000/part-classification/dataset/table-of-contents"`

**Linux:** `curl "http://<server-ip>:8000/part-classification/dataset/table-of-contents"`

#### Per-class file count distribution

```
GET /part-classification/dataset/label-distribution
```

**Windows (PowerShell):** `curl.exe "http://<server-ip>:8000/part-classification/dataset/label-distribution"`

**Linux:** `curl "http://<server-ip>:8000/part-classification/dataset/label-distribution"`

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

**Windows (PowerShell):** `curl.exe "http://<server-ip>:8000/part-classification/dataset/files?label_id=30"`

**Linux:** `curl "http://<server-ip>:8000/part-classification/dataset/files?label_id=30"`

#### Dataset thumbnail preview

Returns the URL of a PNG grid of dataset thumbnails for a given class (same pattern as `/similarity/search`).

```
GET /part-classification/dataset/preview?label_id=<0-44>&k=25&grid_cols=8
```

**Windows (PowerShell):**
```powershell
curl.exe "http://<server-ip>:8000/part-classification/dataset/preview?label_id=30&k=25"
```

**Linux:**
```bash
curl "http://<server-ip>:8000/part-classification/dataset/preview?label_id=30&k=25"
```

**Response:**

```json
{
  "label_id": 30,
  "part_name": "Gears",
  "image_url": "http://<server-ip>:8000/out/<uuid>.png"
}
```

Open `image_url` in a browser to view the thumbnail grid.

> **Note:** Thumbnails are rendered from the `stream_cache/` folder inside the flow directory. This folder is populated when running the ETL step of `3c_workflow_for_Part_classification_fabwave.ipynb`. If `stream_cache/` is empty, the image grid will show "No Preview" placeholders.

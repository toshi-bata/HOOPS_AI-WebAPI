# HOOPS AI MCP

A solution combining the HOOPS AI WebAPI server and MCP server for Claude Desktop integration.

## Structure

```
HOOPS_AI-MCP/
├── webapi/         # FastAPI-based REST API (HOOPS AI WebAPI)
└── mcp_server/     # MCP server for Claude Desktop integration
```

---

## 1. Clone the repository

```bash
git clone https://github.com/toshi-bata/HOOPS_AI-MCP.git
cd HOOPS_AI-MCP
```

---

## 2. WebAPI Server Setup

### Requirements

- Python 3.9 (recommended: Miniconda/Anaconda environment)
- A valid **HOOPS AI license key**
- HOOPS AI Python package (`hoops_ai_cpu` or `hoops_ai_gpu`) installed in your environment

### Install dependencies

Open a terminal, move into the `webapi` folder, and install:

```bash
cd webapi
pip install -r requirements.txt
```

> Install the `hoops_ai_cpu` or `hoops_ai_gpu` package separately according to your HOOPS AI distribution instructions.

### Configure environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` and set each variable:

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
```

### Start the WebAPI server

Run the following from the `webapi/` directory using the Python executable from your HOOPS AI conda environment:

```bash
cd webapi
C:\Users\user_name\miniconda3\envs\hoops_ai_cpu\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001
```

For development with auto-reload:

```bash
cd webapi
C:\Users\user_name\miniconda3\envs\hoops_ai_cpu\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001 --reload
```

The API will be available at `http://127.0.0.1:8001`.  
Interactive docs (Swagger UI) are at `http://127.0.0.1:8001/docs`.

---

### API Usage

#### MFR — Search files by feature

Returns CAD file names that contain a given manufacturing feature.

```
GET /MFR/files/search?feature_name=<name>
```

**Example:**

```bash
curl.exe "http://127.0.0.1:8001/MFR/files/search?feature_name=through%20hole"
# Windows PowerShell
curl.exe "http://127.0.0.1:8001/MFR/files/search?feature_name=circular%20blind%20step"
```

**Response:**

```json
{
  "file_names": ["bracket_a.stp", "housing_b.stp"],
  "file_list": [1, 3]
}
```

---

#### MFR — File thumbnail

Returns the thumbnail PNG image for a given file ID.

```
GET /MFR/files/{file_id}/thumbnail
```

**Example:**

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8001/MFR/files/1/thumbnail" -OutFile "thumbnail.png"
```

**Response:** PNG image (`image/png`)

---

#### MFR — List label descriptions

Returns all MFR label IDs with their names and descriptions.

```
GET /MFR/labels/description
```

**Example:**

```bash
curl.exe "http://127.0.0.1:8001/MFR/labels/description"
```

---

#### MFR — Dataset table of contents

Returns a summary of the loaded MFR dataset.

```
GET /MFR/dataset/table-of-contents
```

**Example:**

```bash
curl.exe "http://127.0.0.1:8001/MFR/dataset/table-of-contents"
```

---

#### MFR — Inference

Upload a CAD file and run MFR inference. Launches the CAD viewer and returns predictions, probabilities, and viewer_url.

```
POST /MFR/inference
```

**Example:**

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

---

#### MFR — Colorize viewer

Apply MFR prediction colors to the last active CAD viewer. Call this after the viewer has fully loaded.

```
POST /MFR/viewer/colorize
```

**Example:**

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

#### CAD Viewer — Browser UI

Open the browser and navigate to:

```
http://127.0.0.1:8001/CAD/viewer
```

The page shows two forms:

1. **Upload CAD file** — choose a local file and click *Open viewer*
2. **CAD file path in shared folder** — enter a filename or path relative to the shared folder and click *Open viewer from path*

Both forms submit to the API and return a JSON response containing `viewer_url`. Copy that URL and open it in your browser to launch the interactive 3D viewer.

> The viewer runs on a **separate port** from the API server. Make sure that port is accessible (not blocked by a firewall).

#### CAD Viewer — Upload via API

```bash
# Windows PowerShell
curl.exe -X POST "http://127.0.0.1:8001/CAD/viewer" `
         -F "file=@C:\path\to\model.stp"
```

**Response:**

```json
{
  "viewer_url": "http://127.0.0.1:<viewer_port>/index.html"
}
```

Open the returned `viewer_url` in your browser to view the model.

#### CAD Viewer — Open by shared path

> This endpoint is used internally by the browser UI form and is not listed in the Swagger docs.

```bash
# Windows PowerShell
curl.exe -X POST "http://127.0.0.1:8001/CAD/viewer/from-path" `
         -d "cad_file_path=model.stp"
```

The path can be a filename relative to the shared folder (`HOOPS_AI_CAD_SHARED_DIR`) or an absolute path within it.

**Response:**

```json
{
  "viewer_url": "http://127.0.0.1:<viewer_port>/index.html"
}
```

Open the returned `viewer_url` in your browser to view the model.

---

#### CAD Viewer — Terminate

Terminate the last active viewer, or all viewers.

```
DELETE /CAD/viewer
DELETE /CAD/viewer?all=true
```

**Example:**

```powershell
# Terminate last viewer
Invoke-RestMethod -Method Delete -Uri "http://127.0.0.1:8001/CAD/viewer"

# Terminate all viewers
Invoke-RestMethod -Method Delete -Uri "http://127.0.0.1:8001/CAD/viewer?all=true"
```

**Response:**

```json
{ "terminated": 1 }
```

---

### Running tests

```bash
python -m unittest discover -s tests
```

---

## 3. MCP Server Setup (Claude Desktop)

The MCP server connects Claude Desktop to the HOOPS AI WebAPI.

### Register the MCP server in Claude Desktop

1. Open **Claude Desktop**
2. Go to **Settings** → **Developer** → **Edit Config**
3. This opens `claude_desktop_config.json`. Add the following entry under `mcpServers`:

```json
{
  "mcpServers": {
    "hoops-ai": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "C:\\path\\to\\HOOPS_AI-MCP\\mcp_server",
        "server.py"
      ]
    }
  }
}
```

> Replace `C:\\path\\to\\HOOPS_AI-MCP` with the actual path where you cloned this repository.

4. Save the file and restart Claude Desktop.

### Available MCP tools

| Tool | Description |
|---|---|
| `open_cad_viewer` | Open a CAD file in the 3D viewer and return the viewer URL |
| `get_MFR_table_of_contents` | Get the MFR dataset summary |
| `get_MFR_labels_description` | List all MFR label IDs, names, and descriptions |
| `search_MFR_files` | Find CAD files that contain a given manufacturing feature |
| `get_MFR_file_thumbnail` | Download the thumbnail PNG for a file ID, returns base64-encoded PNG |
| `run_MFR_inference` | Run MFR inference on a local CAD file, launches viewer, returns predictions + viewer_url |
| `colorize_MFR_viewer` | Apply MFR prediction colors to the last active viewer, returns color_map |
| `terminate_CAD_viewer` | Terminate the last (or all) active CAD viewer(s) |

---

## License

This project uses the **HOOPS AI** library. A valid HOOPS AI license is required to run the server.  
Contact [Tech Soft 3D](https://hoops.com/) for licensing information.

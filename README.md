# HOOPS AI MCP

**HOOPS AI MCP** is a platform for intelligent 3D CAD data analysis powered by [HOOPS AI](https://www.techsoft3d.com/developers/products/hoops-ai/) (Tech Soft 3D). It wraps HOOPS AI capabilities as a FastAPI REST API and exposes them to Claude Desktop via an MCP Server, enabling engineers to perform advanced 3D CAD analysis through natural language — no code required.

## Features

| Feature | Description |
|---|---|
| **3D CAD Viewer** | Renders 30+ formats (STEP, SolidWorks, CATIA, NX, …) interactively in the browser |
| **B-Rep Analysis** | Generates face adjacency graphs; extracts face/edge attributes (type, area, length, dihedral angle) |
| **Manufacturing Feature Recognition (MFR)** | Recognizes 24 machining feature types (holes, slots, pockets, …) using a trained ML model; colorizes results in the viewer |
| **Shape Similarity Search** | Converts shapes to feature vectors with HOOPS Embeddings and retrieves similar parts via a FAISS index (accuracy ≥ 0.99) |

## Architecture

```
Claude Desktop  ──(natural language)──▶  MCP Server
                                              │
                                        REST API calls
                                              │
                                         WebAPI (FastAPI)
                                              │
                                          HOOPS AI
                                     (Tech Soft 3D)
```

- **HOOPS AI** — 3D CAD file loading, geometry encoding, ML inference
- **WebAPI (FastAPI)** — Exposes HOOPS AI as 11 REST endpoints; leverages HOOPS AI's Python API directly
- **MCP Server** — Wraps the WebAPI; bridges Claude Desktop via the MCP protocol
- **Claude Desktop** — Chat AI that autonomously calls MCP tools to carry out end-to-end analysis

## Repository Structure

```
HOOPS_AI-MCP/
├── webapi/         # FastAPI REST API  →  see webapi/README.md
└── mcp_server/     # MCP server for Claude Desktop  →  see mcp_server/README.md
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/toshi-bata/HOOPS_AI-MCP.git
cd HOOPS_AI-MCP
```

### 2. Set up the WebAPI server

See **[webapi/README.md](webapi/README.md)** for full instructions.

**Windows:**
```bat
cd webapi
pip install -r requirements.txt
copy .env.example .env   # then edit .env with your HOOPS AI license key
<Path\to\HOOPS_AI\install\dir>\.venv\Scripts\python.exe main.py --host 0.0.0.0 --port 8000
```

**Linux:**
```bash
cd webapi
pip install -r requirements.txt
cp .env.example .env     # then edit .env with your HOOPS AI license key
/path/to/HOOPS_AI/install/dir/.venv/bin/python main.py --host 0.0.0.0 --port 8000
```

### 3. Set up the MCP server (Claude Desktop)

See **[mcp_server/README.md](mcp_server/README.md)** for full instructions.

Register the MCP server in `claude_desktop_config.json`:

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

> **Same machine:** The config above works as-is — the MCP server defaults to `http://127.0.0.1:8000`.  
> **Different machine:** Add `"env": {"HOOPS_WEBAPI_URL": "http://<server-ip>:8000"}` to the config.  
> See **[mcp_server/README.md](mcp_server/README.md)** for details.

---

## Tested Environment

| Component | Version |
|---|---|
| HOOPS AI | V1.1 |
| Python | 3.12 |
| OS | Windows 11, Linux (Ubuntu 22.04) |

---

## License

This project uses the following **Tech Soft 3D** libraries:

| Library | Required for |
|---|---|
| **HOOPS AI** | CAD file loading, B-Rep analysis, MFR inference, shape similarity search |
| **HOOPS Web Viewer** | 3D CAD Viewer feature (`open_cad_viewer`, `run_MFR_inference`) in a server–client environment |

> **Note:** In a server–client deployment where the WebAPI server and the browser client run on separate machines, a valid **HOOPS Web Viewer** license is required in addition to the HOOPS AI license.

Contact [Tech Soft 3D](https://www.techsoft3d.com/contact/) for licensing information.


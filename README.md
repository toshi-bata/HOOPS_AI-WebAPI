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
├── mcp_server/     # MCP server for Claude Desktop  →  see mcp_server/README.md
└── demo/           # Demo slides and narration scripts
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

```bash
cd webapi
pip install -r requirements.txt
copy .env.example .env   # then edit .env with your HOOPS AI license key
cd ..
C:\Users\<user>\miniconda3\envs\hoops_ai_cpu\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001
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

---

## Demo

A slide deck and narration scripts are included in the [`demo/`](demo/) folder:

- [`demo/narration_en.md`](demo/narration_en.md) — English narration
- [`demo/narration_ja.md`](demo/narration_ja.md) — Japanese narration
- [`demo/index_en.html`](demo/index_en.html) — English slide deck
- [`demo/index_jp.html`](demo/index_jp.html) — Japanese slide deck

### What the demo covers

1. Ask Claude what HOOPS AI tools are available
2. Display a 3D CAD file (`helloworld.stp`) in the browser viewer
3. Run B-Rep analysis on a flange model and display organized results
4. Query the MFR dataset overview in natural language
5. Run MFR inference on a SOLIDWORKS part and colorize by feature type
6. Perform shape similarity search and review results

---

## License

This project uses the **HOOPS AI** library. A valid HOOPS AI license is required to run the server.  
Contact [Tech Soft 3D](https://www.techsoft3d.com/developers/products/hoops-ai/) for licensing information.


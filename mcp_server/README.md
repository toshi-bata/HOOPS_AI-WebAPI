# HOOPS AI MCP Server

An MCP (Model Context Protocol) server that bridges [Claude Desktop](https://claude.ai/download) to the HOOPS AI WebAPI.  
With this server registered in Claude Desktop, users can perform 3D CAD analysis through natural language — no code required.  
See the root [README](../README.md) for an overview of the full platform.

---

## Prerequisites

- [uv](https://github.com/astral-sh/uv) installed on the **Claude Desktop machine** (Claude Desktop uses `uv` to launch the MCP server process)
- The **WebAPI server** running and accessible (default: `http://127.0.0.1:8000`)  
  → See [webapi/README.md](../webapi/README.md) for setup instructions

---

## Setup

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

**Same machine (default):**

No additional configuration is needed.  
The MCP server defaults to `http://127.0.0.1:8000`, so if the WebAPI server is running on the same machine, the basic config above works as-is.

**When the WebAPI server is on a different machine (client-server setup):**

Add `"env": {"HOOPS_WEBAPI_URL": "..."}` to the config — no system environment variable is needed.  
Claude Desktop passes this value to the MCP server process automatically:

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
      ],
      "env": {
        "HOOPS_WEBAPI_URL": "http://192.168.0.6:8000"
      }
    }
  }
}
```

> Replace `192.168.0.6` with the actual IP address of the machine running the WebAPI server.  
> This is the **only configuration change needed** on the client machine.

4. Save the file and **restart Claude Desktop**.

---

## Available MCP Tools

Claude Desktop can call these 11 tools using natural language:

| Tool | Description |
|---|---|
| `open_cad_viewer` | Upload a local CAD file to the server and open it in the interactive 3D browser viewer. Returns `viewer_url` and `image_url` (PNG preview). |
| `terminate_CAD_viewer` | Terminate the last active viewer, or all viewers (`terminate_all=True`). |
| `get_brep_adjacency_graph` | Build a face adjacency graph from a local CAD file. Returns graph data (nodes, edges, counts) and `image_url` (PNG visualization URL). |
| `get_brep_attributes` | Extract face and edge attributes (types, areas, lengths, dihedral angles, etc.) from a local CAD file. |
| `get_MFR_table_of_contents` | Get a summary of the Manufacturing Feature Recognition (MFR) dataset. |
| `get_MFR_labels_description` | List all MFR label IDs, feature names, and descriptions. |
| `search_MFR_files` | Find CAD files in the MFR dataset that contain a given manufacturing feature. |
| `get_MFR_file_thumbnail` | Download the thumbnail PNG for a given dataset file ID. Returns base64-encoded PNG. |
| `run_MFR_inference` | Run MFR inference on a local CAD file. Launches the viewer and returns predictions, probabilities, `viewer_url`, and `image_url` (PNG preview). |
| `colorize_MFR_viewer` | Apply MFR prediction colors to the last active viewer. Returns the color map. Call only after the model has fully loaded in the browser. |
| `search_similar_shapes` | Upload a local CAD file and find the top-k most similar parts using HOOPS Embeddings and a FAISS index. Returns match IDs, similarity scores, and `image_url` (result grid image URL). |

---

## Example Usage in Claude Desktop

Once the MCP server is registered and the WebAPI server is running, you can chat with Claude:

```
What HOOPS AI tools are available?
```

```
"C:\temp\helloworld.stp" — please display this 3D CAD file.
```

```
"C:\temp\Flange287.stp" — show this model and give me its B-Rep information.
```

```
Tell me about the manufacturing feature recognition dataset.
```

```
"C:\temp\nist_ftc_06_asme1_rd_sw1802.SLDPRT" — run manufacturing feature recognition and colorize by feature type.
```

```
"C:\temp\idler_sprocket.step" — search for similar parts to this component.
```

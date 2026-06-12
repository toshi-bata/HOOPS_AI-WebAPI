# HOOPS AI MCP

A solution combining the HOOPS AI WebAPI server and MCP server.

## Structure

```
HOOPS_AI-MCP/
├── webapi/         # FastAPI-based REST API (HOOPS AI WebAPI)
└── mcp_server/     # MCP server for Claude integration
```

## Projects

### webapi
FastAPI server exposing HOOPS AI capabilities (MFR search, CAD Viewer).  
See [webapi/README.md](webapi/README.md) for setup and usage.

### mcp_server
MCP server that connects Claude to the HOOPS AI WebAPI.  
See [mcp_server/README.md](mcp_server/README.md) for setup and usage.

## Quick Start

1. Start the WebAPI server:
```
C:\Users\user_name\miniconda3\envs\hoops_ai_cpu\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001
```
*(run from the `webapi/` directory)*

2. Connect Claude Desktop to the MCP server via `mcp_server/server.py`.

# HOOPS AI MCP — Demo Narration (English)

---

## Slide 1 — Title

Today I'd like to introduce HOOPS AI MCP — a platform for intelligent CAD data analysis. It delivers four core capabilities: CAD Viewer, B-Rep Analysis, Manufacturing Feature Recognition, and Shape Similarity Search. These are accessible through two interfaces: a REST API built with FastAPI, and an MCP Server that lets Claude Desktop control everything using natural language.

---

## Slide 2 — Architecture

The system is organized in four layers. At the foundation is HOOPS AI by Tech Soft 3D, which handles CAD file loading, geometry analysis, and ML inference. On top of that sits the FastAPI WebAPI, which exposes those capabilities as REST endpoints. The MCP Server wraps the WebAPI and acts as a bridge, allowing Claude Desktop to invoke tools using natural language. Finally, Claude Desktop operates as the AI agent — autonomously calling MCP tools to carry out end-to-end CAD analysis tasks.

---

## Slide 3 — Features

Let me walk through the four features. First, the CAD Viewer renders 30+ formats — including STEP, SolidWorks, CATIA, and NX — interactively in the browser. B-Rep Analysis generates face adjacency graphs and extracts face and edge attributes such as type, area, length, and dihedral angle. MFR uses a trained ML model to automatically recognize 24 machining feature types like holes, slots, and pockets, with results visualized as color overlays in the viewer. Finally, CAD Similarity Search converts shapes into feature vectors with HOOPS Embeddings and retrieves similar parts at high speed using a FAISS index.

---

## Slide 4 — WebAPI Endpoints

The WebAPI organizes endpoints by feature. It offers 11 endpoints in total: CAD Viewer launch and termination; B-Rep face adjacency graph generation and attribute extraction; MFR file search, thumbnail retrieval, label listing, inference execution, and viewer colorization; and shape similarity search. All endpoints are immediately testable through the built-in Swagger UI.

---

## Slide 5 — MCP Tools

MCP Tools are the toolset available to Claude Desktop. We provide 11 tools mapped to each WebAPI feature. Claude can launch the viewer, recognize machining features, or search for similar shapes — all from natural language instructions alone. The key advantage is that engineers can conduct interactive CAD analysis without writing any code.

---

## Slide 6 — Summary

To summarize: HOOPS AI MCP delivers CAD intelligence through two complementary approaches — a REST API and AI agent integration. B-Rep analysis and manufacturing feature recognition provide deep understanding of CAD data, while the combination of HOOPS Embeddings and FAISS has demonstrated similarity search accuracy of 0.99 and above. We invite you to explore the new possibilities this platform brings to CAD data utilization.

---

## Live Demo Script

Let's now take a look at the actual demo.

HOOPS AI is pre-installed on the server. This is the server that exposes HOOPS AI capabilities as a REST API. When the server starts, a license key is passed to HOOPS AI and it becomes activated.

This is the MCP server code. This MCP server has been registered in the Claude Desktop configuration.

Let's go ahead and access HOOPS AI from Claude Desktop.

Let's ask: "Hello HOOPS AI, what can you do?" Claude accesses HOOPS AI via MCP and summarizes the information it retrieves.

It looks like CAD viewing is available, so let's have it display a CAD file. (`"C:\temp\helloworld.stp" Please display this CAD file.`)
The file is uploaded to the server and the viewer is launched. When the viewer starts, a URL is returned — opening that link displays the 3D model in the browser.

B-Rep analysis is also available, so let's have it look into another model. (`"C:\temp\Flange287.stp" Please display this model and tell me about it.`)
A separate viewer instance is launched. Claude takes the B-Rep data returned by HOOPS AI and presents it in a clear, organized way.

Let's ask for an overview of the manufacturing feature recognition dataset. (`Tell me about the manufacturing feature recognition dataset.`)
HOOPS AI returns the data in a complex JSON format, but Claude summarizes it clearly.

Next, let's have it run manufacturing feature recognition. (`"C:\temp\nist_ftc_06_asme1_rd_sw1802.SLDPRT" Please run manufacturing feature recognition on this model.`)
Then let's launch the viewer and colorize the model by feature type. (`Please colorize it.`)
Claude also generates a color legend to make the results easy to understand.

Finally, let's run a shape similarity search. (`"C:\temp\idler_sprocket.step" Please search for similar parts to this component.`)
Claude explains the results returned by HOOPS AI in an easy-to-understand way.

HOOPS AI — previously operated by AI engineers and data scientists using Python in Jupyter Notebooks — can now be used by anyone, without writing any code, simply by wrapping it in a Web API and exposing it through MCP to AI tools like Claude Desktop. Furthermore, the complex data returned by HOOPS AI is organized and communicated clearly by Claude's AI model, leading to significant improvements in workflow efficiency.

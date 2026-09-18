import os

MCP_HOST = os.environ.get("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.environ.get("MCP_PORT", "8765"))

# kb_api owns the embedding model; kb_mcp calls its /internal/search route
# instead of importing kb_common.hybrid_search/embed itself, so this process
# never needs torch/sentence-transformers or a GPU.
KB_API_URL = os.environ.get("KB_API_URL", "http://localhost:8080")

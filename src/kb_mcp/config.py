import os

MCP_HOST = os.environ.get("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.environ.get("MCP_PORT", "8765"))

MCP_ALLOWED_HOSTS = [
    h.strip()
    for h in os.environ.get("MCP_ALLOWED_HOSTS", "localhost:8765,127.0.0.1:8765").split(",")
    if h.strip()
]

KB_API_URL = os.environ.get("KB_API_URL", "http://localhost:8080")

KB_MCP_INTERNAL_TOKEN = os.environ.get("KB_MCP_INTERNAL_TOKEN", "")

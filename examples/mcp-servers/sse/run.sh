#!/bin/bash
# Run SSE MCP Calculator Server

echo "Starting SSE MCP Calculator Server..."
echo "Server will be available at: http://localhost:8001"
echo "SSE endpoint: http://localhost:8001/mcp/sse"
echo "Press Ctrl+C to stop"

cd "$(dirname "$0")"
python3 sse_server.py
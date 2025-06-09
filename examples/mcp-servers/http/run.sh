#!/bin/bash
# Run HTTP MCP Calculator Server

echo "Starting HTTP MCP Calculator Server..."
echo "Server will be available at: http://localhost:8003"
echo "HTTP endpoint: http://localhost:8003/mcp"
echo "Simple HTTP endpoint: http://localhost:8003/mcp/simple"
echo "Press Ctrl+C to stop"

cd "$(dirname "$0")"
python3 http_server.py
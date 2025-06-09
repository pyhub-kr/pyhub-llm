#!/bin/bash
# Run WebSocket MCP Calculator Server

echo "Starting WebSocket MCP Calculator Server..."
echo "Server will be available at: http://localhost:8002"
echo "WebSocket endpoint: ws://localhost:8002/mcp/ws"
echo "Simple WebSocket endpoint: ws://localhost:8002/mcp/ws/simple"
echo "Press Ctrl+C to stop"

cd "$(dirname "$0")"
python3 ws_server.py
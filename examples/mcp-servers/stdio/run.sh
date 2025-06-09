#!/bin/bash
# Run STDIO MCP Calculator Server

echo "Starting STDIO MCP Calculator Server..."
echo "This server communicates via standard input/output"
echo "Press Ctrl+C to stop"

cd "$(dirname "$0")"
python3 calculator_server.py
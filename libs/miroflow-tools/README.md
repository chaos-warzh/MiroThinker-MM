# MiroFlow Tools

Tool management and MCP server utilities for MiroFlow.

## Features

- Tool manager for handling MCP (Model Context Protocol) servers
- Various MCP servers for different functionalities:
  - Python code execution
  - Vision and image processing
  - Audio transcription
  - Web searching
  - Reasoning engines
  - Document reading

## Installation

```bash
pip install miroflow-tools
```

## Usage

```python
from miroflow_tools.manager import ToolManager

# Initialize tool manager with server configurations
tool_manager = ToolManager(server_configs)

# Execute tool calls
result = await tool_manager.execute_tool_call(
    server_name="tool-python",
    tool_name="run_python_code",
    arguments={"code": "print('Hello, World!')"}
)
```

## Development

This package is part of the MiroFlow project and is developed alongside the main application. 
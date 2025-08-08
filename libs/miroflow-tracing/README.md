# MiroFlow Tracing

Distributed tracing utilities for MiroFlow applications.

## Features

- Distributed tracing with OpenTelemetry
- Span and trace management
- OTLP (OpenTelemetry Protocol) exporters
- Function and method tracing decorators
- Trace provider configuration

## Installation

```bash
pip install miroflow-tracing
```

## Usage

```python
from miroflow_tracing import trace, function_span

# Trace a function
@trace("my_function")
async def my_function():
    # Function implementation
    pass

# Create a span
@function_span("my_span")
async def my_span_function():
    # Span implementation
    pass
```

## Development

This package is part of the MiroFlow project and is developed alongside the main application. 
# Quick Start Guide

## Using the CLI

You can run the benchmark using the command-line interface:

```bash
python3 -m run --engine <engine> --tasks <tasks> --limit <limit> --save_outputs
```

### Parameters:
- `engine`: The engine implementation to benchmark
- `tasks`: The tasks to run
- `limit`: Maximum number of samples to run on each task
- `save_outputs`: Save execution outputs for later analysis

## Analyzing Results

If you have saved outputs, you can generate a report:

```bash
python3 -m analyze --outputs <outputs_path>
```

## Using the Python API

You can also create a Python script to use the library directly. This approach allows you to create a custom engine and run the benchmark with more flexibility.

```python
from core.bench import bench
from core.engine import Engine

# Initialize your engine with configuration
engine = Engine(config=config)

# Run benchmark
outputs = bench(engine, tasks, limit=limit, save_outputs=True)
```

For instructions on creating your custom engine, see the [Custom Engine Tutorial](/docs/custom_engine.md).

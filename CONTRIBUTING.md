# Contributing to Sentinel-AIOps

First off, thank you for considering contributing to Sentinel-AIOps! It's people like you that make Sentinel-AIOps such a great event-driven MLOps framework.

## 🌪️ How to Add "Chaos" Scenarios

The `stream_simulator.py` uses Chaos Engineering principles to test our Population Stability Index (PSI) drift monitor. If you want to add a new anomaly injection vector:

1. Open `data/stream_simulator.py`.
2. Locate the `_generate_log()` function and the `is_chaos` condition block.
3. Add your custom OOD (Out-of-Distribution) multiplier.

**Example: Simulating a massive CPU spike:**

```python
if is_chaos:
    # Existing chaos logic
    ...
    # Your new CPU spike simulation (e.g., 10x the normal max)
    cpu = min(100.0, NORMAL_RANGES["cpu_usage_pct"][1] * 10 * random.uniform(0.9, 1.0))
```

## 🧪 Testing Your Contributions

1. Run the local test suite using `pytest`:
   ```bash
   python -m pytest tests/ -v
   ```
2. Verify PEP8 compliance using `flake8`:
   ```bash
   flake8 models/ mcp-server/ data/stream_simulator.py dashboard/ tests/
   ```

Our GitHub Actions CI pipeline will automatically run these checks on every Pull Request.

## 🐛 Bug Reports

If you find a bug, please create an issue on GitHub with:
1. Steps to reproduce the bug
2. Expected behavior
3. Actual behavior
4. Versions of Python, Docker, and any relevant libraries.

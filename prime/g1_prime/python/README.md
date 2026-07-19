# Python package source

Setuptools package root for the installable [`g1cal`](g1cal/README.md) library
and CLI. The parent [`pyproject.toml`](../pyproject.toml) owns its metadata.

The C++ build places `_g1cal_cpp` beside the package modules. Install from the
implementation root after building:

```bash
python -m pip install -e .
g1cal --help
```

Return to the [G1 implementation](../README.md).

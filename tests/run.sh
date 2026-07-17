#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python3}"

PYTHONPATH="$root/python" "$python_bin" -m pytest "$root/python/tests"

if [[ -x "$root/prime/.build/prime_fie" ]]; then
  PYTHONPATH="$root/prime" "$python_bin" -m pytest "$root/prime/tests"
fi

if command -v matlab >/dev/null 2>&1; then
  matlab -batch "cd('$root/matlab'); addpath('tests'); test_fast_fie"
fi

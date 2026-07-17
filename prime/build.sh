#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cmake -Wno-author -S "$ROOT" -B "$ROOT/.build" -GNinja \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/.build" --parallel

#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BUILD_TYPE="${1:-Release}"
NPROC=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"

echo "=== Building ExecuTorch LLM libraries (${BUILD_TYPE}) ==="
# Clean previously installed headers so --fresh configure produces consistent
# results (--fresh only removes the CMake cache, not installed artifacts).
rm -rf cmake-out/include cmake-out/lib
cmake --workflow --preset "llm-$(echo "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')" --fresh

echo ""
echo "=== Building Llama runner (${BUILD_TYPE}) ==="
cd examples/models/llama
cmake --workflow --preset "llama-$(echo "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')" --fresh
cd "$SCRIPT_DIR"

echo ""
echo "=== Build complete ==="
echo "Binary: cmake-out/examples/models/llama/llama_main"

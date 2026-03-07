# Build and Run Samsung `enn_executor_runner` on Host PC

Build `enn_backend` and `enn_executor_runner` without Android NDK or the
Exynos AI LiteCore SDK, then run models on a host machine.

## Prerequisites

- ExecuTorch installed (`./install_executorch.sh` or `pip install -e .`)
- A `.pte` model file (e.g., `add.pte` or any portable-ops model)

## Build

```bash
./backends/samsung/build.sh --build android
```

This builds for the host machine despite the `android` name. The Android
cross-compilation flags (NDK toolchain, ABI, platform) have been removed.
The SDK is not required because the ENN API is loaded via `dlopen` at runtime.

### What changed from the original Android build

| Removed | Why |
|---------|-----|
| `-DCMAKE_TOOLCHAIN_FILE` (NDK toolchain) | Builds for host instead of ARM |
| `-DANDROID_NDK`, `-DANDROID_ABI`, `-DANDROID_PLATFORM` | No cross-compilation |
| `-DEXYNOS_AI_LITECORE_ROOT` | SDK not needed at compile time |
| `android` and `log` link libraries | Not available on host |

Added `-DEXECUTORCH_BUILD_ENN_BACKEND=ON` to trigger the `enn_backend` build
without setting the global `ANDROID` CMake variable (which would break other
libraries like cpuinfo).

### Build outputs

- `build_samsung_android/lib/libenn_backend.a`
- `build_samsung_android/backends/samsung/enn_executor_runner`

## Run

```bash
./build_samsung_android/backends/samsung/enn_executor_runner --model model.pte
```

When the Samsung NPU library (`libenn_public_api_cpp.so`) is not available,
the runner logs a warning and skips ENN initialization. The model executes
using portable ops instead.

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `model.pte` | Path to the `.pte` model file |
| `--input` | (empty) | Input file paths, space-separated |
| `--num_executions` | `1` | Number of inference iterations |
| `--warm_up` | `0` | Warm-up iterations before timing |
| `--dump_statistics` | `false` | Write timing to `statistics.txt` |
| `--output_path` | (empty) | Directory to save output tensors |

### Example with inputs

```bash
./build_samsung_android/backends/samsung/enn_executor_runner \
  --model model.pte \
  --input "input_0.bin input_1.bin" \
  --num_executions 10 \
  --warm_up 3 \
  --dump_statistics
```

## Files modified

| File | Change |
|------|--------|
| `backends/samsung/CMakeLists.txt` | SDK optional (warning not fatal), removed `android`/`log` link deps, added `EXECUTORCH_BUILD_ENN_BACKEND` guard |
| `backends/samsung/build.sh` | Removed NDK/toolchain flags from `build_android()` |
| `backends/samsung/runtime/CMakeLists.txt` | Widened source guard to `ANDROID OR EXECUTORCH_BUILD_ENN_BACKEND` |
| `examples/samsung/executor_runner/enn_executor_runner.cpp` | Graceful skip when ENN API library unavailable |

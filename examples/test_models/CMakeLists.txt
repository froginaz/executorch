# CMakeLists.txt for Mamba runner
#
# Build:
#   cd /path/to/executorch
#   mkdir build_mamba && cd build_mamba
#   cmake ../examples/test_models \
#       -DCMAKE_PREFIX_PATH="<executorch_install_dir>" \
#       -DEXECUTORCH_ROOT=..
#   make -j$(nproc)
#
# Run:
#   ./mamba_runner --model_path=mamba.pte --prompt="1,2,3" --max_tokens=20

cmake_minimum_required(VERSION 3.19)
project(mamba_runner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

# Find ExecuTorch packages
find_package(executorch REQUIRED
  COMPONENTS
    executorch
    extension_data_loader
    extension_runner_util
    portable_kernels
    portable_ops_lib
)

# Optional: XNNPACK backend
find_package(xnnpack_backend QUIET)

# gflags
find_package(gflags REQUIRED)

add_executable(mamba_runner mamba_runner.cpp)

target_link_libraries(mamba_runner
  PRIVATE
    executorch::executorch
    executorch::extension_data_loader
    executorch::extension_runner_util
    executorch::portable_kernels
    executorch::portable_ops_lib
    gflags
)

if(xnnpack_backend_FOUND)
  target_link_libraries(mamba_runner PRIVATE xnnpack_backend::xnnpack_backend)
  target_compile_definitions(mamba_runner PRIVATE ET_USE_XNNPACK=1)
endif()

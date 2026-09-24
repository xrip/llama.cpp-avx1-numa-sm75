#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Build the SM75 cuBLAS path and its RPC server.
cmake -S "$root" -B "$root/build-best" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_RPC=ON \
    -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_HOST_COMPILER=g++-14 -DCMAKE_CUDA_ARCHITECTURES=75 \
    -DGGML_NATIVE=ON -DGGML_OPENMP=ON \
    -DGGML_CUDA=ON -DGGML_VULKAN=OFF -DGGML_BLAS=OFF \
    -DGGML_CUDA_FORCE_MMQ=OFF -DGGML_CUDA_FORCE_CUBLAS=ON \
    -DGGML_CUDA_SM75_Q8_0_FAST_HALF=ON \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON
cmake --build "$root/build-best" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-12}" \
    --target llama-server llama-cli llama-bench test-backend-ops ggml-rpc-server

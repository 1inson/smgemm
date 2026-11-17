#!/usr/bin/env bash
# Reproduce NVIDIA's SM120 79a example on a rented Linux GPU machine.
# This script has not been run against a cloud GPU in this project.
# It downloads CUTLASS only when explicitly run and never updates an existing clone.
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cutlass_dir="$root/third_party/cutlass"
build_dir="$root/build/cutlass-79a"
tag=v4.3.6
target=79a_blackwell_geforce_nvfp4_bf16_gemm

if [[ $(uname -s) != Linux ]]; then
  echo 'Run this script on the Linux GPU instance.' >&2
  exit 1
fi
for tool in git cmake nvcc nvidia-smi; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Missing $tool. Install a supported CUDA development image and build tools first." >&2
    exit 1
  fi
done

echo 'GPU and driver:'
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version --format=csv
echo 'CUDA compiler:'
nvcc --version
if ! nvcc --list-gpu-code | grep -Fxq 'sm_120a'; then
  echo 'This nvcc does not report sm_120a. Use a suitable CUDA Toolkit (12.8+).' >&2
  exit 1
fi

if [[ ! -e "$cutlass_dir" ]]; then
  mkdir -p "$(dirname "$cutlass_dir")"
  echo "Cloning NVIDIA CUTLASS $tag into $cutlass_dir"
  git clone --depth 1 --branch "$tag" https://github.com/NVIDIA/cutlass.git "$cutlass_dir"
else
  # Do not change a user's existing checkout or continue after a partial clone.
  if [[ ! -d "$cutlass_dir/.git" ]] || ! git -C "$cutlass_dir" describe --exact-match --tags HEAD 2>/dev/null | grep -Fxq "$tag"; then
    echo "Existing $cutlass_dir is not an exact $tag checkout; leaving it untouched." >&2
    echo 'Move it aside yourself or use a separate project directory.' >&2
    exit 1
  fi
fi
echo "CUTLASS revision: $(git -C "$cutlass_dir" rev-parse HEAD)"
echo 'Configuring and building only the 79a example; this can take several minutes.'
cmake -S "$cutlass_dir" -B "$build_dir" -DCUTLASS_NVCC_ARCHS=120a -DCUTLASS_ENABLE_TESTS=OFF
cmake --build "$build_dir" --target "$target" --parallel "${BUILD_JOBS:-2}"

binary="$build_dir/examples/79_blackwell_geforce_gemm/$target"
if [[ ! -x "$binary" ]]; then
  echo "Executable missing: $binary" >&2
  exit 1
fi
output=$(mktemp)
trap 'rm -f "$output"' EXIT
echo 'Running example with synthetic input; the host reference calculation can be slow.'
"$binary" --m=1024 --n=1024 --k=1024 --iterations=10 2>&1 | tee "$output"
if ! grep -Eq 'Disposition:[[:space:]]*Passed' "$output" || ! grep -Eq 'Avg runtime:' "$output"; then
  echo 'Example did not report both verification and timing; do not treat this run as a pass.' >&2
  exit 1
fi
echo 'Official synthetic-input GEMM passed. It is not connected to this project quantizer.'

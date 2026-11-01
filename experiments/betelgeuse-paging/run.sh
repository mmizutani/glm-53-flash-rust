#!/usr/bin/env bash
set -euo pipefail
source_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
unset CDPATH
output_dir=${1:?usage: run.sh OUTSIDE_GIT_EVIDENCE_DIRECTORY}
mkdir -p -- "$output_dir"
output_dir=$(CDPATH= cd -- "$output_dir" && pwd -P)
repo_dir=$(CDPATH= cd -- "$source_dir/../.." && pwd -P)
case "$output_dir/" in "$repo_dir/"*) echo 'Evidence must be outside the source checkout' >&2; exit 2;; esac
export RUSTUP_HOME=${RUSTUP_HOME:-$HOME/.cache/betelgeuse-experiment-tools/rustup}
export CARGO_HOME=${CARGO_HOME:-$HOME/.cache/betelgeuse-experiment-tools/cargo}
rustc=${RUSTC:-$CARGO_HOME/bin/rustc}
"$rustc" --version > "$output_dir/toolchain.txt"
/usr/local/cuda/bin/nvcc --version >> "$output_dir/toolchain.txt"
nvidia-smi --query-gpu=name,driver_version --format=csv >> "$output_dir/toolchain.txt"
"$rustc" --edition=2024 --test "$source_dir/probe.rs" -o "$output_dir/tests"
"$output_dir/tests" | tee "$output_dir/green.log"
/usr/local/cuda/bin/nvcc -O3 -arch=sm_121 -std=c++17 -c "$source_dir/shim.cu" -o "$output_dir/shim.o"
ar rcs "$output_dir/libshim.a" "$output_dir/shim.o"
"$rustc" --edition=2024 -O "$source_dir/probe.rs" -o "$output_dir/probe" \
  -L "native=$output_dir" -l static=shim -L native=/usr/local/cuda/lib64 -l cudart -l stdc++ \
  -C link-arg=-Wl,-rpath,/usr/local/cuda/lib64
sha256sum "$source_dir/probe.rs" "$source_dir/shim.cu" "$source_dir/../../native/exl3.cu" "$output_dir/probe" > "$output_dir/hashes.txt"
"$output_dir/probe" "$output_dir" "${@:2}" | tee "$output_dir/results.csv"

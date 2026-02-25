#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -eq 0 ]]; then
  echo "Run as normal user (script uses sudo for apt)." >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y \
  build-essential \
  ninja-build \
  cmake \
  pkg-config \
  swig \
  python3-dev \
  python3-venv \
  python3-pip \
  libopenblas-dev \
  liblapack-dev \
  linux-tools-common \
  linux-tools-generic \
  linux-tools-$(uname -r) || true

python3 -m venv .venv-faiss-simd
source .venv-faiss-simd/bin/activate

python -m pip install --upgrade pip wheel setuptools
python -m pip install numpy

echo "Setup complete. Activate with:"
echo "  source .venv-faiss-simd/bin/activate"

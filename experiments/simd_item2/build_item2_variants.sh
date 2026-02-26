#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Activate virtualenv first (expected .venv-faiss-simd)." >&2
  exit 1
fi

JOBS="${JOBS:-$(nproc)}"

detect_no_vec_flags() {
  local cxx_bin="${CXX:-c++}"
  local ver
  ver="$(${cxx_bin} --version 2>/dev/null || true)"
  if echo "${ver}" | grep -qi clang; then
    echo "-fno-vectorize -fno-slp-vectorize"
  else
    # GCC and compatible
    echo "-fno-tree-vectorize"
  fi
}

build_one() {
  local build_dir="$1"
  local extra_flags="$2"
  local -a cmake_args
  cmake_args=(
    -S .
    -B "${build_dir}"
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DFAISS_ENABLE_GPU=OFF
    -DFAISS_ENABLE_PYTHON=ON
    -DFAISS_ENABLE_EXTRAS=OFF
    -DBUILD_TESTING=OFF
    -DFAISS_OPT_LEVEL=dd
  )

  # Keep toolchain default Release flags and append only extra flags for
  # this variant.
  if [[ -n "${extra_flags}" ]]; then
    cmake_args+=("-DCMAKE_CXX_FLAGS_RELEASE:STRING=${extra_flags}")
  fi

  echo "==> Configuring ${build_dir}"
  cmake "${cmake_args[@]}"

  echo "==> Building faiss + swigfaiss (${build_dir})"
  cmake --build "${build_dir}" -j "${JOBS}" --target faiss swigfaiss

  echo "==> Building local Python module (${build_dir})"
  (
    cd "${build_dir}/faiss/python"
    python setup.py build >/dev/null
  )

  local py_mod_dir
  py_mod_dir="$(ls -d "${build_dir}"/faiss/python/build/lib* | head -n1)"
  if [[ -z "${py_mod_dir}" ]]; then
    echo "Failed to locate built python module path in ${build_dir}" >&2
    exit 1
  fi
  echo "${ROOT_DIR}/${py_mod_dir}" > "${build_dir}/PYTHONPATH.txt"
}

NO_VEC_FLAGS="$(detect_no_vec_flags)"

# autovec ON: default release flags
build_one "build_dd_autovec_on" ""

# autovec OFF: disable compiler auto-vectorization
build_one "build_dd_autovec_off" "${NO_VEC_FLAGS}"

echo "Builds complete."
echo " - build_dd_autovec_on"
echo " - build_dd_autovec_off"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Activate virtualenv first (expected .venv-faiss-simd)." >&2
  exit 1
fi

BASE_RESULT_DIR="experiments/simd_item2/sweep_results"
mkdir -p "${BASE_RESULT_DIR}"

COMMON_BENCH_ARGS="${COMMON_BENCH_ARGS:---dataset-dir dataset/sift1m --k 10 --nprobe 16 --omp-threads 1 --train-from-base 100000}"
OMP_THREADS="${OMP_THREADS:-1}"
INDEX_METHOD="${INDEX_METHOD:-ivfpq}"

# d,m pairs. Must satisfy d % m == 0.
CONFIGS=(
  "128,16"
  "112,16"
  "96,16"
  "64,16"
  "32,16"
  "128,32"
)

for cfg in "${CONFIGS[@]}"; do
  IFS=',' read -r d m <<< "${cfg}"
  if (( d % m != 0 )); then
    echo "Skipping invalid config d=${d}, m=${m} (d % m != 0)." >&2
    continue
  fi
  dsub=$((d / m))
  case_dir="${BASE_RESULT_DIR}/d${d}_m${m}_dsub${dsub}_idx${INDEX_METHOD}"
  mkdir -p "${case_dir}"

  echo "==> Sweep case d=${d} m=${m} dsub=${dsub}"
  export RESULT_DIR="${case_dir}"
  export INDEX_METHOD="${INDEX_METHOD}"
  export OMP_NUM_THREADS="${OMP_THREADS}"
  export BENCH_ARGS="${COMMON_BENCH_ARGS} --d ${d} --m ${m}"
  bash experiments/simd_item2/run_item2.sh
done

python - <<'PY'
import csv
import json
import os
from pathlib import Path

base = Path("experiments/simd_item2/sweep_results")
index_method = os.environ.get("INDEX_METHOD", "ivfpq")
rows = []
for case_dir in sorted(base.glob(f"d*_m*_dsub*_idx{index_method}")):
    summary_json = case_dir / f"summary_{index_method}.json"
    if not summary_json.exists():
        continue
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    result_map = payload.get("results", {})
    derived = payload.get("derived", {})

    autovec = result_map.get("autovec_only", {})
    avx512 = result_map.get("intrinsics_avx512", {})

    # Parse d, m, dsub from folder name d{d}_m{m}_dsub{dsub}_idx{index}
    parts = case_dir.name.replace("d", "").split("_m")
    d = int(parts[0])
    m_str, dsub_part = parts[1].split("_dsub")
    dsub_str, _idx = dsub_part.split("_idx")
    m = int(m_str)
    dsub = int(dsub_str)

    row = {
        "case": case_dir.name,
        "d": d,
        "m": m,
        "dsub": dsub,
        "autovec_qps": autovec.get("qps"),
        "avx512_qps": avx512.get("qps"),
        "autovec_p50_ms": autovec.get("latency_p50_ms"),
        "avx512_p50_ms": avx512.get("latency_p50_ms"),
        "autovec_recall_at_k": autovec.get("recall_at_k"),
        "avx512_recall_at_k": avx512.get("recall_at_k"),
        "qps_speedup_intrinsics_avx512_over_autovec": derived.get(
            "qps_speedup_intrinsics_avx512_over_autovec"
        ),
        "p50_latency_ratio_intrinsics_avx512_over_autovec": derived.get(
            "p50_latency_ratio_intrinsics_avx512_over_autovec"
        ),
        "cycles_ratio_intrinsics_avx512_over_autovec": derived.get(
            "cycles_ratio_intrinsics_avx512_over_autovec"
        ),
    }
    rows.append(row)

if rows:
    out_json = base / f"sweep_summary_{index_method}.json"
    out_csv = base / f"sweep_summary_{index_method}.csv"
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    keys = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")
else:
    print("No case results found.")
PY

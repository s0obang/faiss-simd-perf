#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Activate virtualenv first (expected .venv-faiss-simd)." >&2
  exit 1
fi

BASE_RESULT_DIR="experiments/simd_item2/sweep_results_dsub"
mkdir -p "${BASE_RESULT_DIR}"

COMMON_BENCH_ARGS="${COMMON_BENCH_ARGS:---k 10 --nprobe 16 --omp-threads 1 --train-from-base 100000}"
OMP_THREADS="${OMP_THREADS:-1}"
INDEX_METHOD="${INDEX_METHOD:-ivfpq}"

# Dsub sweep mode:
# - fixed_m: keep m constant, d = m * dsub (recommended for varying dsub without fixed d)
# - fixed_d: keep d constant, m = d / dsub (requires d % dsub == 0)
SWEEP_MODE="${SWEEP_MODE:-fixed_m}"
FIXED_D="${FIXED_D:-128}"
FIXED_M="${FIXED_M:-16}"

DSUB_VALUES="${DSUB_VALUES:-2 4 8 16 32}"
if [[ "${DSUB_VALUES}" == *","* ]]; then
  DSUB_VALUES="${DSUB_VALUES//,/ }"
fi
read -r -a DSUB_LIST <<< "${DSUB_VALUES}"

if [[ ${#DSUB_LIST[@]} -eq 0 ]]; then
  echo "No DSUB values provided. Set DSUB_VALUES=\"2 4 8 16 32\"." >&2
  exit 1
fi

for dsub in "${DSUB_LIST[@]}"; do
  if (( dsub <= 0 )); then
    echo "Skip invalid dsub=${dsub}."
    continue
  fi

  case "${SWEEP_MODE}" in
    fixed_d)
      d="${FIXED_D}"
      if (( d % dsub != 0 )); then
        echo "Skip d=${d} with dsub=${dsub} because d % dsub != 0." >&2
        continue
      fi
      m=$((d / dsub))
      ;;
    fixed_m|*)
      m="${FIXED_M}"
      d=$((m * dsub))
      ;;
  esac

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

base = Path("experiments/simd_item2/sweep_results_dsub")
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

    parts = case_dir.name.replace("d", "").split("_m")
    d = int(parts[0])
    m_str, dsub_part = parts[1].split("_dsub")
    dsub_str, _idx = dsub_part.split("_idx")
    m = int(m_str)
    dsub = int(dsub_str)

    def ratio(a, b):
        return a / b if b else None

    row = {
        "case": case_dir.name,
        "d": d,
        "m": m,
        "dsub": dsub,
        "autovec_qps": autovec.get("qps"),
        "avx512_qps": avx512.get("qps"),
        "autovec_p50_ms": autovec.get("latency_p50_ms"),
        "avx512_p50_ms": avx512.get("latency_p50_ms"),
        "autovec_p95_ms": autovec.get("latency_p95_ms"),
        "avx512_p95_ms": avx512.get("latency_p95_ms"),
        "autovec_recall_at_k": autovec.get("recall_at_k"),
        "avx512_recall_at_k": avx512.get("recall_at_k"),
        "autovec_query_memory_avg_mb": autovec.get("query_memory_avg_mb"),
        "avx512_query_memory_avg_mb": avx512.get("query_memory_avg_mb"),
        "autovec_query_memory_max_mb": autovec.get("query_memory_max_mb"),
        "avx512_query_memory_max_mb": avx512.get("query_memory_max_mb"),
        "autovec_query_memory_samples": autovec.get("query_memory_samples"),
        "avx512_query_memory_samples": avx512.get("query_memory_samples"),
        "autovec_query_search_memory_avg_mb": autovec.get("query_search_memory_avg_mb"),
        "avx512_query_search_memory_avg_mb": avx512.get("query_search_memory_avg_mb"),
        "autovec_query_search_memory_max_mb": autovec.get("query_search_memory_max_mb"),
        "avx512_query_search_memory_max_mb": avx512.get("query_search_memory_max_mb"),
        "autovec_query_search_memory_samples": autovec.get("query_search_memory_samples"),
        "avx512_query_search_memory_samples": avx512.get("query_search_memory_samples"),
        "autovec_query_latency_memory_avg_mb": autovec.get("query_latency_memory_avg_mb"),
        "avx512_query_latency_memory_avg_mb": avx512.get("query_latency_memory_avg_mb"),
        "autovec_query_latency_memory_max_mb": autovec.get("query_latency_memory_max_mb"),
        "avx512_query_latency_memory_max_mb": avx512.get("query_latency_memory_max_mb"),
        "autovec_query_latency_memory_samples": autovec.get("query_latency_memory_samples"),
        "avx512_query_latency_memory_samples": avx512.get("query_latency_memory_samples"),
        "qps_speedup_intrinsics_avx512_over_autovec": ratio(
            avx512.get("qps"), autovec.get("qps")
        ),
        "p50_latency_ratio_intrinsics_avx512_over_autovec": ratio(
            avx512.get("latency_p50_ms"), autovec.get("latency_p50_ms")
        ),
        "cycles_ratio_intrinsics_avx512_over_autovec": derived.get(
            "cycles_ratio_intrinsics_avx512_over_autovec"
        ),
        "memory_ratio_query_avg_mb": ratio(
            avx512.get("query_memory_avg_mb"), autovec.get("query_memory_avg_mb")
        ),
        "memory_ratio_query_max_mb": ratio(
            avx512.get("query_memory_max_mb"), autovec.get("query_memory_max_mb")
        ),
        "memory_ratio_query_search_avg_mb": ratio(
            avx512.get("query_search_memory_avg_mb"),
            autovec.get("query_search_memory_avg_mb"),
        ),
        "memory_ratio_query_search_max_mb": ratio(
            avx512.get("query_search_memory_max_mb"),
            autovec.get("query_search_memory_max_mb"),
        ),
        "memory_ratio_query_latency_avg_mb": ratio(
            avx512.get("query_latency_memory_avg_mb"),
            autovec.get("query_latency_memory_avg_mb"),
        ),
        "memory_ratio_query_latency_max_mb": ratio(
            avx512.get("query_latency_memory_max_mb"),
            autovec.get("query_latency_memory_max_mb"),
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

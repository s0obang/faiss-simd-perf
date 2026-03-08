#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Activate virtualenv first (expected .venv-faiss-simd)." >&2
  exit 1
fi

BASE_RESULT_DIR="${BASE_RESULT_DIR:-experiments/simd_item2_intrinsics_only/sweep_results_fastscan}"
mkdir -p "${BASE_RESULT_DIR}"

BUILD_DIR="${BUILD_DIR:-build_dd_autovec_on}"
PYTHONPATH_FILE="${BUILD_DIR}/PYTHONPATH.txt"

COMMON_BENCH_ARGS="${COMMON_BENCH_ARGS:---k 10 --nprobe 16 --omp-threads 1 --train-from-base 100000}"
OMP_THREADS="${OMP_THREADS:-1}"
SWEEP_MODE="${SWEEP_MODE:-fixed_d}"
FIXED_D="${FIXED_D:-128}"
FIXED_M="${FIXED_M:-16}"
AGGREGATE_ONLY="${AGGREGATE_ONLY:-0}"

DSUB_VALUES="${DSUB_VALUES:-2 4 8 16 32}"
if [[ "${DSUB_VALUES}" == *","* ]]; then
  DSUB_VALUES="${DSUB_VALUES//,/ }"
fi
read -r -a DSUB_LIST <<< "${DSUB_VALUES}"

if [[ ${#DSUB_LIST[@]} -eq 0 ]]; then
  echo "No DSUB values provided. Set DSUB_VALUES=\"2 4 8 16 32\"." >&2
  exit 1
fi

run_case() {
  local d="$1"
  local m="$2"
  local dsub="$3"
  local case_dir="$4"

  mkdir -p "${case_dir}"
  local out_json="${case_dir}/fastscan_ivfpq_fastscan.json"
  local out_csv="${case_dir}/fastscan_ivfpq_fastscan.csv"

  echo "==> Running fastscan ivfpq_fastscan (d=${d}, m=${m}, dsub=${dsub})"

  local -a bench_args_arr=()
  if [[ -n "${COMMON_BENCH_ARGS}" ]]; then
    # shellcheck disable=SC2206
    bench_args_arr=(${COMMON_BENCH_ARGS})
  fi

  local -a bench_cmd=(
    python experiments/simd_item2/bench_item2_e2e.py
    --output "${out_json}"
    --output-csv "${out_csv}"
    --index-type ivfpq_fastscan
    --d "${d}"
    --m "${m}"
    --nbits "4"
    --omp-threads "${OMP_THREADS}"
  )
  bench_cmd+=("${bench_args_arr[@]}")

  local -a env_cmd=(
    PYTHONPATH="${PYTHON_PATH}"
    FAISS_SIMD_LEVEL=AVX512
    OMP_NUM_THREADS="${OMP_THREADS}"
  )

  ("${env_cmd[@]}" ${bench_cmd[@]})
}

if [[ "${AGGREGATE_ONLY}" == "1" ]]; then
  echo "AGGREGATE_ONLY=1: skipping benchmark runs."
else
  if [[ ! -f "${PYTHONPATH_FILE}" ]]; then
    echo "Missing ${PYTHONPATH_FILE}. Build first with build_item2_variants.sh." >&2
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
          echo "Skip d=${d} with dsub=${dsub} because d % dsub != 0."
          continue
        fi
        m=$((d / dsub))
        ;;
      fixed_m|*)
        m="${FIXED_M}"
        d=$((m * dsub))
        ;;
    esac

    case_dir="${BASE_RESULT_DIR}/d${d}_m${m}_dsub${dsub}_idxivfpq_fastscan"
    run_case "${d}" "${m}" "${dsub}" "${case_dir}"
  done
fi

python - <<'PY'
import csv
import json
from pathlib import Path
import os
import re

base = Path(os.environ.get("BASE_RESULT_DIR", "experiments/simd_item2_intrinsics_only/sweep_results_fastscan"))
index_method = "ivfpq_fastscan"
rows = []

for case_dir in sorted(base.glob("d*_m*_dsub*_idxivfpq_fastscan")):
    metric_json = case_dir / f"fastscan_{index_method}.json"
    if not metric_json.exists():
        continue

    payload = json.loads(metric_json.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    config = payload.get("config", {})
    m = re.match(r"d(\d+)_m(\d+)_dsub(\d+)_idx", case_dir.name)
    if not m:
        print(f"Skip unknown case directory: {case_dir}")
        continue

    d, m_sub, dsub = map(int, m.groups())

    row = {
        "case": case_dir.name,
        "d": d,
        "m": m_sub,
        "dsub": dsub,
        "index_type": config.get("index_type", index_method),
        "qps": metrics.get("qps"),
        "search_time_s": metrics.get("search_time_s"),
        "latency_p50_ms": metrics.get("latency_p50_ms"),
        "latency_p95_ms": metrics.get("latency_p95_ms"),
        "recall_at_k": metrics.get("recall_at_k"),
        "query_memory_avg_mb": metrics.get("query_memory_avg_mb"),
        "query_memory_max_mb": metrics.get("query_memory_max_mb"),
        "query_memory_samples": metrics.get("query_memory_samples"),
        "query_search_memory_avg_mb": metrics.get("query_search_memory_avg_mb"),
        "query_search_memory_max_mb": metrics.get("query_search_memory_max_mb"),
        "query_search_memory_samples": metrics.get("query_search_memory_samples"),
        "query_latency_memory_avg_mb": metrics.get("query_latency_memory_avg_mb"),
        "query_latency_memory_max_mb": metrics.get("query_latency_memory_max_mb"),
        "query_latency_memory_samples": metrics.get("query_latency_memory_samples"),
    }
    rows.append(row)

if rows:
    out_json = base / f"sweep_summary_{index_method}.json"
    out_csv = base / f"sweep_summary_{index_method}.csv"
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        keys = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")
else:
    print("No case results found.")
PY

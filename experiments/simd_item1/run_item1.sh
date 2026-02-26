#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Activate virtualenv first (expected .venv-faiss-simd)." >&2
  exit 1
fi

RESULT_DIR="experiments/simd_item1/results"
mkdir -p "${RESULT_DIR}"

OMP_THREADS="${OMP_THREADS:-1}"
BENCH_ARGS="${BENCH_ARGS:-}"
INDEX_METHOD="${INDEX_METHOD:-ivfpq}"

run_case() {
  local case_name="$1"
  local build_dir="$2"

  local py_path_file="${build_dir}/PYTHONPATH.txt"
  if [[ ! -f "${py_path_file}" ]]; then
    echo "Missing ${py_path_file}. Run build_item1_variants.sh first." >&2
    exit 1
  fi

  local py_mod_path
  py_mod_path="$(cat "${py_path_file}")"
  local out_json="${RESULT_DIR}/${case_name}_${INDEX_METHOD}.json"
  local out_csv="${RESULT_DIR}/${case_name}_${INDEX_METHOD}.csv"
  local out_perf="${RESULT_DIR}/${case_name}_${INDEX_METHOD}.perf.csv"

  echo "==> Running ${case_name}"
  export PYTHONPATH="${py_mod_path}"
  export FAISS_SIMD_LEVEL=NONE
  export OMP_NUM_THREADS="${OMP_THREADS}"
  local -a bench_args_arr=()
  if [[ -n "${BENCH_ARGS}" ]]; then
    # shellcheck disable=SC2206
    bench_args_arr=(${BENCH_ARGS})
  fi

  local -a bench_cmd=(
    python experiments/simd_item1/bench_item1_e2e.py
    --output "${out_json}"
    --output-csv "${out_csv}"
    --index-type "${INDEX_METHOD}"
  )
  bench_cmd+=("${bench_args_arr[@]}")

  if command -v perf >/dev/null 2>&1; then
    if ! perf stat -x, -e cycles,instructions,task-clock \
      -o "${out_perf}" -- \
      "${bench_cmd[@]}"; then
      echo "perf stat failed for ${case_name}; rerunning without perf." >&2
      "${bench_cmd[@]}"
    fi
  else
    "${bench_cmd[@]}"
  fi
}

run_case "no_simd" "build_dd_autovec_off"
run_case "autovec_only" "build_dd_autovec_on"

python - <<'PY'
import csv
import json
import os
from pathlib import Path

result_dir = Path("experiments/simd_item1/results")

def load_metrics(case_name):
    j = json.loads((result_dir / f"{case_name}.json").read_text(encoding="utf-8"))
    out = dict(j["metrics"])
    perf_path = result_dir / f"{case_name}.perf.csv"
    if perf_path.exists():
        with perf_path.open("r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 3:
                    continue
                val, _, event = row[0].strip(), row[1].strip(), row[2].strip()
                if event in ("cycles", "instructions", "task-clock"):
                    try:
                        out[event.replace("-", "_")] = float(val)
                    except ValueError:
                        pass
    return out

summary = {
    "no_simd": load_metrics(f"no_simd_{os.environ['INDEX_METHOD']}"),
    "autovec_only": load_metrics(f"autovec_only_{os.environ['INDEX_METHOD']}"),
}

def ratio(a, b):
    return a / b if b else None

derived = {}
if "qps" in summary["no_simd"] and "qps" in summary["autovec_only"]:
    derived["qps_speedup_autovec_over_no_simd"] = ratio(
        summary["autovec_only"]["qps"], summary["no_simd"]["qps"]
    )
if "latency_p50_ms" in summary["no_simd"] and "latency_p50_ms" in summary["autovec_only"]:
    derived["p50_latency_ratio_autovec_over_no_simd"] = ratio(
        summary["autovec_only"]["latency_p50_ms"], summary["no_simd"]["latency_p50_ms"]
    )
if "cycles" in summary["no_simd"] and "cycles" in summary["autovec_only"]:
    derived["cycles_ratio_autovec_over_no_simd"] = ratio(
        summary["autovec_only"]["cycles"], summary["no_simd"]["cycles"]
    )

out = {"results": summary, "derived": derived, "index_method": os.environ["INDEX_METHOD"]}
(result_dir / f"summary_{os.environ['INDEX_METHOD']}.json").write_text(
    json.dumps(out, indent=2), encoding="utf-8"
)

# Also save a flat CSV summary for easy spreadsheet import
rows = []
for case_name, metrics in summary.items():
    row = {"case": case_name}
    row.update(metrics)
    rows.append(row)

if rows:
    keys = sorted({k for row in rows for k in row.keys()})
    with (result_dir / f"summary_{os.environ['INDEX_METHOD']}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

print(json.dumps(out, indent=2))
PY

echo "Done. See ${RESULT_DIR}/summary_${INDEX_METHOD}.json"

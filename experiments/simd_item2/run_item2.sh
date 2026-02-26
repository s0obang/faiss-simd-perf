#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Activate virtualenv first (expected .venv-faiss-simd)." >&2
  exit 1
fi

RESULT_DIR="${RESULT_DIR:-experiments/simd_item2/results}"
mkdir -p "${RESULT_DIR}"

OMP_THREADS="${OMP_THREADS:-1}"
BENCH_ARGS="${BENCH_ARGS:-}"

run_case() {
  local case_name="$1"
  local build_dir="$2"
  local simd_level="$3"

  local py_path_file="${build_dir}/PYTHONPATH.txt"
  if [[ ! -f "${py_path_file}" ]]; then
    echo "Missing ${py_path_file}. Run build_item2_variants.sh first." >&2
    exit 1
  fi

  local py_mod_path
  py_mod_path="$(cat "${py_path_file}")"
  local out_json="${RESULT_DIR}/${case_name}.json"
  local out_csv="${RESULT_DIR}/${case_name}.csv"
  local out_perf="${RESULT_DIR}/${case_name}.perf.csv"

  echo "==> Running ${case_name}"
  export PYTHONPATH="${py_mod_path}"
  export FAISS_SIMD_LEVEL="${simd_level}"
  export OMP_NUM_THREADS="${OMP_THREADS}"
  local -a bench_args_arr=()
  if [[ -n "${BENCH_ARGS}" ]]; then
    # shellcheck disable=SC2206
    bench_args_arr=(${BENCH_ARGS})
  fi

  local -a bench_cmd=(
    python experiments/simd_item2/bench_item2_e2e.py
    --output "${out_json}"
    --output-csv "${out_csv}"
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

run_case "autovec_only" "build_dd_autovec_on" "NONE"
run_case "intrinsics_avx512" "build_dd_autovec_on" "AVX512"

python - <<'PY'
import csv
import json
from pathlib import Path

result_dir = Path("experiments/simd_item2/results")

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
    "autovec_only": load_metrics("autovec_only"),
    "intrinsics_avx512": load_metrics("intrinsics_avx512"),
}

def ratio(a, b):
    return a / b if b else None

derived = {}
if "qps" in summary["autovec_only"] and "qps" in summary["intrinsics_avx512"]:
    derived["qps_speedup_intrinsics_avx512_over_autovec"] = ratio(
        summary["intrinsics_avx512"]["qps"], summary["autovec_only"]["qps"]
    )
if "latency_p50_ms" in summary["autovec_only"] and "latency_p50_ms" in summary["intrinsics_avx512"]:
    derived["p50_latency_ratio_intrinsics_avx512_over_autovec"] = ratio(
        summary["intrinsics_avx512"]["latency_p50_ms"], summary["autovec_only"]["latency_p50_ms"]
    )
if "cycles" in summary["autovec_only"] and "cycles" in summary["intrinsics_avx512"]:
    derived["cycles_ratio_intrinsics_avx512_over_autovec"] = ratio(
        summary["intrinsics_avx512"]["cycles"], summary["autovec_only"]["cycles"]
    )

out = {"results": summary, "derived": derived}
(result_dir / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

# Also save a flat CSV summary for easy spreadsheet import
rows = []
for case_name, metrics in summary.items():
    row = {"case": case_name}
    row.update(metrics)
    rows.append(row)

if rows:
    keys = sorted({k for row in rows for k in row.keys()})
    with (result_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

print(json.dumps(out, indent=2))
PY

echo "Done. See ${RESULT_DIR}/summary.json"

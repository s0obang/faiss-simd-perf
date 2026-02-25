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
  local out_json="${RESULT_DIR}/${case_name}.json"
  local out_perf="${RESULT_DIR}/${case_name}.perf.csv"

  echo "==> Running ${case_name}"
  export PYTHONPATH="${py_mod_path}"
  export FAISS_SIMD_LEVEL=NONE
  export OMP_NUM_THREADS="${OMP_THREADS}"

  if command -v perf >/dev/null 2>&1; then
    perf stat -x, -e cycles,instructions,task-clock \
      -o "${out_perf}" -- \
      python experiments/simd_item1/bench_item1_e2e.py \
      --output "${out_json}" ${BENCH_ARGS}
  else
    python experiments/simd_item1/bench_item1_e2e.py \
      --output "${out_json}" ${BENCH_ARGS}
  fi
}

run_case "no_simd" "build_dd_autovec_off"
run_case "autovec_only" "build_dd_autovec_on"

python - <<'PY'
import csv
import json
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
    "no_simd": load_metrics("no_simd"),
    "autovec_only": load_metrics("autovec_only"),
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

out = {"results": summary, "derived": derived}
(result_dir / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
PY

echo "Done. See ${RESULT_DIR}/summary.json"

# Item2 Experiment: Intrinsics (AVX512) Effect

This folder automates experiment item 2:

- (b) `Autovec-only`: `FAISS_SIMD_LEVEL=NONE` + compiler autovec on
- (c) `Intrinsics-AVX512`: `FAISS_SIMD_LEVEL=AVX512` + compiler autovec on

Metrics collected:

- QPS
- search latency (`p50_ms`, `p95_ms`)
- recall@k
- CPU cycles (via `perf stat`, if available)

## 0) Assumptions

- Linux x86_64 server (AVX-512 capable is fine)
- You run commands from repository root
- You want CPU-only builds

## 1) Environment setup

```bash
bash experiments/simd_item2/setup_ubuntu.sh
source .venv-faiss-simd/bin/activate
```

## 2) Build two DD variants

```bash
bash experiments/simd_item2/build_item2_variants.sh
```

This produces:

- `build_dd_autovec_on` (vectorize on)
- `build_dd_autovec_off` (built too, but not used in default item2 run)

Each build stores its Python module path in:

- `<build_dir>/PYTHONPATH.txt`

## 3) Run benchmark

```bash
bash experiments/simd_item2/run_item2.sh
```

With SIFT1M files in `dataset/sift1m` (expected file names):

- `sift1m_base.fvecs`
- `sift1m_query.fvecs`
- `sift1m_groundtruth.ivecs`

run:

```bash
BENCH_ARGS="--dataset-dir dataset/sift1m --k 10 --nprobe 16 --omp-threads 1" \
bash experiments/simd_item2/run_item2.sh
```

Choose index family at runtime:

```bash
INDEX_METHOD=ivfpq   # or ivfflat
bash experiments/simd_item2/run_item2.sh
```

Results are written to:

- `experiments/simd_item2/results/autovec_only_<index>.json`
- `experiments/simd_item2/results/intrinsics_avx512_<index>.json`
- `experiments/simd_item2/results/summary_<index>.json`

If `perf` is available and permitted:

- `experiments/simd_item2/results/autovec_only_<index>.perf.csv`
- `experiments/simd_item2/results/intrinsics_avx512_<index>.perf.csv`

## 4) Run Dsub Sweep

To sweep multiple `(d, M)` pairs:

```bash
bash experiments/simd_item2/run_item2_sweep.sh
```

Sweep outputs:

- per-case folders under `experiments/simd_item2/sweep_results/`
- merged summary:
- `experiments/simd_item2/sweep_results/sweep_summary.json`
- `experiments/simd_item2/sweep_results/sweep_summary.csv`

### Dsub sweep (recommended for this question)

To sweep `dsub` with recall + latency + query-memory (avg/max RSS) metrics:

```bash
SWEEP_MODE=fixed_m \
DSUB_VALUES="2 4 8 16 32" \
COMMON_BENCH_ARGS="--k 10 --nprobe 16 --omp-threads 1 --train-from-base 100000" \
OMP_THREADS=1 \
INDEX_METHOD=ivfpq \
FIXED_M=16 \
bash experiments/simd_item2/run_item2_dsub_sweep.sh
```

Output:

- case results under `experiments/simd_item2/sweep_results_dsub/`
- merged:
  - `experiments/simd_item2/sweep_results_dsub/sweep_summary_ivfpq.json`
  - `experiments/simd_item2/sweep_results_dsub/sweep_summary_ivfpq.csv`

The merged summary includes per-case:

- `autovec/avx512` latency (`p50_ms`, `p95_ms`)
- `autovec/avx512` recall (`recall_at_k`)
- `autovec/avx512` throughput (`qps`)
- query RSS metrics in MiB:
  - `query_memory_avg_mb`, `query_memory_max_mb`: combined query workload (throughput + latency)
  - `query_search_memory_*`: throughput batch-search phase
  - `query_latency_memory_*`: single-query latency phase
- speedup/ratio columns for qps, p50 latency, cycles, and memory.

Notes:

- `SWEEP_MODE=fixed_m` varies `d = m * dsub` (recommended when you want to test `dsub` from small to large).
- `SWEEP_MODE=fixed_d` keeps `d` constant and skips non-divisible `dsub` values.
- For fixed-d datasets (e.g. SIFT1M), set `SWEEP_MODE=fixed_d` and keep `d` matching the dataset dimension.

## 5) Notes

- Only SIMD level changes between cases (`NONE` vs `AVX512`), using the same build and parameters.
- This isolates explicit intrinsics path gain over autovec-only path.
- Keep thread count fixed for fair comparison (`OMP_NUM_THREADS=1` by default in runner).
- Default index is `IVF4096,PQ16x8`, so recall is meaningful vs exact ground truth.

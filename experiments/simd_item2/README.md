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

## 5) Notes

- Only SIMD level changes between cases (`NONE` vs `AVX512`), using the same build and parameters.
- This isolates explicit intrinsics path gain over autovec-only path.
- Keep thread count fixed for fair comparison (`OMP_NUM_THREADS=1` by default in runner).
- Default index is `IVF4096,PQ16x8`, so recall is meaningful vs exact ground truth.

# Item1 Experiment: Autovec Effect (Implicit SIMD)

This folder automates experiment item 1:

- (a) `No-SIMD`: `FAISS_SIMD_LEVEL=NONE` + compiler autovec off
- (b) `Autovec-only`: `FAISS_SIMD_LEVEL=NONE` + compiler autovec on

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
bash experiments/simd_item1/setup_ubuntu.sh
source .venv-faiss-simd/bin/activate
```

## 2) Build two DD variants

```bash
bash experiments/simd_item1/build_item1_variants.sh
```

This produces:

- `build_dd_autovec_on` (vectorize on)
- `build_dd_autovec_off` (vectorize off)

Each build stores its Python module path in:

- `<build_dir>/PYTHONPATH.txt`

## 3) Run benchmark

```bash
bash experiments/simd_item1/run_item1.sh
```

With SIFT1M files in `dataset/sift1m` (expected file names):

- `sift1m_base.fvecs`
- `sift1m_query.fvecs`
- `sift1m_groundtruth.ivecs`

run:

```bash
BENCH_ARGS="--dataset-dir dataset/sift1m --k 10 --nprobe 16 --omp-threads 1" \
bash experiments/simd_item1/run_item1.sh
```

Results are written to:

- `experiments/simd_item1/results/no_simd_<index>.json`
- `experiments/simd_item1/results/autovec_only_<index>.json`
- `experiments/simd_item1/results/summary_<index>.json`

If `perf` is available and permitted:

- `experiments/simd_item1/results/no_simd_<index>.perf.csv`
- `experiments/simd_item1/results/autovec_only_<index>.perf.csv`

Choose index family at runtime:

```bash
INDEX_METHOD=ivfpq   # or ivfflat
bash experiments/simd_item1/run_item1.sh
```

## 4) Notes

- Both cases force runtime dispatch to scalar path with `FAISS_SIMD_LEVEL=NONE`.
- The only intended difference is compiler auto-vectorization on/off.
- Keep thread count fixed for fair comparison (`OMP_NUM_THREADS=1` by default in runner).
- Default index is `IVF4096,PQ16x8`, so recall is meaningful vs exact ground truth.

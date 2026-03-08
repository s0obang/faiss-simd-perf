# Intrinsics-only Experiments (item2)

이 폴더는 `autovec_only` 케이스를 제외하고 **AVX512 인트린식 실행만** 수행하는
별도 스크립트 모음입니다.

- `run_avx512_dsub_sweep.sh`: 기존 IVF-PQ(`ivfpq`)를 AVX512-only로 dsub 스윕
- `run_fastscan_dsub_sweep.sh`: FastScan 계열(`ivfpq_fastscan`)을 AVX512-only로 dsub 스윕

## 준비
- 가상환경 활성화
- `build_item2_variants.sh`로 `build_dd_autovec_on` 생성

## 공통 환경 변수
- `BUILD_DIR` (default: `build_dd_autovec_on`)
- `COMMON_BENCH_ARGS`
  - 기본값: `"--k 10 --nprobe 16 --omp-threads 1 --train-from-base 100000"`
- `AGGREGATE_ONLY=1` 설정 시 케이스 실행 없이 집계만 수행
- `DSUB_VALUES` (기본 `"2 4 8 16 32"`)
- `SWEEP_MODE`
  - `fixed_d`: `d` 고정 (`FIXED_D`, 기본 128), `m = d/dsub`
  - `fixed_m`: `m` 고정 (`FIXED_M`, 기본 16), `d = m*dsub`
- `OMP_THREADS`
- `BASE_RESULT_DIR`

## FastScan 스윕 유의사항
- 현재 `ivfpq_fastscan`은 `nbits=4`만 지원됩니다.
- 스크립트 내부에서 `--nbits 4`로 고정합니다.

## 고차원 데이터셋
- `--dataset-dir`은 `sift1m_*` 또는 `sift_*` 파일명 규칙(`base/query/groundtruth/learn`)을 사용합니다.
- `FIXED_D`가 고정 모드일 때는 `FIXED_D % dsub == 0`인 경우만 실행됩니다.

## 출력
- 케이스별 결과: `.../d{d}_m{m}_dsub{dsub}_.../`
  - IVF-PQ: `intrinsics_avx512_ivfpq.json|csv`
  - FastScan: `fastscan_ivfpq_fastscan.json|csv`
- 스윕 요약: `sweep_summary_ivfpq.json|csv`, `sweep_summary_ivfpq_fastscan.json|csv`

## 실행 예시

### AVX512-only IVFPQ
```bash
SWEEP_MODE=fixed_d \
DSUB_VALUES="2 4 8 16 32" \
FIXED_D=128 \
COMMON_BENCH_ARGS="--dataset-dir ../dataset/sift1m --k 10 --nprobe 16 --omp-threads 1 --train-from-base 200000" \
OMP_THREADS=1 \
INDEX_METHOD=ivfpq \
bash experiments/simd_item2_intrinsics_only/run_avx512_dsub_sweep.sh
```

### FastScan
```bash
SWEEP_MODE=fixed_d \
DSUB_VALUES="2 4 8 16" \
FIXED_D=128 \
COMMON_BENCH_ARGS="--dataset-dir ../dataset/sift1m --k 10 --nprobe 16 --omp-threads 1 --train-from-base 200000" \
OMP_THREADS=1 \
bash experiments/simd_item2_intrinsics_only/run_fastscan_dsub_sweep.sh
```

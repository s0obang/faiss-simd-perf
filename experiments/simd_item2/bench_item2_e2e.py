#!/usr/bin/env python3
import argparse
import csv
import json
import os
import time

import numpy as np

import faiss


def fvecs_read(path):
    a = np.fromfile(path, dtype=np.int32)
    if a.size == 0:
        raise RuntimeError(f"empty fvecs file: {path}")
    d = int(a[0])
    return a.reshape(-1, d + 1)[:, 1:].view(np.float32).copy()


def ivecs_read(path):
    a = np.fromfile(path, dtype=np.int32)
    if a.size == 0:
        raise RuntimeError(f"empty ivecs file: {path}")
    d = int(a[0])
    return a.reshape(-1, d + 1)[:, 1:].copy()


def parse_args():
    p = argparse.ArgumentParser(description="E2E benchmark for item2 intrinsics/AVX512 study")
    p.add_argument("--output", required=True)
    p.add_argument("--output-csv", default="")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--nb", type=int, default=200000)
    p.add_argument("--nt", type=int, default=50000, help="train vectors")
    p.add_argument("--nq", type=int, default=5000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--nlist", type=int, default=4096)
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--nbits", type=int, default=8)
    p.add_argument("--nprobe", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--latency-queries", type=int, default=1000)
    p.add_argument("--omp-threads", type=int, default=1)
    p.add_argument(
        "--dataset-dir",
        default="",
        help=(
            "Optional SIFT1M folder containing "
            "sift1m_base.fvecs, sift1m_query.fvecs, sift1m_groundtruth.ivecs"
        ),
    )
    p.add_argument(
        "--train-from-base",
        type=int,
        default=100000,
        help="If dataset-dir is used and no train file exists, take first N base vectors for training",
    )
    return p.parse_args()


def make_data(seed, nb, nt, nq, d):
    rng = np.random.default_rng(seed)
    xb = rng.random((nb, d), dtype=np.float32)
    xt = rng.random((nt, d), dtype=np.float32)
    xq = rng.random((nq, d), dtype=np.float32)
    return xb, xt, xq


def resolve_sift_paths(dataset_dir):
    # Support both naming conventions:
    # 1) sift1m_base/query/groundtruth/learn
    # 2) sift_base/query/groundtruth/learn
    patterns = [
        (
            "sift1m_base.fvecs",
            "sift1m_query.fvecs",
            "sift1m_groundtruth.ivecs",
            "sift1m_learn.fvecs",
        ),
        (
            "sift_base.fvecs",
            "sift_query.fvecs",
            "sift_groundtruth.ivecs",
            "sift_learn.fvecs",
        ),
    ]

    for base_name, query_name, gt_name, learn_name in patterns:
        base_path = os.path.join(dataset_dir, base_name)
        query_path = os.path.join(dataset_dir, query_name)
        gt_path = os.path.join(dataset_dir, gt_name)
        learn_path = os.path.join(dataset_dir, learn_name)
        if (
            os.path.exists(base_path)
            and os.path.exists(query_path)
            and os.path.exists(gt_path)
        ):
            return base_path, query_path, gt_path, learn_path

    raise RuntimeError(
        "Could not find supported SIFT files in dataset-dir. "
        "Expected either sift1m_* or sift_* naming."
    )


def load_sift1m_from_dir(dataset_dir, train_from_base):
    base_path, query_path, gt_path, train_path = resolve_sift_paths(dataset_dir)

    xb = fvecs_read(base_path)
    xq = fvecs_read(query_path)
    i_gt = ivecs_read(gt_path)

    if os.path.exists(train_path):
        xt = fvecs_read(train_path)
    else:
        ntrain = min(train_from_base, xb.shape[0])
        xt = xb[:ntrain].copy()

    return xb, xt, xq, i_gt


def recall_at_k(i_pred, i_gt):
    # average intersection ratio per query
    k = i_pred.shape[1]
    total = 0.0
    for a, b in zip(i_pred, i_gt):
        total += len(set(a.tolist()) & set(b.tolist())) / float(k)
    return total / i_pred.shape[0]


def timed_search(index, xq, k, batch_size):
    nq = xq.shape[0]
    i_all = np.empty((nq, k), dtype=np.int64)
    d_all = np.empty((nq, k), dtype=np.float32)

    t0 = time.perf_counter()
    for i0 in range(0, nq, batch_size):
        i1 = min(i0 + batch_size, nq)
        d, i = index.search(xq[i0:i1], k)
        d_all[i0:i1] = d
        i_all[i0:i1] = i
    t1 = time.perf_counter()
    return (t1 - t0), d_all, i_all


def per_query_latency(index, xq, k):
    lat_ms = []
    for i in range(xq.shape[0]):
        t0 = time.perf_counter_ns()
        index.search(xq[i : i + 1], k)
        t1 = time.perf_counter_ns()
        lat_ms.append((t1 - t0) / 1e6)
    return np.asarray(lat_ms, dtype=np.float64)


def main():
    args = parse_args()

    faiss.omp_set_num_threads(args.omp_threads)

    if args.dataset_dir:
        xb, xt, xq, i_gt = load_sift1m_from_dir(
            args.dataset_dir, args.train_from_base
        )
        d = int(xb.shape[1])
        if d != args.d:
            args.d = d
    else:
        xb, xt, xq = make_data(args.seed, args.nb, args.nt, args.nq, args.d)
        # Exact ground truth for recall
        gt = faiss.IndexFlatL2(args.d)
        gt.add(xb)
        _, i_gt = gt.search(xq, args.k)

    # ANN index under test
    quantizer = faiss.IndexFlatL2(args.d)
    index = faiss.IndexIVFPQ(quantizer, args.d, args.nlist, args.m, args.nbits)
    assert not index.is_trained
    index.train(xt)
    index.add(xb)
    index.nprobe = args.nprobe

    # Warmup
    _ = index.search(xq[: min(200, args.nq)], args.k)

    # Throughput / total search time
    total_s, _, i_pred = timed_search(index, xq, args.k, args.batch_size)
    qps = float(xq.shape[0]) / total_s

    # Single-query latency distribution on a subset
    lq = min(args.latency_queries, xq.shape[0])
    lats = per_query_latency(index, xq[:lq], args.k)

    rec = recall_at_k(i_pred, i_gt[:, : args.k])

    result = {
        "config": {
            "seed": args.seed,
            "d": args.d,
            "nb": int(xb.shape[0]),
            "nt": int(xt.shape[0]),
            "nq": int(xq.shape[0]),
            "k": args.k,
            "nlist": args.nlist,
            "m": args.m,
            "nbits": args.nbits,
            "nprobe": args.nprobe,
            "batch_size": args.batch_size,
            "latency_queries": lq,
            "omp_threads": args.omp_threads,
            "faiss_simd_level_env": os.environ.get("FAISS_SIMD_LEVEL", ""),
            "dataset_dir": args.dataset_dir,
        },
        "metrics": {
            "qps": qps,
            "search_time_s": total_s,
            "latency_p50_ms": float(np.percentile(lats, 50)),
            "latency_p95_ms": float(np.percentile(lats, 95)),
            "recall_at_k": rec,
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if args.output_csv:
        flat = {}
        for k, v in result["config"].items():
            flat[f"config.{k}"] = v
        for k, v in result["metrics"].items():
            flat[f"metrics.{k}"] = v
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(flat.keys()))
            w.writeheader()
            w.writerow(flat)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

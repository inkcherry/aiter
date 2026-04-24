# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""
Workload-equivalent MLA decode microbenchmark.

This UT mirrors what sglang's ``aiter_backend`` does at decode time for DSR1
(nhead=128, fp8/fp8, persistent mode, fast_mode=True / intra_batch_mode=False),
under a *cold* L2 (a 512 MB scratch buffer is overwritten between every timed
launch). The cold-L2 protocol matches reality: between two successive decode
steps the previously-touched KV pages have already been evicted by the rest
of the model (FFN, all-reduce, sampler, etc.), so reusing the same kv_buffer
in a hot loop overstates kernel performance dramatically.

Why it lives in op_tests/
-------------------------
Because the gain from routing nhead=128 fp8 to the qh128-native ASM kernel
(instead of the qh16-fold path) only shows up under exactly these conditions:

  * cold L2 between iterations,
  * persistent mode with sglang-style metadata,
  * production batch size (256 / card on 4P1D MINI).

If any of these are missing, the apparent ranking of the two kernels flips
and the regression looks artificial. This UT keeps that protocol in tree.

Usage::

    # informational sweep (default):
    python3 op_tests/test_mla_decode_workload_bench.py

    # also assert a TB/s floor at bs=256 (CI-style guard):
    AITER_MLA_BENCH_ASSERT=1 \
      python3 op_tests/test_mla_decode_workload_bench.py
"""

import os
import sys

import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.mla import mla_decode_fwd


# Production-relevant DSR1 decode shape on 4P1D MINI:
#   per-card concurrency = 256, ctx ≈ 8k, fp8/fp8.
_NHEAD = 128
_NHEAD_KV = 1
_QK_LORA = 512
_QK_ROPE = 64
_QK_HEAD_DIM = _QK_LORA + _QK_ROPE
_V_HEAD_DIM = _QK_LORA
_PAGE_SIZE = 1
_MAX_Q_LEN = 1
_CTX_LEN = 8192

# bs sweep keeps the small-batch sample (4) so the kernel cross-over with the
# qh16-fold path is visible, but the production point (256) is the one any
# regression check should hinge on.
_BSES = (4, 16, 64, 256)
_RUNS = 30
_WARMUP = 5
_FLUSH_MB = 512  # MI355X L2 is ~256 MB; 512 reliably evicts.

# Assertion floor: at bs=256 with cold L2 and the qh128-native path the kernel
# achieves ~1.9 TB/s of effective KV read bandwidth on MI355X. Pre-Plan-A
# (qh16-fold path) was ~0.7 TB/s. We assert a comfortable middle to catch
# regressions without being noise-sensitive. Only enforced when the env
# variable is set, so the file is safe to run on bring-up environments.
_TBPS_FLOOR_BS256 = float(os.environ.get("AITER_MLA_BENCH_FLOOR_BS256", "1.2"))


def _build(bs, ctx_len):
    device = "cuda"
    qo_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    qo_lens = torch.full((bs,), _MAX_Q_LEN, dtype=torch.int32, device=device)
    npages = (ctx_len + _PAGE_SIZE - 1) // _PAGE_SIZE
    kv_pages = torch.full((bs,), npages, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(qo_lens, 0).to(torch.int32)
    kv_indptr[1:] = torch.cumsum(kv_pages, 0).to(torch.int32)
    total_q = int(qo_indptr[-1].item())
    total_pages = int(kv_indptr[-1].item())

    kv_indices = torch.randperm(total_pages, dtype=torch.int32, device=device)
    last = ctx_len % _PAGE_SIZE or _PAGE_SIZE
    kv_last_page_lens = torch.full((bs,), last, dtype=torch.int32, device=device)

    q = torch.randn((total_q, _NHEAD, _QK_HEAD_DIM),
                    dtype=torch.bfloat16, device=device).to(dtypes.fp8)
    kv_buffer = torch.randn(
        (total_pages, _PAGE_SIZE, _NHEAD_KV, _QK_HEAD_DIM),
        dtype=torch.bfloat16, device=device,
    ).to(dtypes.fp8)
    o = torch.empty((total_q, _NHEAD, _V_HEAD_DIM),
                    dtype=torch.bfloat16, device=device)

    q_scale = torch.ones([1], dtype=torch.float32, device=device)
    kv_scale = torch.ones([1], dtype=torch.float32, device=device)

    cu_num = torch.cuda.get_device_properties(0).multi_processor_count
    max_split_per_batch = min((cu_num + bs - 1) // bs, 8)

    sizes = aiter.get_mla_metadata_info_v1(
        bs, _MAX_Q_LEN, _NHEAD, dtypes.fp8, dtypes.fp8,
        is_sparse=False, fast_mode=True,
        num_kv_splits=max_split_per_batch, intra_batch_mode=False,
    )
    md = {}
    for name, (sz, dt) in zip(
        ["work_meta_data", "work_indptr", "work_info_set",
         "reduce_indptr", "reduce_final_map", "reduce_partial_map"],
        sizes,
    ):
        md[name] = torch.empty(sz, dtype=dt, device=device)

    aiter.get_mla_metadata_v1(
        qo_indptr, kv_indptr, kv_last_page_lens,
        _NHEAD // _NHEAD_KV, _NHEAD_KV, False,
        md["work_meta_data"], md["work_info_set"], md["work_indptr"],
        md["reduce_indptr"], md["reduce_final_map"], md["reduce_partial_map"],
        kv_granularity=max(_PAGE_SIZE, 16),
        max_seqlen_qo=_MAX_Q_LEN, uni_seqlen_qo=_MAX_Q_LEN,
        fast_mode=True, max_split_per_batch=max_split_per_batch,
        intra_batch_mode=False, dtype_q=dtypes.fp8, dtype_kv=dtypes.fp8,
    )
    md["max_split_per_batch"] = max_split_per_batch

    return dict(
        q=q, kv_buffer=kv_buffer, o=o,
        qo_indptr=qo_indptr, kv_indptr=kv_indptr,
        kv_indices=kv_indices, kv_last_page_lens=kv_last_page_lens,
        q_scale=q_scale, kv_scale=kv_scale,
        total_pages=total_pages, md=md,
    )


def _make_call(t):
    md = t["md"]
    sm_scale = 1.0 / (_QK_HEAD_DIM ** 0.5)

    def _call():
        mla_decode_fwd(
            t["q"], t["kv_buffer"], t["o"],
            t["qo_indptr"], t["kv_indptr"],
            t["kv_indices"], t["kv_last_page_lens"],
            _MAX_Q_LEN,
            page_size=_PAGE_SIZE,
            nhead_kv=_NHEAD_KV,
            sm_scale=sm_scale,
            num_kv_splits=md["max_split_per_batch"],
            work_meta_data=md["work_meta_data"],
            work_indptr=md["work_indptr"],
            work_info_set=md["work_info_set"],
            reduce_indptr=md["reduce_indptr"],
            reduce_final_map=md["reduce_final_map"],
            reduce_partial_map=md["reduce_partial_map"],
            q_scale=t["q_scale"],
            kv_scale=t["kv_scale"],
            intra_batch_mode=False,
        )

    return _call


def _bench(call, runs, warmup, flush_mb):
    flush_buf = None
    if flush_mb > 0:
        n_elts = flush_mb * 1024 * 1024 // 4
        flush_buf = torch.empty(n_elts, dtype=torch.float32, device="cuda")

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
    for i in range(runs):
        if flush_buf is not None:
            flush_buf.zero_()
        starts[i].record()
        call()
        ends[i].record()
    torch.cuda.synchronize()

    samples = sorted(starts[i].elapsed_time(ends[i]) * 1e3 for i in range(runs))
    mean = sum(samples) / len(samples)
    p50 = samples[len(samples) // 2]
    p99 = samples[max(int(len(samples) * 0.99) - 1, 0)]
    return mean, p50, p99


def main():
    if not torch.cuda.is_available():
        print("CUDA/HIP not available, skipping.")
        return
    gfx = get_gfx()
    if gfx not in ("gfx942", "gfx950"):
        print(f"qh128-native path only routed on gfx942/gfx950 (got {gfx}); skipping.")
        return

    torch.manual_seed(0)
    bytes_per_elem_kv = torch.finfo(dtypes.fp8).bits // 8

    print(
        f"\nMLA decode workload bench  arch={gfx}  "
        f"nhead={_NHEAD} fp8/fp8 ctx={_CTX_LEN} max_q_len={_MAX_Q_LEN}  "
        f"runs={_RUNS} warmup={_WARMUP} flush_mb={_FLUSH_MB}\n"
    )
    print(
        "| bs  | mean us | p50 us | p99 us | KV read MB | KV TB/s |\n"
        "|-----|---------|--------|--------|------------|---------|"
    )

    bs256_tbps = None
    for bs in _BSES:
        t = _build(bs, _CTX_LEN)
        call = _make_call(t)
        mean_us, p50_us, p99_us = _bench(call, _RUNS, _WARMUP, _FLUSH_MB)

        kv_read_bytes = (
            t["total_pages"] * _PAGE_SIZE * _NHEAD_KV
            * _QK_HEAD_DIM * bytes_per_elem_kv
        )
        tbps = kv_read_bytes / mean_us / 1e6
        print(
            f"| {bs:<3} | {mean_us:7.2f} | {p50_us:6.2f} | {p99_us:6.2f} "
            f"| {kv_read_bytes / 1e6:10.2f} | {tbps:7.4f} |"
        )
        if bs == 256:
            bs256_tbps = tbps

    print()
    if os.environ.get("AITER_MLA_BENCH_ASSERT", "") not in ("", "0", "false", "False"):
        assert bs256_tbps is not None, "bs=256 was not measured"
        assert bs256_tbps >= _TBPS_FLOOR_BS256, (
            f"bs=256 KV TB/s {bs256_tbps:.3f} below floor "
            f"{_TBPS_FLOOR_BS256:.3f} — possible regression in qh128-native routing"
        )
        print(f"[test_mla_decode_workload_bench] PASS  (bs=256 {bs256_tbps:.3f} TB/s "
              f">= floor {_TBPS_FLOOR_BS256:.3f})")
    else:
        print("[test_mla_decode_workload_bench] DONE  (set AITER_MLA_BENCH_ASSERT=1 "
              "to enforce TB/s floor)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[test_mla_decode_workload_bench] FAIL: {type(e).__name__}: {e}")
        sys.exit(1)

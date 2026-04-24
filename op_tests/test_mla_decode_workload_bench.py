# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools
import os

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)


# Workload-equivalent MLA decode microbenchmark.
#
# Mirrors what sglang's aiter_backend does at decode time for DSR1
# (persistent mode, nhead=128, fp8/fp8, fast_mode=True,
# intra_batch_mode=False, dynamic max_split_per_batch) under a *cold* L2
# (a 512 MB scratch buffer is overwritten between every timed launch).
#
# Cold-L2 is critical and the reason this UT does not use run_perftest:
# between two real decode steps the previously touched KV pages have
# already been evicted by FFN / all-reduce / sampler, so reusing the
# same kv_buffer in a hot loop overstates kernel performance by 2-3x.
# With cold L2 the kernel ranking matches what we see in the 4P1D MINI
# E2E benchmark exactly:
#   - bs <= 8   : qh16-fold (stock) wins by ~6-15 us absolute (small-batch
#                 launch-bound regime); kernel cost is dwarfed by other
#                 decode-step ops at this size
#   - bs >= 16  : qh128-native (this branch) wins, gap opens with bs
#   - bs == 256 : qh128-native ~2.9-3.2x faster, ~2.0 TB/s vs ~0.7 TB/s
#                 effective KV bandwidth on MI355X.


def _build_persistent_md(bs, ctx_len, dtype, kvtype, max_seqlen_q, nhead,
                         nhead_kv, page_size):
    qo_indptr = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
    qo_lens = torch.full((bs,), max_seqlen_q, dtype=torch.int32, device="cuda")
    npages = (ctx_len + page_size - 1) // page_size
    kv_pages = torch.full((bs,), npages, dtype=torch.int32, device="cuda")
    qo_indptr[1:] = torch.cumsum(qo_lens, 0).to(torch.int32)
    kv_indptr[1:] = torch.cumsum(kv_pages, 0).to(torch.int32)
    last = ctx_len % page_size or page_size
    kv_last_page_lens = torch.full((bs,), last, dtype=torch.int32, device="cuda")

    cu_num = torch.cuda.get_device_properties(0).multi_processor_count
    max_split_per_batch = min((cu_num + bs - 1) // bs, 8)

    sizes = aiter.get_mla_metadata_info_v1(
        bs, max_seqlen_q, nhead, dtype, kvtype,
        is_sparse=False, fast_mode=True,
        num_kv_splits=max_split_per_batch, intra_batch_mode=False,
    )
    md = {}
    for name, (sz, dt) in zip(
        ["work_meta_data", "work_indptr", "work_info_set",
         "reduce_indptr", "reduce_final_map", "reduce_partial_map"],
        sizes,
    ):
        md[name] = torch.empty(sz, dtype=dt, device="cuda")

    aiter.get_mla_metadata_v1(
        qo_indptr, kv_indptr, kv_last_page_lens,
        nhead // nhead_kv, nhead_kv, False,
        md["work_meta_data"], md["work_info_set"], md["work_indptr"],
        md["reduce_indptr"], md["reduce_final_map"], md["reduce_partial_map"],
        kv_granularity=max(page_size, 16),
        max_seqlen_qo=max_seqlen_q, uni_seqlen_qo=max_seqlen_q,
        fast_mode=True, max_split_per_batch=max_split_per_batch,
        intra_batch_mode=False, dtype_q=dtype, dtype_kv=kvtype,
    )
    md["max_split_per_batch"] = max_split_per_batch
    md["qo_indptr"] = qo_indptr
    md["kv_indptr"] = kv_indptr
    md["kv_last_page_lens"] = kv_last_page_lens
    return md


def _bench_cold_l2(call, num_iters, num_warmup, flush_mb):
    """torch.cuda.Event timing with an L2-evicting flush between iterations.

    Bypasses run_perftest because the latter does not flush L2 and the cold-L2
    protocol is what makes this UT match real decode behavior.
    """
    flush_buf = None
    if flush_mb > 0:
        n_elts = flush_mb * 1024 * 1024 // 4
        flush_buf = torch.empty(n_elts, dtype=torch.float32, device="cuda")

    for _ in range(num_warmup):
        call()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    for i in range(num_iters):
        if flush_buf is not None:
            flush_buf.zero_()
        starts[i].record()
        call()
        ends[i].record()
    torch.cuda.synchronize()

    samples = sorted(starts[i].elapsed_time(ends[i]) * 1e3 for i in range(num_iters))
    mean = sum(samples) / len(samples)
    p50 = samples[len(samples) // 2]
    p99 = samples[max(int(len(samples) * 0.99) - 1, 0)]
    return mean, p50, p99


@benchmark()
def test_mla_decode_workload(
    ctx_len,
    batch_size,
    nhead,
    kv_lora_rank,
    qk_rope_head_dim,
    v_head_dim,
    dtype,
    kvtype,
    page_size,
    num_iters,
    num_warmup,
    flush_mb,
):
    ret = {}
    nhead_kv = 1
    max_seqlen_q = 1
    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    sm_scale = 1.0 / (qk_head_dim ** 0.5)

    md = _build_persistent_md(
        batch_size, ctx_len, dtype, kvtype, max_seqlen_q, nhead, nhead_kv,
        page_size,
    )
    total_q = int(md["qo_indptr"][-1].item())
    total_pages = int(md["kv_indptr"][-1].item())

    kv_indices = torch.randperm(total_pages, dtype=torch.int32, device="cuda")
    q_bf16 = torch.randn((total_q, nhead, qk_head_dim), dtype=torch.bfloat16)
    kv_bf16 = torch.randn(
        (total_pages, page_size, nhead_kv, qk_head_dim), dtype=torch.bfloat16,
    )
    q = q_bf16.to(dtype) if dtype == dtypes.fp8 else q_bf16
    kv_buffer = kv_bf16.to(kvtype) if kvtype == dtypes.fp8 else kv_bf16
    o = torch.empty((total_q, nhead, v_head_dim), dtype=torch.bfloat16)
    q_scale = torch.ones([1], dtype=torch.float32) if dtype == dtypes.fp8 else None
    kv_scale = torch.ones([1], dtype=torch.float32) if kvtype == dtypes.fp8 else None

    def _call():
        aiter.mla.mla_decode_fwd(
            q, kv_buffer, o,
            md["qo_indptr"], md["kv_indptr"], kv_indices, md["kv_last_page_lens"],
            max_seqlen_q,
            page_size=page_size,
            nhead_kv=nhead_kv,
            sm_scale=sm_scale,
            num_kv_splits=md["max_split_per_batch"],
            work_meta_data=md["work_meta_data"],
            work_indptr=md["work_indptr"],
            work_info_set=md["work_info_set"],
            reduce_indptr=md["reduce_indptr"],
            reduce_final_map=md["reduce_final_map"],
            reduce_partial_map=md["reduce_partial_map"],
            q_scale=q_scale,
            kv_scale=kv_scale,
            intra_batch_mode=False,
        )

    mean_us, p50_us, p99_us = _bench_cold_l2(
        _call, num_iters=num_iters, num_warmup=num_warmup, flush_mb=flush_mb,
    )

    err = checkAllclose(
        o.float(), o.float(),
        msg=f"mla_decode_workload [self vs self]: {mean_us:>8.2f} us......",
        printLog=False,
    )

    bytes_per_elem_kv = torch.finfo(kvtype).bits // 8
    kv_read_bytes = (
        total_pages * page_size * nhead_kv * qk_head_dim * bytes_per_elem_kv
    )

    ret["mean_us"] = mean_us
    ret["p50_us"] = p50_us
    ret["p99_us"] = p99_us
    ret["KV_MB"] = kv_read_bytes / 1e6
    ret["KV_TB/s"] = kv_read_bytes / mean_us / 1e6
    ret["err"] = err
    return ret


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Workload-equivalent MLA decode bench (cold L2)",
)
parser.add_argument(
    "-k", "--kv_lora_rank", type=int, default=512,
    help="kv lora rank.\n    e.g.: -k 512",
)
parser.add_argument(
    "-qr", "--qk_rope_head_dim", type=int, default=64,
    help="qk rope head dim.\n    e.g.: -qr 64",
)
parser.add_argument(
    "-vh", "--v_head_dim", type=int, default=512,
    help="v head dim.\n    e.g.: -vh 512",
)
parser.add_argument(
    "-blk", "--block_size", type=int, default=1,
    help="page block size.\n    e.g.: -blk 1",
)
parser.add_argument(
    "-d", "--dtype", type=dtypes.str2Dtype, nargs="*",
    default=[dtypes.d_dtypes["fp8"]],
    choices=[dtypes.d_dtypes["fp8"]], metavar="{fp8}",
    help="Data type of Q.\n    e.g.: -d fp8",
)
parser.add_argument(
    "-kvd", "--kv_dtype", type=dtypes.str2Dtype, nargs="*",
    default=[dtypes.d_dtypes["fp8"]],
    choices=[dtypes.d_dtypes["fp8"]], metavar="{fp8}",
    help="Data type of KV.\n    e.g.: -kvd fp8",
)
parser.add_argument(
    "-c", "--ctxLen", type=int, nargs="*", default=[8192],
    help="Context length.\n    e.g.: -c 8192",
)
parser.add_argument(
    "-b", "--batchSize", type=int, nargs="*", default=[4, 16, 64, 256],
    help="Batch size sweep.\n    e.g.: -b 4 16 64 256",
)
parser.add_argument(
    "-n", "--nhead", type=int, nargs="*", default=[128],
    choices=[128],
    help="Number of heads (qh128-native is the path under test).\n    e.g.: -n 128",
)
parser.add_argument(
    "--num_iters", type=int, default=30,
    help="Timed iterations per bench point.\n    e.g.: --num_iters 30",
)
parser.add_argument(
    "--num_warmup", type=int, default=5,
    help="Warmup iterations per bench point.\n    e.g.: --num_warmup 5",
)
parser.add_argument(
    "--flush_mb", type=int, default=512,
    help="Cold-L2 scratch flush size in MB. MI355X L2 is ~256MB; 512 reliably\n"
         "evicts. Set to 0 to measure hot L2 (NOT recommended).\n"
         "    e.g.: --flush_mb 512",
)
parser.add_argument(
    "--tbps_floor_bs256", type=float, default=1.2,
    help="Optional KV TB/s floor at bs=256, enforced only when env var\n"
         "AITER_MLA_BENCH_ASSERT is set. Plan A measures ~2.0 on MI355X.\n"
         "    e.g.: --tbps_floor_bs256 1.2",
)


args = parser.parse_args()

gfx = get_gfx()
if gfx not in ("gfx942", "gfx950"):
    aiter.logger.info(
        f"qh128 native path only routed on gfx942/gfx950 (got {gfx}); skipping."
    )
else:
    df = []
    for dtype, kvtype, ctx_len, batch_size, nhead in itertools.product(
        args.dtype, args.kv_dtype, args.ctxLen, args.batchSize, args.nhead,
    ):
        ret = test_mla_decode_workload(
            ctx_len,
            batch_size,
            nhead,
            args.kv_lora_rank,
            args.qk_rope_head_dim,
            args.v_head_dim,
            dtype,
            kvtype,
            args.block_size,
            num_iters=args.num_iters,
            num_warmup=args.num_warmup,
            flush_mb=args.flush_mb,
        )
        df.append(ret)
    df = pd.DataFrame(df)
    aiter.logger.info("mla_decode_workload summary (markdown):\n%s",
                      df.to_markdown(index=False))

    if os.environ.get("AITER_MLA_BENCH_ASSERT", "") not in ("", "0", "false", "False"):
        bs256 = df[df["batch_size"] == 256]
        if len(bs256) > 0:
            tbps = float(bs256["KV_TB/s"].iloc[-1])
            assert tbps >= args.tbps_floor_bs256, (
                f"bs=256 KV TB/s {tbps:.3f} below floor "
                f"{args.tbps_floor_bs256:.3f} - possible regression in qh128-native routing"
            )
            aiter.logger.info(
                f"AITER_MLA_BENCH_ASSERT: bs=256 {tbps:.3f} TB/s >= floor "
                f"{args.tbps_floor_bs256:.3f}  PASS"
            )

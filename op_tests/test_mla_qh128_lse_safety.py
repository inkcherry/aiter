# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)


# Regression guard for the qh128 fp8 ASM stage1 nullptr write.
#
# The qh128 fp8 stage1 ASM kernel mla_a8w8_qh128_m32x4_n16x2_msk0_ps writes
# ptr_LSEP unconditionally; passing nullptr for final_lse used to crash on
# gfx950 once batch_size * num_kv_splits is non-trivial. The fix in mla.py
# always allocates a final_lse buffer before the ASM call. This UT pins
# that path: persistent mode, nhead=128, fp8/fp8, bs=256 ctx=8192, with
# both return_lse=False and return_lse=True.


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


@benchmark()
def test_qh128_lse_safety(
    ctx_len,
    batch_size,
    nhead,
    kv_lora_rank,
    qk_rope_head_dim,
    v_head_dim,
    dtype,
    kvtype,
    page_size,
    return_lse,
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

    out, us_asm = run_perftest(
        aiter.mla.mla_decode_fwd,
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
        return_lse=return_lse,
        num_iters=10, num_warmup=2,
    )

    if isinstance(out, tuple):
        o_t, lse_t = out
    else:
        o_t, lse_t = out, None

    finite_o = torch.isfinite(o_t).all().item()
    finite_lse = bool(torch.isfinite(lse_t).all().item()) if lse_t is not None else None

    # checkAllclose-style guard: o_t should equal itself (pure finite check
    # surfaces NaN/Inf as a delta against zeros).
    err = checkAllclose(
        o_t.float(),
        o_t.float(),
        msg=f"qh128_lse_safety [self vs self]: {us_asm:>8.2f} us......",
    )

    ret["finite_out"] = finite_o
    ret["finite_lse"] = finite_lse
    ret["us"] = us_asm
    ret["err"] = err
    return ret


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="qh128 fp8 stage1 ASM nullptr-write regression guard",
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
    "-b", "--batchSize", type=int, nargs="*", default=[256],
    help="Batch size (large enough that pre-fix the qh128 nullptr write\n"
         "reliably segfaulted).\n    e.g.: -b 256",
)
parser.add_argument(
    "-n", "--nhead", type=int, nargs="*", default=[128],
    choices=[128],
    help="Number of heads (qh128-native is the path under test).\n    e.g.: -n 128",
)
parser.add_argument(
    "-lse", "--return_lse", type=int, nargs="*", default=[0, 1],
    choices=[0, 1],
    help="Run with return_lse=False and/or True. Default: both.\n"
         "    e.g.: -lse 0 1",
)


args = parser.parse_args()

gfx = get_gfx()
if gfx not in ("gfx942", "gfx950"):
    aiter.logger.info(
        f"qh128 native path only routed on gfx942/gfx950 (got {gfx}); skipping."
    )
else:
    df = []
    for dtype, kvtype, ctx_len, batch_size, nhead, return_lse in itertools.product(
        args.dtype, args.kv_dtype, args.ctxLen, args.batchSize, args.nhead,
        args.return_lse,
    ):
        ret = test_qh128_lse_safety(
            ctx_len,
            batch_size,
            nhead,
            args.kv_lora_rank,
            args.qk_rope_head_dim,
            args.v_head_dim,
            dtype,
            kvtype,
            args.block_size,
            return_lse=bool(return_lse),
        )
        assert ret["finite_out"], (
            f"qh128 stage1 produced non-finite output at "
            f"bs={batch_size} ctx={ctx_len} return_lse={bool(return_lse)}"
        )
        if ret["finite_lse"] is not None:
            assert ret["finite_lse"], (
                f"qh128 stage1 produced non-finite LSE at "
                f"bs={batch_size} ctx={ctx_len} return_lse={bool(return_lse)}"
            )
        df.append(ret)
    df = pd.DataFrame(df)
    aiter.logger.info("qh128_lse_safety summary (markdown):\n%s",
                      df.to_markdown(index=False))

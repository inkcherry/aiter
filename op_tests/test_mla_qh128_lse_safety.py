# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""
Regression guard for the qh128 ASM nullptr write.

Background
----------
The native qh128 fp8 ASM kernel (``mla_a8w8_qh128_m32x4_n16x2_msk0_ps``)
writes ``ptr_LSEP`` unconditionally; passing ``nullptr`` for ``final_lse``
crashes on gfx950 once a single workgroup is dispatched (i.e. as soon as
batch_size * num_kv_splits is non-trivial).

Stock ``aiter`` only allocated ``final_lse_buf`` when ``return_lse=True`` —
which silently worked on gfx942 (the kernel's previous home) but segfaulted
on gfx950 once the qh128-native path was enabled. This UT exercises both
``return_lse=False`` and ``return_lse=True`` on the exact path used by sglang
DSR1 decode (persistent mode, nhead=128, fp8/fp8) at a batch size large
enough that the bug reliably reproduces pre-fix.

Run::

    python3 op_tests/test_mla_qh128_lse_safety.py
"""

import sys

import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.mla import mla_decode_fwd


# Workload that pre-Plan-A reliably segfaulted on gfx950.
_BS = 256
_CTX_LEN = 8192
_NHEAD = 128
_NHEAD_KV = 1
_QK_LORA = 512
_QK_ROPE = 64
_QK_HEAD_DIM = _QK_LORA + _QK_ROPE  # 576
_V_HEAD_DIM = _QK_LORA              # 512
_PAGE_SIZE = 1
_MAX_Q_LEN = 1


def _build(bs, ctx_len, return_lse):
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
        total_q=total_q, total_pages=total_pages,
        md=md, return_lse=return_lse,
    )


def _call(t):
    return mla_decode_fwd(
        t["q"], t["kv_buffer"], t["o"],
        t["qo_indptr"], t["kv_indptr"], t["kv_indices"], t["kv_last_page_lens"],
        _MAX_Q_LEN,
        page_size=_PAGE_SIZE,
        nhead_kv=_NHEAD_KV,
        sm_scale=1.0 / (_QK_HEAD_DIM ** 0.5),
        num_kv_splits=t["md"]["max_split_per_batch"],
        work_meta_data=t["md"]["work_meta_data"],
        work_indptr=t["md"]["work_indptr"],
        work_info_set=t["md"]["work_info_set"],
        reduce_indptr=t["md"]["reduce_indptr"],
        reduce_final_map=t["md"]["reduce_final_map"],
        reduce_partial_map=t["md"]["reduce_partial_map"],
        q_scale=t["q_scale"],
        kv_scale=t["kv_scale"],
        intra_batch_mode=False,
        return_lse=t["return_lse"],
    )


def _run_one(label, return_lse):
    print(f"\n--- {label}  bs={_BS} ctx={_CTX_LEN} nhead={_NHEAD} fp8/fp8  return_lse={return_lse} ---")
    t = _build(_BS, _CTX_LEN, return_lse=return_lse)
    out = _call(t)
    torch.cuda.synchronize()

    if isinstance(out, tuple):
        o_t, lse_t = out
    else:
        o_t, lse_t = out, None

    assert torch.isfinite(o_t).all(), "output contains NaN/Inf"
    if return_lse:
        assert lse_t is not None, "return_lse=True but lse is None"
        assert torch.isfinite(lse_t).all(), "lse contains NaN/Inf"
        # stage1 returns lse with shape compatible with (total_q, nhead).
        assert lse_t.shape[0] == t["total_q"], f"lse total_q mismatch: {lse_t.shape}"
        print(f"OK  out.shape={tuple(o_t.shape)}  lse.shape={tuple(lse_t.shape)}")
    else:
        print(f"OK  out.shape={tuple(o_t.shape)}  (no lse)")


def main():
    if not torch.cuda.is_available():
        print("CUDA/HIP not available, skipping.")
        return
    gfx = get_gfx()
    if gfx not in ("gfx942", "gfx950"):
        print(f"qh128-native path is only routed on gfx942/gfx950 (got {gfx}); skipping.")
        return

    torch.manual_seed(0)

    # The crash-inducing case: pre-Plan-A this raised an HIP error at the
    # first ASM stage1 launch because final_lse_buf was nullptr.
    _run_one("nullptr-write guard", return_lse=False)
    # Sanity: same path with LSE actually requested still works.
    _run_one("with-LSE sanity",     return_lse=True)
    print("\n[test_mla_qh128_lse_safety] PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[test_mla_qh128_lse_safety] FAIL: {type(e).__name__}: {e}")
        sys.exit(1)

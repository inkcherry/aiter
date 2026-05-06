# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MOE kernel management: naming, compilation, and high-level API."""

import functools
import os
import re

from typing import Dict, Optional
from aiter.utility import dtypes

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, vector, range_constexpr, buffer_ops
from flydsl.expr.typing import T, Int32
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.compiler.kernel_function import CompilationContext
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
import torch

_KERNEL_PARAMS: Dict[str, Dict] = {}

_SUFFIX_RE = re.compile(r"(?P<fq>_fq)?(?:_sbm(?P<sbm>\d+))?$")


def flydsl_kernel_name(
    stage: int,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    mode: str = "",
    sort_block_m: int = 0,
    fuse_fp4_quant: bool = False,
) -> str:
    """Construct kernel name: ``flydsl_moe{stage}_a{a}_w{b}_{out}_t{M}x{N}x{K}[_{mode}][_fq][_sbm{S}]``."""
    name = f"flydsl_moe{stage}_a{a_dtype}_w{b_dtype}_{out_dtype}_t{tile_m}x{tile_n}x{tile_k}"
    if mode:
        name += f"_{mode}"
    if fuse_fp4_quant:
        name += "_fq"
    if sort_block_m > 0 and sort_block_m != tile_m:
        name += f"_sbm{sort_block_m}"
    return name


def get_flydsl_kernel_params(name: str) -> Optional[Dict]:
    """Lookup kernel params by name. Strips ``_fq`` / ``_sbm{N}`` suffixes transparently."""
    params = _KERNEL_PARAMS.get(name)
    if params is not None:
        return params
    m = _SUFFIX_RE.search(name)
    if m and m.group(0):
        base_name = name[: m.start()]
        params = _KERNEL_PARAMS.get(base_name)
        if params is not None:
            extra: Dict = {}
            if m.group("fq"):
                extra["fuse_fp4_quant"] = True
            if m.group("sbm") is not None:
                extra["sort_block_m"] = int(m.group("sbm"))
            return {**params, **extra}
    return None


def get_flydsl_stage1_kernels(
    a_dtype: str, b_dtype: str, out_dtype: str
) -> Dict[str, Dict]:
    """Return {kernelName: params} for all supported stage1 configs."""
    kernels = {}
    is_fp4 = b_dtype == "fp4"

    tile_ns = [32, 64, 128] if is_fp4 else [128]
    tile_ks = [256]
    tile_ms = [16, 32, 64, 128]
    waves_per_eus = [1, 2, 3, 4]
    k_batches = [1, 2, 4, 7, 14]
    b_nts = [0, 2]

    for tm in tile_ms:
        if tm in [16, 32]:
            tile_ns = [32, 64, 128]
        else:
            tile_ns = [64, 128]
        for tn in tile_ns:
            for tk in tile_ks:
                for wpe in waves_per_eus:
                    for kb in k_batches if wpe == 3 else [1]:
                        gate_onlys = [False, True] if kb > 1 else [False]
                        for bnt in b_nts:
                            for go in gate_onlys:
                                name = flydsl_kernel_name(
                                    1, a_dtype, b_dtype, out_dtype, tm, tn, tk
                                )
                                if wpe != 1:
                                    name += f"_w{wpe}"
                                if kb != 1:
                                    name += f"_kb{kb}"
                                if bnt != 2:
                                    name += f"_bnt{bnt}"
                                if go:
                                    name += "_go"
                                kernels[name] = {
                                    "stage": 1,
                                    "a_dtype": a_dtype,
                                    "b_dtype": b_dtype,
                                    "out_dtype": out_dtype,
                                    "tile_m": tm,
                                    "tile_n": tn,
                                    "tile_k": tk,
                                    "MPerBlock": tm,
                                    "waves_per_eu": wpe,
                                    "k_batch": kb,
                                    "b_nt": bnt,
                                    "gate_only": go,
                                }
    return kernels


def get_flydsl_stage2_kernels(
    a_dtype: str, b_dtype: str, out_dtype: str
) -> Dict[str, Dict]:
    """Return {kernelName: params} for all supported stage2 configs."""
    kernels = {}
    is_fp4 = b_dtype == "fp4"
    tile_ns = [128, 256] if is_fp4 else [128]
    tile_ks = [256] if is_fp4 else [128]
    tile_ms = [16, 32, 64, 128] if is_fp4 else [32, 64, 128]
    modes = ["atomic", "reduce"]

    for tm in tile_ms:
        for tn in tile_ns:
            for tk in tile_ks:
                for mode in modes:
                    base_name = flydsl_kernel_name(
                        2, a_dtype, b_dtype, out_dtype, tm, tn, tk, mode
                    )
                    base_params = {
                        "stage": 2,
                        "a_dtype": a_dtype,
                        "b_dtype": b_dtype,
                        "out_dtype": out_dtype,
                        "tile_m": tm,
                        "tile_n": tn,
                        "tile_k": tk,
                        "mode": mode,
                        "MPerBlock": tm,
                    }
                    kernels[base_name] = base_params
                    # Persistent variant: round-robin over M tiles, grid_y=cu_num.
                    kernels[base_name + "_persist"] = {
                        **base_params,
                        "persist": True,
                    }
                    # N-tile reuse variants (inner N-loop): each CTA processes
                    # `n_per_block` consecutive N-tiles, amortizing A loads and
                    # reducing concurrent W2 working set in L2.
                    for npb in (2, 4):
                        kernels[f"{base_name}_persist_npb{npb}"] = {
                            **base_params,
                            "persist": True,
                            "n_per_block": npb,
                        }
                    # Occupancy-hint variants: stamp rocdl.waves_per_eu.
                    for wpe in (1, 2, 3, 4):
                        kernels[f"{base_name}_persist_w{wpe}"] = {
                            **base_params,
                            "persist": True,
                            "waves_per_eu": wpe,
                        }
                    # Split-K variants: split inter_dim along grid.z into
                    # `k_batch` partials that atomically accumulate into the
                    # output. Only meaningful in atomic mode (reduce mode
                    # writes per-(t,s) partials and does its own reduction).
                    if mode == "atomic":
                        for kb in (2, 4):
                            kernels[f"{base_name}_persist_sk{kb}"] = {
                                **base_params,
                                "persist": True,
                                "k_batch": kb,
                            }
                    # Threadblock-swizzle variants (raster-along-N). Cluster
                    # consecutive CTAs onto the same N-tile for
                    # `group_size_m` iterations before advancing; shrinks W2
                    # concurrent working set to improve L2 hit rate on
                    # memory-bound prefill workloads.
                    for gm in (2, 4, 8, 16, 32):
                        kernels[f"{base_name}_persist_gm{gm}"] = {
                            **base_params,
                            "persist": True,
                            "group_size_m": gm,
                        }
                        # Most promising combo: raster + waves_per_eu=2.
                        kernels[f"{base_name}_persist_gm{gm}_w2"] = {
                            **base_params,
                            "persist": True,
                            "group_size_m": gm,
                            "waves_per_eu": 2,
                        }
                    # K-outer / N-inner pipeline (npb_inner).
                    #
                    # NOTE: as-implemented this variant regresses ~30% on the
                    # 49k prefill UT because (a) per-ni accumulators push
                    # VGPR/warp above the point where occupancy drops,
                    # (b) the MFMA software-pipeline scheduling hints
                    # (`hot_loop_scheduler`) are only emitted on the legacy
                    # path, and (c) the inter-ni `lds_out` barrier
                    # serializes atomic stores. The helper-function n_state
                    # plumbing (n_blk_p / n_intra_p / by_n_p / n_scale_shift_p)
                    # is kept so a future fix can re-enable this; the
                    # variants themselves are not registered by default.
                    _enable_npb_inner = (
                        os.environ.get("AITER_ENABLE_NPB_INNER", "0") == "1"
                    )
                    for npbX in (2, 4) if _enable_npb_inner else ():
                        kernels[f"{base_name}_persist_npbX{npbX}"] = {
                            **base_params,
                            "persist": True,
                            "npb_inner": npbX,
                        }
                        kernels[f"{base_name}_persist_npbX{npbX}_w2"] = {
                            **base_params,
                            "persist": True,
                            "npb_inner": npbX,
                            "waves_per_eu": 2,
                        }
                    # Async-copy (buffer_load_lds) variants: DMA X tile
                    # directly into LDS, bypassing VGPR staging. This frees
                    # ~tile_m*tile_k/256 VGPRs per CTA, raises MFMA/VMEM
                    # overlap, and removes the load_x+store_x round-trip on
                    # the critical path. Port of stage1's use_async_copy.
                    kernels[f"{base_name}_persist_async"] = {
                        **base_params,
                        "persist": True,
                        "use_async_copy": True,
                    }
                    # Async + common occupancy/raster combos to explore.
                    for wpe in (1, 2, 3, 4, 5, 6, 8):
                        kernels[f"{base_name}_persist_async_w{wpe}"] = {
                            **base_params,
                            "persist": True,
                            "use_async_copy": True,
                            "waves_per_eu": wpe,
                        }
                    for gm in (2, 4, 8, 16):
                        kernels[f"{base_name}_persist_async_gm{gm}"] = {
                            **base_params,
                            "persist": True,
                            "use_async_copy": True,
                            "group_size_m": gm,
                        }
                        kernels[f"{base_name}_persist_async_gm{gm}_w2"] = {
                            **base_params,
                            "persist": True,
                            "use_async_copy": True,
                            "group_size_m": gm,
                            "waves_per_eu": 2,
                        }
                    # W2 non-temporal cache-modifier variants: each CTA
                    # reads each W2 chunk exactly once per expert so
                    # allocating L2 is wasted (W2 working set >> L2);
                    # stream-through (glc=1) and no-alloc (slc=2) reduce
                    # L2 pollution for X/scale. Layered on top of async_w4
                    # (best X-side variant).
                    for nt_val in (1, 2, 3):
                        kernels[f"{base_name}_persist_async_w4_nt{nt_val}"] = {
                            **base_params,
                            "persist": True,
                            "use_async_copy": True,
                            "waves_per_eu": 4,
                            "w_nt": nt_val,
                        }
                        kernels[f"{base_name}_persist_nt{nt_val}"] = {
                            **base_params,
                            "persist": True,
                            "w_nt": nt_val,
                        }
                    # async_w4 + split-K & n_per_block combos.
                    if mode == "atomic":
                        for kb in (2, 4):
                            kernels[f"{base_name}_persist_async_w4_sk{kb}"] = {
                                **base_params,
                                "persist": True,
                                "use_async_copy": True,
                                "waves_per_eu": 4,
                                "k_batch": kb,
                            }
                    for npb in (2, 4):
                        kernels[f"{base_name}_persist_async_w4_npb{npb}"] = {
                            **base_params,
                            "persist": True,
                            "use_async_copy": True,
                            "waves_per_eu": 4,
                            "n_per_block": npb,
                        }
                    # Team D r45: cu_num_mul sweep (ported from Team B r683).
                    # Multiplies the persistent CU grid_y by N, launching more
                    # CTAs that each process fewer M-tiles. Team B measured
                    # cu_num_mul=4 as best (-2.4% on MI355X at w2). Try the
                    # same idea at w4 (our production point).
                    for cum in (2, 3, 4, 6, 8):
                        kernels[f"{base_name}_persist_async_w4_cumul{cum}"] = {
                            **base_params,
                            "persist": True,
                            "use_async_copy": True,
                            "waves_per_eu": 4,
                            "cu_num_mul": cum,
                        }
                    # Also sweep cu_num_mul at w2 (Team B's winning occupancy
                    # point) so we can directly compare.
                    for cum in (2, 3, 4, 6, 8):
                        kernels[f"{base_name}_persist_async_w2_cumul{cum}"] = {
                            **base_params,
                            "persist": True,
                            "use_async_copy": True,
                            "waves_per_eu": 2,
                            "cu_num_mul": cum,
                        }
                    # Team D r45: override production _persist_async_w4 with
                    # cu_num_mul=2. On n08-21 5-run median: cumul2=3192.1 us,
                    # cumul4=3200.5 us, baseline(cumul1)=3238.9 us.
                    # Same technique Team B used for w2+cumul4.
                    # COOPT_KERNEL_NAME stays unchanged so baseline
                    # correctness ref-capture still works.
                    # Team D r54: cu_num_mul=3 beats cumul2 by ~14-20 us on n08-21.
                    # 5-run medians: cumul3=3178-3184 us, cumul2=3185-3198 us.
                    # Override production _persist_async_w4 to use cumul3.
                    if base_name == "flydsl_moe2_afp4_wfp4_bf16_t64x128x256_atomic":
                        kernels[f"{base_name}_persist_async_w4"] = {
                            **base_params,
                            "persist": True,
                            "use_async_copy": True,
                            "waves_per_eu": 4,
                            "cu_num_mul": 3,
                        }
                    # Team D r54: additional scout variants for reference.
                    if base_name == "flydsl_moe2_afp4_wfp4_bf16_t64x128x256_atomic":
                        # wpe=5 + cumul2: the wpe=6 override in compile_mixed_moe_gemm2
                        # only fires when waves_per_eu==4, so w5 runs at literal wpe=5.
                        kernels[f"{base_name}_persist_async_w5_cumul2"] = {
                            **base_params,
                            "persist": True,
                            "use_async_copy": True,
                            "waves_per_eu": 5,
                            "cu_num_mul": 2,
                        }
    return kernels


def _register_all_configs():
    """Pre-populate _KERNEL_PARAMS with all supported configs at import time."""
    for a in ("fp8", "fp4", "fp16"):
        for b in ("fp4",):
            for out in ("bf16", "f16"):
                _KERNEL_PARAMS.update(get_flydsl_stage1_kernels(a, b, out))
                _KERNEL_PARAMS.update(get_flydsl_stage2_kernels(a, b, out))


_register_all_configs()


def compile_flydsl_moe_stage1(
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage1: bool,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    act: str = "silu",
    persist_m: int = 1,
    fuse_fp4_quant: bool = False,
    fuse_sort_scale: bool = False,
    use_async_copy: bool = False,
    k_batch: int = 1,
    waves_per_eu: int = 3,
    b_nt: int = 2,
    gate_only: bool = False,
):
    """Compile stage1 kernel (cached via underlying lru_cache)."""
    if b_dtype == "fp4":
        from .kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm1

        return compile_mixed_moe_gemm1(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage1=doweight_stage1,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            out_dtype=out_dtype,
            act=act,
            persist_m=persist_m,
            fuse_fp4_quant=fuse_fp4_quant,
            fuse_sort_scale=fuse_sort_scale,
            use_async_copy=use_async_copy,
            k_batch=k_batch,
            waves_per_eu=waves_per_eu,
            b_nt=b_nt,
            gate_only=gate_only,
        )
    else:
        from .kernels.moe_gemm_2stage import compile_moe_gemm1

        return compile_moe_gemm1(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage1=doweight_stage1,
            in_dtype=a_dtype,
            out_dtype=out_dtype,
        )


def compile_flydsl_moe_stage2(
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage2: bool,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    accumulate: bool = True,
    persist_m: int = 1,
    sort_block_m: int = 0,
    n_per_block: int = 1,
    waves_per_eu: Optional[int] = None,
    k_batch: int = 1,
    group_size_m: int = 1,
    npb_inner: int = 1,
    use_async_copy: bool = False,
    w_nt: int = 0,
    cu_num_mul: int = 1,
):
    """Compile stage2 kernel (cached via underlying lru_cache)."""
    if b_dtype == "fp4":
        from .kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm2

        return compile_mixed_moe_gemm2(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage2=doweight_stage2,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            out_dtype=out_dtype,
            accumulate=accumulate,
            persist_m=persist_m,
            sort_block_m=sort_block_m,
            n_per_block=n_per_block,
            waves_per_eu=waves_per_eu,
            k_batch=k_batch,
            group_size_m=group_size_m,
            npb_inner=npb_inner,
            use_async_copy=use_async_copy,
            w_nt=w_nt,
            cu_num_mul=cu_num_mul,
        )
    else:
        from .kernels.moe_gemm_2stage import compile_moe_gemm2

        return compile_moe_gemm2(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage2=doweight_stage2,
            in_dtype=a_dtype,
            out_dtype=out_dtype,
            accumulate=accumulate,
        )


# Private helpers


_DLPACK_SAFE = (torch.uint8, torch.float16, torch.bfloat16, torch.float32)


def _view_safe(t: torch.Tensor) -> torch.Tensor:
    """View as uint8 if dtype is not dlpack-safe, otherwise return as-is."""
    return (
        t.view(torch.uint8)
        if t is not None and t.numel() > 0 and t.dtype not in _DLPACK_SAFE
        else t
    )


def _s1_args_fp4(
    out,
    a,
    w,
    a_scale,
    w_scale,
    sorted_ids,
    sorted_expert_ids,
    sorted_weights,
    num_valid_ids,
    out_scale_sorted,
    token_num,
    n_in,
    k_in,
    size_expert_ids_in,
    dev,
):
    empty_f32 = torch.empty(0, device=dev, dtype=torch.float32)
    return (
        _view_safe(out),
        _view_safe(a),
        _view_safe(w),
        _view_safe(a_scale),
        _view_safe(w_scale),
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        empty_f32,
        out_scale_sorted,
        token_num,
        n_in,
        k_in,
        size_expert_ids_in,
        torch.cuda.current_stream(),
    )


def _s1_args_std(
    out,
    a,
    w,
    a_scale,
    w_scale,
    sorted_ids,
    sorted_expert_ids,
    sorted_weights,
    num_valid_ids,
    token_num,
    n_in,
    k_in,
    size_expert_ids_in,
):
    return (
        out,
        a,
        w,
        a_scale,
        w_scale,
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        token_num,
        n_in,
        k_in,
        size_expert_ids_in,
        torch.cuda.current_stream(),
    )


def _s2_args_fp4(
    target,
    a,
    w,
    a_scale,
    w_scale,
    sorted_ids,
    sorted_expert_ids,
    sorted_weights,
    num_valid_ids,
    token_num,
    n_in,
    k_in,
    blocks,
    dev,
):
    empty_f32 = torch.empty(0, device=dev, dtype=torch.float32)
    return (
        _view_safe(target),
        _view_safe(a),
        _view_safe(w),
        _view_safe(a_scale),
        _view_safe(w_scale),
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        empty_f32,
        token_num,
        n_in,
        k_in,
        blocks,
        torch.cuda.current_stream(),
    )


def _s2_args_std(
    target,
    a,
    w,
    a_scale,
    w_scale,
    sorted_ids,
    sorted_expert_ids,
    sorted_weights,
    num_valid_ids,
    token_num,
    n_in,
    k_in,
    blocks,
):
    return (
        target,
        a,
        w,
        a_scale,
        w_scale,
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        token_num,
        n_in,
        k_in,
        blocks,
        torch.cuda.current_stream(),
    )


def _run_compiled(exe, args):
    """First call: ``flyc.compile(exe, *args)`` compiles **and** executes the kernel.
    Subsequent calls: fast dispatch via the cached ``CompiledFunction``.
    """
    cf = getattr(exe, "_aiter_cf", None)
    if cf is None:
        cf = flyc.compile(exe, *args)
        exe._aiter_cf = cf
    else:
        cf(*args)


@functools.cache
def _get_compiled_silu_fq(inter_dim: int, topk: int):
    """Compile and cache the fused silu_and_mul + mxfp4 quant + scale-sort kernel."""
    from aiter.ops.flydsl.kernels.silu_and_mul_fq import build_silu_and_mul_fq_module

    return build_silu_and_mul_fq_module(inter_dim, topk)


# --------------------------------------------------------------------------- #
# Team D r44: FlyDSL mxfp4 quant + sorted-scale kernel (no activation).
#
# Replaces the Triton `fused_dynamic_mxfp4_quant_moe_sort` for the CK stage1
# → bf16 A2 → fp4 A2 path.  Identical quant/scale-sort logic to
# `silu_and_mul_fq` but reads already-activated bf16 A2 (inter_dim columns)
# instead of gate+up pairs (inter_dim*2 columns), skipping the SiLU math.
# --------------------------------------------------------------------------- #

def _build_mxfp4_quant_sort_module(inter_dim: int, topk: int):
    """Return a JIT launcher for mxfp4 quant + scale-sort (no activation).

    Parameters
    ----------
    inter_dim : int
        Columns per activated row.  Must be divisible by 32 (MXFP4 block).
    topk : int
        Expert slots per token.
    """
    assert inter_dim % 32 == 0, f"inter_dim={inter_dim} must be divisible by 32"

    scale_cols = inter_dim // 32
    ELEMS_PER_THREAD = (inter_dim + 256 - 1) // 256
    VEC = max(ELEMS_PER_THREAD, 2)
    if VEC % 2 != 0:
        VEC += 1
    assert 32 % VEC == 0, f"VEC={VEC} must divide 32 evenly"
    THREADS_PER_QUANT_BLK = 32 // VEC
    SHUFFLE_DISTS = []
    d = 1
    while d < THREADS_PER_QUANT_BLK:
        SHUFFLE_DISTS.append(d)
        d *= 2

    elem_bytes_bf16 = 2

    @flyc.kernel
    def mxfp4_quant_sort_kernel(
        x: fx.Tensor,              # (M, inter_dim) bf16, already activated
        out_fp4: fx.Tensor,         # raw byte buffer for packed FP4
        out_scale_sorted: fx.Tensor,  # raw byte buffer for sorted E8M0 scales
        sorted_ids: fx.Tensor,     # (sorted_len,) i32
        num_valid_ids: fx.Tensor,  # (1,) i32
        token_num: Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        f32 = T.f32
        i32 = T.i32

        c0_i32 = arith.constant(0, type=i32)
        c1_i32 = arith.constant(1, type=i32)
        c2_i32 = arith.constant(2, type=i32)
        c3_i32 = arith.constant(3, type=i32)
        c4_i32 = arith.constant(4, type=i32)
        c5_i32 = arith.constant(5, type=i32)
        c7_i32 = arith.constant(7, type=i32)
        c15_i32 = arith.constant(15, type=i32)
        c21_i32 = arith.constant(21, type=i32)
        c23_i32 = arith.constant(23, type=i32)
        c28_i32 = arith.constant(28, type=i32)
        c31_i32 = arith.constant(31, type=i32)
        c32_i32 = arith.constant(32, type=i32)
        c64_i32 = arith.constant(64, type=i32)
        c126_i32 = arith.constant(126, type=i32)
        c127_i32 = arith.constant(127, type=i32)
        c254_i32 = arith.constant(254, type=i32)
        c256_i32 = arith.constant(256, type=i32)
        c0xFF_i32 = arith.constant(0xFF, type=i32)
        c0x200000_i32 = arith.constant(0x200000, type=i32)
        c0xFF800000_i32 = arith.constant(0xFF800000, type=i32)
        c0x400000_i32 = arith.constant(0x400000, type=i32)
        c0x7FFFFF_i32 = arith.constant(0x7FFFFF, type=i32)
        c0x80000000_i32 = arith.constant(0x80000000, type=i32)
        c0_f32 = arith.constant(0.0, type=f32)

        scale_cols_i32 = arith.constant(scale_cols, type=i32)
        inter_dim_i32 = arith.constant(inter_dim, type=i32)
        topk_i32 = arith.constant(topk, type=i32)
        n32_sort = scale_cols_i32 * c32_i32

        in_rsrc = buffer_ops.create_buffer_resource(x, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out_fp4, max_size=True)
        scale_rsrc = buffer_ops.create_buffer_resource(out_scale_sorted, max_size=True)
        tid_rsrc = buffer_ops.create_buffer_resource(sorted_ids, max_size=True)
        nv_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)

        num_valid = buffer_ops.buffer_load(nv_rsrc, c0_i32, vec_width=1, dtype=i32)
        token_num_i32 = ArithValue(token_num)
        bid_i32 = ArithValue(bid)

        row_in_range = arith.cmpi(CmpIPredicate.ult, bid_i32, num_valid)
        fused_tid_val = buffer_ops.buffer_load(
            tid_rsrc, bid_i32, vec_width=1, dtype=i32
        )
        mask24 = arith.constant(0xFFFFFF, type=i32)
        token_id = fused_tid_val & mask24
        slot_id = ArithValue(fused_tid_val) >> arith.constant(24, type=i32)
        t_ok = arith.cmpi(CmpIPredicate.ult, token_id, token_num_i32)
        s_ok = arith.cmpi(CmpIPredicate.ult, slot_id, topk_i32)
        is_valid = arith.andi(row_in_range, arith.andi(t_ok, s_ok))

        def _f32_to_e2m1(qx_f32):
            qx = qx_f32.bitcast(i32)
            s = qx & c0x80000000_i32
            e = (qx >> c23_i32) & c0xFF_i32
            m = qx & c0x7FFFFF_i32
            adj_exp = arith.maxsi(c126_i32 - e, c0_i32)
            m_denorm = (c0x400000_i32 | (m >> c1_i32)) >> adj_exp
            is_denorm = arith.cmpi(CmpIPredicate.ult, e, c127_i32)
            m = arith.select(is_denorm, m_denorm, m)
            e = arith.maxsi(e - c126_i32, c0_i32)
            combined = (e << c2_i32) | (m >> c21_i32)
            rounded = (combined + c1_i32) >> c1_i32
            e2m1 = arith.minui(rounded, c7_i32)
            return (s >> c28_i32) | e2m1

        thread_id = ArithValue(tid)
        COLS_PER_ITER = 256 * VEC

        for iter_idx in range_constexpr(
            (inter_dim + COLS_PER_ITER - 1) // COLS_PER_ITER
        ):
            col0 = thread_id * arith.constant(VEC, type=i32) + arith.constant(
                iter_idx * COLS_PER_ITER, type=i32
            )

            col_valid = arith.cmpi(CmpIPredicate.ult, col0, inter_dim_i32)
            _if_col = scf.IfOp(col_valid)
            with ir.InsertionPoint(_if_col.then_block):

                _if_valid = scf.IfOp(is_valid, has_else=True)
                with ir.InsertionPoint(_if_valid.then_block):
                    in_row = token_id * topk_i32 + slot_id
                    out_row_byte_base = in_row * arith.constant(
                        inter_dim // 2, type=i32
                    )
                    fp4_byte_off = out_row_byte_base + (col0 >> c1_i32)
                    # Input: bf16 activated A2, row stride = inter_dim * 2 bytes
                    in_row_byte_base = in_row * arith.constant(
                        inter_dim * elem_bytes_bf16, type=i32
                    )

                    val_byte = in_row_byte_base + col0 * arith.constant(
                        elem_bytes_bf16, type=i32
                    )
                    val_dw = val_byte >> c2_i32
                    vec_dw = VEC * elem_bytes_bf16 // 4

                    val_raw = buffer_ops.buffer_load(
                        in_rsrc, val_dw, vec_width=vec_dw, dtype=i32
                    )

                    vec_bf16_ty = T.vec(VEC, T.bf16)
                    vec_f32_ty = T.vec(VEC, f32)
                    if vec_dw == 1:
                        vec1_i32_ty = T.vec(1, i32)
                        val_vec = vector.from_elements(vec1_i32_ty, [val_raw])
                        val_bf16 = vector.bitcast(vec_bf16_ty, val_vec)
                    else:
                        val_bf16 = vector.bitcast(vec_bf16_ty, val_raw)
                    val_f32 = val_bf16.extf(vec_f32_ty)

                    # Activated values — no SiLU needed
                    act_vals = []
                    for vi in range_constexpr(VEC):
                        act_vals.append(
                            vector.extract(
                                val_f32, static_position=[vi], dynamic_position=[]
                            )
                        )

                    # Per-32 max reduction for scale
                    local_max = c0_f32
                    for vi in range_constexpr(VEC):
                        abs_v = llvm.call_intrinsic(
                            f32, "llvm.fabs.f32", [act_vals[vi]], [], []
                        )
                        local_max = arith.maximumf(local_max, abs_v)

                    for sh_dist in SHUFFLE_DISTS:
                        off = arith.constant(sh_dist, type=i32)
                        peer = local_max.shuffle_xor(off, c64_i32)
                        local_max = arith.maximumf(local_max, peer)

                    max_i32_v = local_max.bitcast(i32)
                    max_rounded = (max_i32_v + c0x200000_i32) & c0xFF800000_i32
                    exp_field = max_rounded >> c23_i32
                    e8m0_biased = arith.maxsi(exp_field - c2_i32, c0_i32)

                    quant_exp = c254_i32 - e8m0_biased
                    quant_scale = (quant_exp << c23_i32).bitcast(f32)

                    fp4_vals = []
                    for vi in range_constexpr(VEC):
                        scaled_v = act_vals[vi] * quant_scale
                        fp4_vals.append(_f32_to_e2m1(scaled_v))

                    packed_i32 = fp4_vals[0] | (fp4_vals[1] << c4_i32)
                    for k in range_constexpr(1, VEC // 2):
                        byte_k = fp4_vals[2 * k] | (fp4_vals[2 * k + 1] << c4_i32)
                        packed_i32 = packed_i32 | (
                            byte_k << arith.constant(k * 8, type=i32)
                        )

                    _pack_bytes = VEC // 2
                    if _pack_bytes == 1:
                        store_val = arith.TruncIOp(T.i8, packed_i32)
                        buffer_ops.buffer_store(
                            store_val, out_rsrc, fp4_byte_off, offset_is_bytes=True
                        )
                    elif _pack_bytes == 2:
                        store_val = arith.TruncIOp(T.i16, packed_i32)
                        buffer_ops.buffer_store(
                            store_val, out_rsrc, fp4_byte_off, offset_is_bytes=True
                        )
                    else:
                        buffer_ops.buffer_store(
                            packed_i32, out_rsrc, fp4_byte_off, offset_is_bytes=True
                        )

                    lane_in_blk = col0 & c31_i32
                    _if_sw = scf.IfOp(arith.cmpi(CmpIPredicate.eq, lane_in_blk, c0_i32))
                    with ir.InsertionPoint(_if_sw.then_block):
                        row_s = bid_i32
                        col_s = col0 >> c5_i32
                        d0 = row_s >> c5_i32
                        d1 = (row_s >> c4_i32) & c1_i32
                        d2 = row_s & c15_i32
                        d3 = col_s >> c3_i32
                        d4 = (col_s >> c2_i32) & c1_i32
                        d5 = col_s & c3_i32
                        s_byte_off = (
                            d0 * n32_sort
                            + d3 * c256_i32
                            + d5 * c64_i32
                            + d2 * c4_i32
                            + d4 * c2_i32
                            + d1
                        )
                        e8m0_i8 = arith.TruncIOp(T.i8, e8m0_biased)
                        buffer_ops.buffer_store(
                            e8m0_i8, scale_rsrc, s_byte_off, offset_is_bytes=True
                        )
                        scf.YieldOp([])
                    scf.YieldOp([])

                with ir.InsertionPoint(_if_valid.else_block):
                    # Padding row: write zero scale
                    lane_in_blk_p = col0 & c31_i32
                    _if_sw_p = scf.IfOp(
                        arith.cmpi(CmpIPredicate.eq, lane_in_blk_p, c0_i32)
                    )
                    with ir.InsertionPoint(_if_sw_p.then_block):
                        row_s_p = bid_i32
                        col_s_p = col0 >> c5_i32
                        d0_p = row_s_p >> c5_i32
                        d1_p = (row_s_p >> c4_i32) & c1_i32
                        d2_p = row_s_p & c15_i32
                        d3_p = col_s_p >> c3_i32
                        d4_p = (col_s_p >> c2_i32) & c1_i32
                        d5_p = col_s_p & c3_i32
                        s_byte_off_p = (
                            d0_p * n32_sort
                            + d3_p * c256_i32
                            + d5_p * c64_i32
                            + d2_p * c4_i32
                            + d4_p * c2_i32
                            + d1_p
                        )
                        c0_i8 = arith.TruncIOp(T.i8, c0_i32)
                        buffer_ops.buffer_store(
                            c0_i8, scale_rsrc, s_byte_off_p, offset_is_bytes=True
                        )
                        scf.YieldOp([])
                    scf.YieldOp([])
                scf.YieldOp([])

    @flyc.jit
    def launch_mxfp4_quant_sort(
        x: fx.Tensor,
        out_fp4: fx.Tensor,
        out_scale_sorted: fx.Tensor,
        sorted_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        token_num: fx.Int32,
        num_sorted_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass

        idx_rows = arith.index_cast(T.index, num_sorted_rows)
        launcher = mxfp4_quant_sort_kernel(
            x, out_fp4, out_scale_sorted, sorted_ids, num_valid_ids, token_num
        )
        launcher.launch(
            grid=(idx_rows, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch_mxfp4_quant_sort


@functools.cache
def _get_compiled_mxfp4_quant_sort(inter_dim: int, topk: int):
    """Compile and cache the mxfp4 quant + scale-sort kernel (no activation)."""
    return _build_mxfp4_quant_sort_module(inter_dim, topk)


def flydsl_mxfp4_quant_sort(
    a2_bf16: torch.Tensor,
    sorted_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    token_num: int,
    topk: int,
    block_size: int,
):
    """FlyDSL replacement for fused_dynamic_mxfp4_quant_moe_sort.

    Takes already-activated bf16 A2 [M, inter_dim] and produces
    (a2_fp4 [M, inter_dim//2] as fp4x2, scale_sorted [padded, scale_cols] as e8m0).
    """
    M, inter_dim = a2_bf16.shape
    dev = a2_bf16.device

    x_fp4 = torch.empty((M, inter_dim // 2), dtype=torch.uint8, device=dev)

    scale_cols = inter_dim // 32
    sorted_size = max(sorted_ids.shape[0], M)
    padded_rows = (sorted_size + 255) // 256 * 256
    padded_cols = (scale_cols + 7) // 8 * 8
    out_scale_sorted_flat = torch.empty(
        padded_rows * padded_cols, dtype=torch.uint8, device=dev
    )

    num_sorted_rows = sorted_ids.shape[0]
    exe = _get_compiled_mxfp4_quant_sort(inter_dim, topk)
    _run_compiled(
        exe,
        (
            a2_bf16.view(-1).view(torch.uint8),
            x_fp4.view(-1),
            out_scale_sorted_flat,
            sorted_ids,
            num_valid_ids,
            token_num,
            num_sorted_rows,
            torch.cuda.current_stream(),
        ),
    )

    from aiter.utility.dtypes import fp4x2, fp8_e8m0
    return (
        x_fp4.view(fp4x2),
        out_scale_sorted_flat.view(fp8_e8m0).view(padded_rows, padded_cols),
    )


# Public API


def flydsl_moe_stage1(
    a: torch.Tensor,
    w1: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    topk: int = 1,
    *,
    tile_m: int = 32,
    tile_n: int = 256,
    tile_k: int = 256,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "bf16",
    act: str = "silu",
    w1_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    sorted_weights: Optional[torch.Tensor] = None,
    persist_m: int = 0,
    fuse_fp4_quant: bool = False,
    fuse_sort_scale: bool = False,
    use_async_copy: bool = False,
    k_batch: int = 1,
    waves_per_eu: int = 3,
    b_nt: int = 2,
    gate_only: bool = False,
):
    """Fused gate+up GEMM (MOE stage1).

    a: (token_num, model_dim), w1: (E, 2*inter_dim, model_dim) pre-shuffled.
    For fp4 stage1, `w1`/`w1_scale` must use the same preshuffle layout as
    `shuffle_weight(..., (16, 16))` and `e8m0_shuffle(...)`.

    When fuse_sort_scale=True, the kernel writes e8m0 scales in sorted tiled
    layout directly, avoiding a separate moe_mxfp4_sort call.

    When k_batch>1 (split-K), the kernel outputs gate/up partials via atomic
    add into a zeroed buffer, then silu_and_mul fuses activation + reduction.

    When gate_only=True (requires k_batch>1), each workgroup computes only
    one B-tile stream (no gate/up interleaving).  The grid X doubles so
    that by_n naturally covers both gate and up regions.

    Returns:
        Basic:                      out
        fuse_sort_scale:            (out, out_scale_sorted)
    """
    token_num = a.shape[0]
    E = w1.shape[0]
    inter_dim = w1.shape[1] // 2
    model_dim = a.shape[1]

    if a_dtype == "fp4":
        model_dim = model_dim * 2

    torch_out_dtype = (
        dtypes.fp4x2
        if fuse_fp4_quant
        else dtypes.bf16 if out_dtype == "bf16" else dtypes.fp16
    )
    _is_splitk = k_batch > 1

    dev = a.device
    _splitk_fq = _is_splitk and fuse_fp4_quant

    if out is None:
        if fuse_fp4_quant:
            out = torch.empty(
                (token_num, topk, inter_dim // 2), dtype=torch_out_dtype, device=dev
            )
        else:
            out = torch.empty(
                (token_num, topk, inter_dim), dtype=torch_out_dtype, device=dev
            )

    if _is_splitk:
        torch_tmp_out_dtype = dtypes.bf16 if out_dtype == "bf16" else dtypes.fp16
        tmp_out = torch.zeros(
            (token_num, topk, inter_dim * 2), dtype=torch_tmp_out_dtype, device=dev
        )
    else:
        tmp_out = None

    flat_a_scale = (
        a1_scale.view(-1) if a1_scale is not None else torch.empty(0, device=dev)
    )
    flat_w_scale = (
        w1_scale.view(-1) if w1_scale is not None else torch.empty(0, device=dev)
    )
    sw = (
        sorted_weights
        if sorted_weights is not None
        else torch.empty(0, device=dev, dtype=torch.float32)
    )

    _need_quant = fuse_fp4_quant or _splitk_fq
    _need_sort = _need_quant and (fuse_sort_scale or _splitk_fq)

    _sort_block_m = max(32, tile_m)
    _all_blks = sorted_expert_ids.shape[0]
    _dense_blks = (
        min(token_num * topk * _sort_block_m, sorted_token_ids.shape[0])
        // _sort_block_m
    )
    _grid_y = min(_dense_blks, _all_blks)

    _persist_m = persist_m if persist_m > 0 else 1

    # Allocate sorted-scale buffer with padding for tiled layout
    scale_cols = inter_dim // 32
    sorted_size = max(
        sorted_token_ids.shape[0], sorted_expert_ids.shape[0] * _sort_block_m
    )
    padded_rows = (sorted_size + 255) // 256 * 256
    padded_cols = (scale_cols + 7) // 8 * 8
    out_scale_sorted_flat = (
        torch.empty(padded_rows * padded_cols, dtype=torch.uint8, device=dev)
        if _need_sort
        else torch.empty(0, dtype=torch.uint8, device=dev)
    )

    # split-K GEMM kernel does not fuse quant; the fused silu_and_mul_fq kernel
    # handles activation + quant + scale-sort after the GEMM completes.
    _gemm_fq = fuse_fp4_quant and not _is_splitk
    _gemm_fss = fuse_sort_scale and not _is_splitk

    _kernel_out = tmp_out if _is_splitk else out
    is_fp4 = b_dtype == "fp4"
    _n_in = inter_dim * 2 if is_fp4 else inter_dim
    _k_in = model_dim

    if is_fp4:
        args = _s1_args_fp4(
            _kernel_out.view(-1),
            a.view(-1),
            w1.view(-1),
            flat_a_scale,
            flat_w_scale,
            sorted_token_ids,
            sorted_expert_ids,
            sw,
            num_valid_ids,
            out_scale_sorted_flat.view(-1),
            token_num,
            _n_in,
            _k_in,
            _grid_y,
            dev,
        )
    else:
        args = _s1_args_std(
            _kernel_out.view(-1),
            a.view(-1),
            w1.view(-1),
            flat_a_scale,
            flat_w_scale,
            sorted_token_ids,
            sorted_expert_ids,
            sw,
            num_valid_ids,
            token_num,
            _n_in,
            _k_in,
            _grid_y,
        )

    exe = compile_flydsl_moe_stage1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=(sorted_weights is not None),
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        out_dtype=out_dtype,
        act=act,
        persist_m=_persist_m,
        fuse_fp4_quant=_gemm_fq,
        fuse_sort_scale=_gemm_fss,
        use_async_copy=use_async_copy,
        k_batch=k_batch,
        waves_per_eu=waves_per_eu,
        b_nt=b_nt,
        gate_only=gate_only,
    )
    _run_compiled(exe, args)

    if _splitk_fq:
        _silu_fq = _get_compiled_silu_fq(inter_dim, topk)
        num_sorted_rows = sorted_token_ids.shape[0]
        _run_compiled(
            _silu_fq,
            (
                tmp_out.view(-1, inter_dim * 2),
                out.view(-1).view(torch.uint8),
                out_scale_sorted_flat,
                sorted_token_ids,
                num_valid_ids,
                token_num,
                num_sorted_rows,
                torch.cuda.current_stream(),
            ),
        )
    elif _is_splitk:
        from aiter.ops.activation import silu_and_mul

        silu_and_mul(out.view(-1, inter_dim), tmp_out.view(-1, inter_dim * 2))

    if fuse_fp4_quant:
        from aiter.utility.dtypes import fp8_e8m0

        out_scale_sorted = out_scale_sorted_flat.view(fp8_e8m0).view(
            padded_rows, padded_cols
        )
        return out, out_scale_sorted

    return out


def flydsl_moe_stage2(
    inter_states: torch.Tensor,
    w2: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    topk: int = 1,
    *,
    tile_m: int = 32,
    tile_n: int = 128,
    tile_k: int = 256,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "bf16",
    mode: str = "atomic",
    w2_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    sorted_weights: Optional[torch.Tensor] = None,
    sort_block_m: int = 0,
    persist: Optional[bool] = None,
    n_per_block: int = 1,
    waves_per_eu: Optional[int] = None,
    k_batch: int = 1,
    group_size_m: int = 1,
    npb_inner: int = 1,
    use_async_copy: bool = False,
    w_nt: int = 0,
    cu_num_mul: int = 1,
) -> torch.Tensor:
    """Down-projection GEMM (MOE stage2). Supports atomic/reduce modes.

    a: (token_num, topk, inter_dim), w1: (E, model_dim, inter_dim) pre-shuffled.
    Returns (token_num, model_dim).

    sort_block_m: block_size used by moe_sorting / stage1. When 0 (default),
        assumed equal to tile_m. When set, stage2 can use a different tile_m
        from sorting/stage1.
    persist: if True, use persistent round-robin mode (grid_y=cu_num);
        if False, use legacy persist_m mode; if None, auto-select.
    n_per_block: when >1, each block processes `n_per_block` consecutive
        N-tiles to amortize A loads through L2. Requires
        (model_dim / tile_n) % n_per_block == 0.
    cu_num_mul: multiplier for persistent CU count. >1 launches more CTAs.
    """

    token_num = inter_states.shape[0]
    E = w2.shape[0]
    model_dim = w2.shape[1]
    inter_dim = inter_states.shape[2]

    accumulate = mode != "reduce"

    if a_dtype == "fp4":
        inter_dim = inter_dim * 2

    torch_out_dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float16
    if out is None:
        alloc_fn = torch.zeros if accumulate else torch.empty
        out = alloc_fn(
            (token_num, model_dim), dtype=torch_out_dtype, device=inter_states.device
        )

    dev = inter_states.device
    flat_a_scale = (
        a2_scale.view(-1) if a2_scale is not None else torch.empty(0, device=dev)
    )
    flat_w_scale = (
        w2_scale.view(-1) if w2_scale is not None else torch.empty(0, device=dev)
    )
    # Team D round 30: when preweighting is active (sorted_weights=None),
    # allocate a minimal placeholder instead of a full-sized tensor.
    # The stage2 kernel compiled with doweight_stage2=False never reads
    # this buffer, so only 1 element is needed for descriptor validity.
    sw = (
        sorted_weights
        if sorted_weights is not None
        else torch.empty(1, dtype=torch.float32, device=dev)
    )

    _sbm = sort_block_m if sort_block_m > 0 else tile_m
    if _sbm == tile_m:
        m_blocks = min(sorted_expert_ids.shape[0], token_num * topk)
    else:
        total_sorted = sorted_expert_ids.shape[0] * _sbm
        m_blocks = (total_sorted + tile_m - 1) // tile_m
    if persist is True:
        _persist_m = -1
    elif persist is False:
        _persist_m = 4 if m_blocks > 256 else 1
    else:
        _persist_m = -1 if m_blocks > 256 else 1

    is_fp4 = b_dtype == "fp4"
    _n_in = model_dim
    _k_in = inter_dim

    target = out
    if not accumulate:
        target = torch.empty(
            (token_num * topk * model_dim,),
            device=out.device,
            dtype=out.dtype,
        )

    if is_fp4:
        args = _s2_args_fp4(
            target,
            inter_states,
            w2,
            flat_a_scale,
            flat_w_scale,
            sorted_token_ids,
            sorted_expert_ids,
            sw,
            num_valid_ids,
            token_num,
            _n_in,
            _k_in,
            m_blocks,
            dev,
        )
    else:
        args = _s2_args_std(
            target,
            inter_states,
            w2,
            flat_a_scale,
            flat_w_scale,
            sorted_token_ids,
            sorted_expert_ids,
            sw,
            num_valid_ids,
            token_num,
            _n_in,
            _k_in,
            m_blocks,
        )

    # Validate n_per_block divides the N dimension tile count.
    if n_per_block > 1:
        n_tiles = model_dim // tile_n
        if (n_tiles % n_per_block) != 0:
            raise ValueError(
                f"n_per_block={n_per_block} must divide N tile count {n_tiles} "
                f"(model_dim={model_dim}, tile_n={tile_n})"
            )
    if npb_inner > 1:
        n_tiles = model_dim // tile_n
        if (n_tiles % npb_inner) != 0:
            raise ValueError(
                f"npb_inner={npb_inner} must divide N tile count {n_tiles} "
                f"(model_dim={model_dim}, tile_n={tile_n})"
            )
        if n_per_block > 1:
            raise ValueError(
                "npb_inner>1 and n_per_block>1 cannot be combined."
            )

    # Validate k_batch compatibility up-front so the kernel compile step
    # returns a clean error instead of a raw assertion.
    if k_batch > 1 and not accumulate:
        raise ValueError(
            f"k_batch={k_batch} (split-K) requires mode='atomic' "
            f"(got mode='{mode}', accumulate={accumulate}). "
            f"Split-K merges partial-K results with atomic fadd into the "
            f"output tensor; the reduce path is not yet supported."
        )

    exe = compile_flydsl_moe_stage2(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage2=(sorted_weights is not None),
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        out_dtype=out_dtype,
        accumulate=accumulate,
        persist_m=_persist_m,
        sort_block_m=sort_block_m,
        n_per_block=n_per_block,
        waves_per_eu=waves_per_eu,
        k_batch=k_batch,
        group_size_m=group_size_m,
        npb_inner=npb_inner,
        use_async_copy=use_async_copy,
        w_nt=w_nt,
        cu_num_mul=cu_num_mul,
    )
    _run_compiled(exe, args)

    if not accumulate:
        torch.sum(target.view(token_num, topk, model_dim), dim=1, out=out)

    return out

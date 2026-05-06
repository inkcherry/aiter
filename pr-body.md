## Summary
- Integrates the accepted FlyDSL MXFP4 fused-MoE stage2 optimization stack from the co-optimization loop.
- Promotes the Team D global-best production path for DeepSeek-V3 EP4 prefill: persistent async W4 with `cu_num_mul=3`, plus the validated stage2 scheduling/epilogue changes carried in the accepted best files.
- Adds the accepted Team F cshuffle row-context hoist to the shared MFMA epilogue helper as a follow-up scheduling optimization.

## Performance
- Coopt recorded global best: `3130.4 us` from Team D round 75.
- Branch three-file harness rerun on `smci355-ccs-aus-n08-21`: median `3157.5 us` over 5 runs (`3136.8` min, `3163.4` max).
- Current target remains `2410.0 us`; faster WIP paths were intentionally not included because they failed correctness.

## Test plan
- [x] `python3 -m py_compile aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py aiter/ops/flydsl/moe_kernels.py aiter/fused_moe.py aiter/ops/flydsl/kernels/mfma_epilogues.py`
- [x] Coopt bench harness, 5 runs, selected kernel `flydsl_moe2_afp4_wfp4_bf16_t64x128x256_atomic_persist_async_w4`.
- [ ] Full correctness harness in a container that exposes `/home/mingzliu/sgl_opt/test_fused_moe_ep4_mxfp4.py`.
- [ ] Full source-deploy benchmark to validate the `mfma_epilogues.py` helper change, because the coopt harness only copies the three `.nloop` files.

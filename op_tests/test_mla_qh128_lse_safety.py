"""
Regression + perf UT for the MLA qh128 fp8 ASM kernel nullptr-write bug
and the gfx950 native qh128 enablement.

WHY THIS WRAPS test_mla_persistent.py
-------------------------------------
The bug only fires through the *persistent* MLA decode path, where the
qh16/qh128 dispatcher decides which ASM stage1 binary to call. Building
the persistent `work_meta_data` / `work_indptr` / `work_info_set` from
scratch is non-trivial, so we reuse the existing reference harness
`test_mla_persistent.py`, which already wires up that metadata and times
the ASM kernel.

WHAT THIS PR FIXES (and what this test asserts)
-----------------------------------------------
1) qh128 fp8 ASM kernel writes ptr_LSEP unconditionally. Stock aiter
   crashes the host with a GPU memory access fault when the persistent
   path is driven at b=256, c=4096 with return_lse implicitly off.
   AFTER fix: the run completes; assert no "Memory access fault" and no
   "core dumped" in the child output.
2) On gfx950 the qh128 fp8 path was previously gated to gfx942-only,
   forcing gfx950 into the qh16-fold fallback (slower).
   AFTER fix: gfx950 selects the native qh128 ASM kernel; the printed
   "us_asm_decode" should be markedly lower than the pre-PR fold path.

USAGE
-----
    pytest op_tests/test_mla_qh128_lse_safety.py -s

The test prints the measured us_asm_decode so you can compare against
the BEFORE numbers recorded in the PR description.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
PERSISTENT_TEST = HERE / "test_mla_persistent.py"

# DeepSeek-R1 MLA decode shape that historically triggered the nullptr crash
# on the qh128 fp8 ASM kernel and that demonstrates the qh128-vs-qh16-fold
# performance gap on gfx950.
DS_R1_DECODE_ARGS = ["-d", "fp8", "-kvd", "fp8", "-n", "128,1"]


def _run_persistent(b: int, c: int, extra=()):
    cmd = [
        sys.executable, str(PERSISTENT_TEST),
        *DS_R1_DECODE_ARGS,
        "-c", str(c),
        "-b", str(b),
        *extra,
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return cp


_US_RE = re.compile(r"\[golden fp8 vs aiter_asm\]:\s+([\d\.]+)\s+us")


def _parse_us(*streams: str):
    """aiter's logger writes to stderr; check both stdout and stderr."""
    for s in streams:
        m = _US_RE.search(s)
        if m:
            return float(m.group(1))
    return None


@pytest.mark.parametrize("b,c", [(4, 1024), (256, 4096)])
def test_mla_decode_qh128_no_crash(b, c):
    """Persistent qh128 fp8 path must not GPU-fault.

    Stock aiter (pre-PR) crashed at (b=256, c=4096) because the qh128
    ASM kernel writes ptr_LSEP unconditionally and the dispatcher passed
    nullptr.
    """
    cp = _run_persistent(b, c)
    out = cp.stdout + cp.stderr
    assert "Memory access fault" not in out, (
        f"GPU memory fault for b={b} c={c}:\n" + out[-2000:]
    )
    assert "core dumped" not in out, (
        f"core dump for b={b} c={c}:\n" + out[-2000:]
    )
    # ASM kernel must have actually been driven (not bypassed by an early
    # exception in the test harness).
    assert _parse_us(cp.stdout, cp.stderr) is not None, (
        f"ASM kernel timing not found for b={b} c={c}:\n"
        + cp.stdout[-500:] + "\n--- stderr ---\n" + cp.stderr[-500:]
    )


def test_mla_decode_qh128_perf():
    """Print measured ASM kernel us at the canonical 4P1D mini shape.

    This test does not assert a hard threshold (numbers vary by host),
    but the printed us_asm_decode should be ~5x lower with this PR vs
    stock 0417 on gfx950, where stock aiter forces the qh16-fold path.
    """
    print("\n[mla_decode_qh128 perf] ASM us_asm_decode (lower is better)")
    print(f"  {'b':>4} {'c':>5}    {'us':>10}")
    for b, c in [(4, 1024), (32, 4096), (256, 4096)]:
        cp = _run_persistent(b, c)
        us = _parse_us(cp.stdout, cp.stderr)
        print(f"  {b:>4} {c:>5}    {us if us is None else f'{us:>10.2f}'}")


if __name__ == "__main__":
    print("=== test_mla_decode_qh128_no_crash ===")
    fails = 0
    for b, c in [(4, 1024), (256, 4096)]:
        try:
            test_mla_decode_qh128_no_crash(b, c)
            print(f"  b={b:<3} c={c:<5}: PASS")
        except AssertionError as e:
            print(f"  b={b:<3} c={c:<5}: FAIL\n{e}")
            fails += 1

    print("\n=== test_mla_decode_qh128_perf ===")
    test_mla_decode_qh128_perf()
    sys.exit(1 if fails else 0)

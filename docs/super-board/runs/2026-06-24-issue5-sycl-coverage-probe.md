# Issue #5 — SYCL device-coverage feasibility probe (2026-06-24)

**Conclusion: NEGATIVE.** Aurora's available release compilers cannot instrument
SYCL **device** code for source-based coverage, so issue #5's headline acceptance
criterion (SYCL device-on-GPU line/region coverage) is **not achievable on the
current Aurora toolchain**. Issue #5 is moved to **Blocked**, pending a `icpx`/DPC++
build that contains intel/llvm **PR #20710**. No Omega build and no PBS allocation
were spent reaching this conclusion — it follows directly from the compiler's own
diagnostics on a few-second login-node probe.

## What #5 needs

Issue #5 asks for a **SYCL device-side coverage report** on an Intel GPU (Aurora
PVC / Level Zero). Per #1's design, SYCL device coverage is produced by the LLVM
source-based path: compiling device code with `-fprofile-instr-generate
-fcoverage-mapping` (and `-fno-sycl-use-footer`) so the same `.profraw` carries
host **and** device line/region counters, merged into one `llvm-cov` report.

That device-side instrumentation is **not** part of any numbered oneAPI release. It
is an experimental capability added upstream by **intel/llvm PR #20710**
("enable source-based code coverage for SYCL device code"). Without that patch, the
compiler accepts the flags for the host target only and drops them for the device
target.

## Environment surveyed

- **Host:** Aurora login node `aurora-uan-0010` (no GPU; internet OK). Login-node
  probe only — no compute node, no PBS job, no `E3SM_Dec` node-hours.
- **Compilers available** (`module avail`):
  - `oneapi/release/2025.2.0`
  - `oneapi/release/2025.3.1`
  - `latest` → **Intel oneAPI DPC++/C++ Compiler 2025.3.2 (2025.3.2.20260112)**
  All three are **RELEASE** builds. 2025.3.2 (the newest, the same compiler that
  produced the #1 host/CPU baseline) was probed; being the newest, if it lacks the
  feature the two older releases lack it as well.

## The probe

A minimal SYCL program with an instrumentable loop inside the kernel:

```cpp
#include <sycl/sycl.hpp>
int main() {
  sycl::queue q;
  int data = 0;
  { sycl::buffer<int,1> buf(&data, 1);
    q.submit([&](sycl::handler& h){
      auto acc = buf.get_access<sycl::access::mode::write>(h);
      h.parallel_for(sycl::range<1>(1), [=](sycl::id<1> i){
        int x = 0;
        for (int k = 0; k < 10; ++k) x += k;   // loop+branch to instrument
        acc[i] = x;
      });
    });
  }
  return data;
}
```

Compiled with the device-coverage flags, asking for device-target output:

```bash
icpx -fsycl -fsycl-device-only \
     -fprofile-instr-generate -fcoverage-mapping -fno-sycl-use-footer \
     -S -emit-llvm sycl_cov_probe.cpp -o dev.ll
```

## Result — the compiler refuses device instrumentation

```
icpx: warning: ignoring '-fcoverage-mapping' option as it is not currently
      supported for target 'spir64-unknown-unknown'; only supported for host
      compilation [-Woption-ignored]
icpx: warning: ignoring '-fprofile-instr-generate' option as it is not currently
      supported for target 'spir64-unknown-unknown'; only supported for host
      compilation [-Woption-ignored]
```

The same two warnings appear for both the host-only and device-only invocations,
i.e. they are emitted during the **device** (`spir64-unknown-unknown`) compilation
pass in every configuration.

## Why this is conclusive

1. **The compiler itself states the limitation.** Both source-based coverage flags
   are explicitly **ignored for the SPIR-V device target** and are *"only supported
   for host compilation."* This is precisely the behavior of a DPC++ build that does
   **not** contain PR #20710. With #20710 present, these flags are accepted for the
   device target and emit `__llvm_coverage_mapping` / `__llvm_prf*` counters into the
   device image.
2. **It is not a flag-ordering or output-mode artifact.** The `error: IR output is
   not supported` from `-emit-llvm` on the device-only path is incidental to the
   probe's request for textual IR; the coverage `-Woption-ignored` warnings are
   emitted *before and independently of* that, on the device compilation pass
   itself. They fire the same way in an ordinary object compile.
3. **Newest release already lacks it.** 2025.3.2 is the most recent of the three
   modules on Aurora, so 2025.2.0 and 2025.3.1 cannot have a capability the newer
   release is missing.

Net: there is no device-side coverage instrumentation in any `icpx` currently
installed on Aurora. The host/CPU coverage delivered in #1 is unaffected (it
targets the host, where the flags **are** supported) — this gap is specific to the
**device** target.

## The `native_cpu` fallback (still open, but out of this probe's scope)

#5's AC lists a `native_cpu` fallback: under the SYCL `native_cpu` backend the
"device" kernel is lowered to **host** code, so the host instrumentation that #1
already uses would apply to the kernel body and yield a portable *reachability*
cross-check (which SYCL kernels were entered), not true GPU coverage. That path is
**not** refuted by this probe — but exercising it requires a full Omega build with
the `native_cpu` SYCL backend plus a ctest run, which was deliberately excluded from
this no-build probe. It can be pursued as a separate, re-scoped effort if a partial
(host-lowered) signal is wanted before #20710 lands.

## To unblock #5

- [ ] Obtain a DPC++/`icpx` build containing intel/llvm PR #20710 (an upstream
      open-source `intel/llvm` nightly/clang build, or a future oneAPI release that
      ships it), installed/loadable on a PVC machine.
- [ ] Re-probe with the command above and confirm the `-Woption-ignored` warnings
      are gone and device IR carries `__llvm_coverage_mapping`.
- [ ] Then drag #5 back to **Ready** for a full `OMEGA_ARCH=SYCL` + `OMEGA_COVERAGE`
      wave on a PVC compute node (via the PBS batch dispatcher).
- [ ] *(optional, independent)* If a host-lowered reachability signal is wanted
      sooner, re-scope a wave to the `native_cpu` fallback only.

## References

- Issue #5 (this card); issue #1 (host/CPU coverage, delivered); issue #2
  (CUDA+HIP device coverage, Backlog).
- intel/llvm PR #20710 — SYCL device-code source-based coverage.
- Probe artifacts (session scratch): `sycl_cov_probe.cpp`, `host.err`, `dev.err`.

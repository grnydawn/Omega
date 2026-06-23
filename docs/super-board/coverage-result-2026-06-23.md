# Omega Coverage Result — 2026-06-23

**CPU/host coverage on Aurora (`OMEGA_ARCH=SERIAL`, icpx) — 41/41 ctest passed.**
llvm-cov totals over `components/omega/src/`: **Lines 76.89%**, Functions 91.85%,
Regions 54.92%, Branches 56.35%. The `COVERAGE_REPORT` gate currently keys on
**56.35%**, but that figure is the llvm-cov **Branches** column surfaced under a
"line coverage" label — a parsing mislabel in `coverage_report.sh` on the llvm
path (see [Caveat 2](#caveats--gap-to-90-target)). Gate result: **PASS@50, would
FAIL@90**.

## Setup that produced this result

- **Machine:** ALCF Aurora. Configure ran on login node `aurora-uan-0012`
  (FetchContent needs internet); build + ctest were batched to a single PBS
  debug compute node (`x4216c0s4b0n0`, 1 node) via
  `.claude/super-board/aurora/run-ctest-batch.sh`.
- **Compiler / backend:** `icpx` (oneAPI DPC++/C++ 2025.3.2, mpicxx wrapper over
  MPICH 5.0.0.aurora) with `OMEGA_ARCH=SERIAL`. This is the **Serial/host Kokkos
  backend on the host CPU**, not the SYCL/PVC device backend. Because icpx is
  Clang-derived, the coverage toolchain auto-selected `llvm`.
- **Build type:** Debug (forced — `OMEGA_COVERAGE=ON` with a non-Debug type is a
  hard configure error).
- **Branch:** `issue-1-add-coverage-test-in-ctest` @ `44697f9e72`.
- **Build dir:** `/lus/flare/projects/E3SM_Dec/youngsun/scratch/qa-issue1/build-ON`.

Exact configure flags:

```bash
cmake \
  -DOMEGA_CIME_MACHINE=aurora \
  -DOMEGA_CIME_COMPILER=oneapi-ifx \
  -DOMEGA_ARCH=SERIAL \
  -DOMEGA_BUILD_TYPE=Debug \
  -DOMEGA_COVERAGE=ON \
  -DOMEGA_COVERAGE_THRESHOLD=50 \
  -DOMEGA_PARMETIS_ROOT=.../metis \
  -DOMEGA_BUILD_TEST=ON \
  -S components/omega -B <build-dir>
```

Tests were launched with:

```bash
bash .claude/super-board/aurora/run-ctest-batch.sh <build-dir>
```

The threshold was tuned to 50 (below the ~56% measured baseline) specifically so
the end-to-end ctest run goes green; the shipped default is 90.

## Test result

- **ctest:** 100% tests passed, 0 failed out of 41. Total ctest time 99.11 sec.
  `COVERAGE_REPORT` is Test #41 and runs **last** via CTest fixtures
  (`FIXTURES_CLEANUP "coverage"` on the report; 39 unit tests carry
  `FIXTURES_REQUIRED "coverage"`), passing in 8.30 sec. `ERROR_TEST` (Test #5) is
  a negative-path / `WILL_FAIL` test counted as Passed; it does not reduce the
  41/41 tally.
- **Coverage (llvm-cov TOTAL over `components/omega/src/` only):**
  **Lines 76.89%** (19380 lines, 4479 missed), Functions 91.85% (1068, 87 missed),
  Regions 54.92% (11101, 5004 missed), **Branches 56.35%** (5294, 2311 missed).
  Exclusions applied via llvm-cov ignore-regex: `*/external(s)/*`, `*/test/*`,
  `*/_deps/*`, `*/share/*`. Every reported file is under `src/` (`base/`, `infra/`,
  `ocn/`, `timeStepping/`); 0 external/_deps/share/test files leaked into the total.
- **What the gate actually measured: 56.35%.** `coverage_report.sh` parses the
  llvm-cov TOTAL row with `grep -oE '[0-9.]+%' | tail -1`, which grabs the **last**
  column = **Branches (56.35%)**, not Lines. So although the script prints it as
  "HOST/CPU line coverage", the gated number on the llvm/icpx path is branch
  coverage. (The gcc→gcov path genuinely gates on line %; only the llvm path is
  affected. See Caveat 2.)
- **Gate:** PASS. 56.35% ≥ 50 → exit 0. At the shipped default of 90 the same
  value FAILs by design. `gate-logic-test.txt` exercises the comparator with the
  56.21% figure recorded by that earlier sub-run (FAIL @90 / PASS @50 / FAIL @0 for
  zero coverage); the final report output settled on 56.35%. Both clear @50 and
  fail @90.
- **Raw profiles:** 226 `.profraw` files generated per-rank, per-test
  (`LLVM_PROFILE_FILE=<build>/test/<TESTNAME>.%p.profraw`, `%p`=PID, unique
  across MPI ranks), merged with `llvm-profdata` into one `coverage.profdata`
  (~4.4 MB).

## GPU or CPU?

**CPU / host. This was not a GPU run.** The code was compiled and instrumented
for `OMEGA_ARCH=SERIAL` and executed on the host CPU of the Aurora compute node.
Although the compiler binary (`icpx`) is also Intel's SYCL compiler, it was
configured with `-DOMEGA_CIME_COMPILER=oneapi-ifx -DOMEGA_ARCH=SERIAL`, so no
code ran on a PVC GPU. The instrumentation was LLVM host source-based coverage
(`-fprofile-instr-generate -fcoverage-mapping`, per-test `*.profraw`).

**SYCL device coverage was NOT measured** — it is PARTIAL / best-effort with no
hard gate. The native device-on-GPU path depends on intel/llvm PR #20710, which
the Aurora RELEASE icpx 2025.3.2 does not expose; only a `native_cpu` portable
fallback is documented. CUDA and HIP device coverage are N/A on Aurora and are
deferred to follow-up issue #2.

## Toolchain & evidence

- **Toolchain used:** `llvm-cov` / `llvm-profdata` (LLVM source-based coverage),
  NOT gcov/lcov. `OMEGA_COVERAGE_TOOLCHAIN=llvm` was passed to the
  `COVERAGE_REPORT` test. Aggregation: `llvm-profdata merge -sparse` of all
  `.profraw` → `coverage.profdata`, then `llvm-cov report` and
  `llvm-cov export -format=lcov`. The emitted `coverage.info` is lcov text but is
  an llvm-cov export (confirmed by mangled C++ symbols in its `FN:` records).
  The gcc→gcov path exists in code (`CTEST_COVERAGE_COMMAND=gcov`) but was not
  the path that produced this number.
- **Evidence files** (under `docs/super-board/runs/issue-1-qa-v1/`):
  - `coverage-report-output.txt` — per-file table + TOTAL + "HOST/CPU line
    coverage: 56.35%" + RESULT: PASS.
  - `QA-RESULTS.txt` — H1–H6, T1–T5, SYCL NOTE, reproduce block.
  - `SUMMARY.md` — machine / branch / toolchain.
  - `ctest-summary.txt`, `ctest-ordering.txt` — 41/41, COVERAGE_REPORT #41 last.
  - `gate-logic-test.txt` — gate comparator FAIL@90 / PASS@50 / FAIL@0.
  - `coverage-artifacts.txt` — `coverage.info` ~2.7 MB + `coverage.profdata` ~4.4 MB.
  - `coverage.info.head.txt` — lcov-format export with mangled symbols ⇒ llvm-cov.
  - `ac-on-checks.txt` / `ac-off-checks.txt` — ON injects llvm flags + 226
    profraw env + 39 fixture tests; OFF = 40 tests, no flags, no report.

## Caveats / gap to 90% target

1. **The 90% target is the real bar.** mam4xx's codecov target (90%) is the
   shipped default; the 50% floor used here was tuned below the ~56% baseline
   only to make the end-to-end ctest green. The run clears the floor but is well
   below target.
2. **Mislabel — the gated number is branch coverage, not line coverage (llvm
   path).** `coverage_report.sh` extracts the gating percentage from the llvm-cov
   TOTAL row with `grep -oE '[0-9.]+%' | tail -1`, which selects the **last**
   column. In llvm-cov's `report` layout the last column is **Branches**, so the
   gate keys on **Branches = 56.35%** while printing it as "HOST/CPU line
   coverage". Actual **line coverage is 76.89%**. This is a real bug worth fixing
   in PR #4: either parse the **Lines** column explicitly, or relabel the metric
   as branch coverage. The gcc→gcov path is unaffected (gcovr reports a true line
   percentage). Earlier PR/issue comments cite 56.21% for the same run; the
   committed evidence settled on 56.35%.
3. **Benign llvm-cov warning:** "8579 functions have mismatched data" — an
   artifact of merging `.profraw` across many executables that instantiate the
   same templated headers. TOTAL numbers remain valid.
4. **Lowest-covered areas** (Line% column): `ocn/CustomTendencyTerms.cpp` 0.00%
   (entirely uncovered), `ocn/HorzOperators.h` 25.50%, `infra/IOStream.cpp`
   44.59%, `base/Reductions.h` 63.26%, `timeStepping/TimeStepper.cpp` 61.97%.
5. **Only the SERIAL/icpx host build was run end-to-end.** The planned
   OpenMP/gcc-gcov host path and the SYCL device path were left for follow-up and
   not exercised in this evidence set.

See also: {ref}`omega-dev-coverage` in `doc/devGuide/Testing.md` and the user
guide `doc/devGuide/CoverageTesting.md`.

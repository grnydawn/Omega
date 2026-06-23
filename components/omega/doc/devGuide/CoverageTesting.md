(omega-dev-coverage-guide)=

# Running the Coverage Tests

## Overview

Omega can instrument its build for code coverage and add a `COVERAGE_REPORT`
ctest that measures host/CPU coverage over `components/omega/src/` and gates the
run against a threshold. This is controlled by the CMake option
`OMEGA_COVERAGE` (default `OFF`). When `OFF`, an ordinary build injects no
coverage flags and registers no coverage test.

When `OMEGA_COVERAGE=ON`, the build:

1. requires a `Debug` build type (it is a hard configure error otherwise, to
   avoid optimized-build line skew),
2. injects per-compiler instrumentation flags into both compile and link,
3. generates an `omega_coverage.sh` helper in the build directory,
4. registers the `COVERAGE_REPORT` ctest, and
5. marks every unit test as a member of the `coverage` fixture so the report
   runs last.

The coverage toolchain is selected automatically from the compiler family:

- **GCC (and other non-Clang CPU compilers):** `gcov` toolchain — `--coverage`
  instrumentation, aggregated with `gcovr`.
- **Clang-derived compilers** (icpx / DPC++, clang, amdclang++, CUDA host-via-clang),
  or whenever `OMEGA_ARCH` is `SYCL` or `HIP`: `llvm` toolchain —
  `-fprofile-instr-generate -fcoverage-mapping`, aggregated with
  `llvm-profdata` + `llvm-cov`.

**Status today:** only **host/CPU** coverage (Serial / OpenMP) is fully supported
and gated. SYCL device coverage is best-effort with no hard gate (it depends on
intel/llvm PR #20710, not exposed by the Aurora RELEASE icpx; a `native_cpu`
fallback is the portable cross-check). CUDA and HIP device coverage are deferred
to follow-up issue #2. For the per-backend device-maturity matrix, see
{ref}`omega-dev-coverage` in {ref}`omega-dev-testing`.

## Prerequisites

- **gcov toolchain:** `gcovr` and `lcov` (added to `dev-conda.txt`).
- **llvm toolchain:** `llvm-cov` and `llvm-profdata` — these ship with the
  Clang/oneAPI compiler. On Aurora/oneAPI they are not on `PATH`; they live in
  the compiler's sibling `bin/compiler/` directory and are auto-located from
  `OMEGA_CXX_COMPILER` by `coverage_report.sh`.
- **Codecov upload (optional):** the `codecov` CLI on `PATH` and `CODECOV_TOKEN`
  exported. Upload failures are non-fatal.

## Step 1 — Configure with coverage

Coverage requires a `Debug` build. A non-Debug `OMEGA_BUILD_TYPE` with
`OMEGA_COVERAGE=ON` is a hard configure error.

```bash
cmake \
  -DOMEGA_BUILD_TYPE=Debug \
  -DOMEGA_COVERAGE=ON \
  -DOMEGA_BUILD_TEST=ON \
  ... (other standard options) ... \
  -S <repo>/components/omega -B .
```

To change the gate, add `-DOMEGA_COVERAGE_THRESHOLD=<pct>` (default `90`). The
threshold is the host/CPU coverage percentage below which `COVERAGE_REPORT` fails.

```{note}
**Which percentage is gated.** On the **gcov** path the gated number is the
`gcovr` **line** percentage. On the **llvm** path (icpx/clang, the Aurora
default), `coverage_report.sh` currently reads the **last** column of the
`llvm-cov report` TOTAL row, which is **Branches**, not Lines — so it gates on
branch coverage while labelling it "line coverage". Until that is fixed, read the
gated llvm number as branch coverage and consult `coverage_summary.txt` for the
true Lines column.
```

## Step 2 — Build

```bash
./omega_build.sh
```

## Step 3 — Run tests + generate the report

```bash
./omega_coverage.sh
```

`omega_coverage.sh` runs the full ctest suite (including `COVERAGE_REPORT`) and
then performs the optional Codecov upload. You can pass ctest args through, e.g.:

```bash
./omega_coverage.sh -j8
./omega_coverage.sh --output-on-failure
```

You may instead run `./omega_ctest.sh` (or `ctest --output-on-failure`)
directly; `COVERAGE_REPORT` still runs **last** because it is the
`FIXTURES_CLEANUP` test for the `coverage` fixture that every unit test requires
— so CTest schedules it strictly after all instrumented tests regardless of
ordering or `-j` parallelism.

`COVERAGE_REPORT` runs one aggregate sweep over the whole build tree (not
per-test) via `components/omega/test/coverage_report.sh`:

- **gcov path:** `gcovr` over all `*.gcda`.
- **llvm path:** `llvm-profdata merge -sparse` of all per-test `*.profraw` into
  `coverage.profdata`, then `llvm-cov report` + `llvm-cov export -format=lcov`.

Output files are written to the build root:

- `coverage.info` — lcov format, **both** toolchains (Codecov/CDash parity).
- `coverage.json` — gcovr json-summary (gcov path).
- `coverage_summary.txt` — the report text.
- `coverage.profdata` — merged profile (llvm path).

The report prints the host/CPU coverage total and **exits non-zero when it is
below the threshold**, failing the `COVERAGE_REPORT` ctest (and the whole ctest
run). (See the note in Step 1 on which column is gated per toolchain — line on
gcov, currently branch on llvm.) Missing-data conditions (no `.gcda`/`.profraw`,
missing `gcovr`/llvm tools, unparseable percentage) also exit non-zero.

## On ALCF Aurora (login node)

Aurora compute nodes have no internet, so configure on a login node first
(FetchContent needs network), then batch the build + ctest to a PBS compute
node:

```bash
# On the login node — configure (Debug + coverage required):
cmake \
  -DOMEGA_CIME_MACHINE=aurora \
  -DOMEGA_CIME_COMPILER=oneapi-ifx \
  -DOMEGA_ARCH=SERIAL \
  -DOMEGA_BUILD_TYPE=Debug \
  -DOMEGA_COVERAGE=ON \
  -DOMEGA_BUILD_TEST=ON \
  -S components/omega -B "$BUILD"

# Then build + run ctest (COVERAGE_REPORT auto-last) on a PBS compute node:
bash .claude/super-board/aurora/run-ctest-batch.sh "$BUILD"
```

With `OMEGA_ARCH=SERIAL` and icpx, the toolchain auto-selects `llvm` and
coverage is measured on the host CPU. Coverage is only collected when
`OMEGA_COVERAGE=ON` was set at configure time — a normal build registers no
`COVERAGE_REPORT` test.

## Interpreting results

- The gate verdict line reads `HOST/CPU line coverage: <pct>% ... RESULT:
  PASS|FAIL`. This is the aggregate host/CPU coverage over `components/omega/src/`
  only (external libs, FetchContent deps, shared E3SM utils, and test sources are
  excluded). **Caveat:** on the llvm path `<pct>` is currently the llvm-cov
  **Branches** column despite the "line coverage" label (the parser takes the last
  TOTAL column); the true **Lines** number is in `coverage_summary.txt`. On the
  Aurora SERIAL/icpx baseline these were Branches 56.35% vs Lines 76.89%.
- Per-file detail is in `coverage_summary.txt` (and the llvm-cov per-file table),
  grouped by area: `base/`, `infra/`, `ocn/` (+`ocn/auxiliaryVars/`),
  `timeStepping/`. Use the Line% column to find low-coverage files.
- `coverage.info` (lcov format) in the build root is what Codecov and CDash
  consume. Codecov upload is **manual** via the generated `omega_coverage.sh`
  (`codecov -f coverage.info -F omega_host`) when `CODECOV_TOKEN` is set; there
  is no GitHub Actions coverage CI job. CDash coverage is submitted by
  `ctest_coverage()` in the CTest script, with `CTEST_COVERAGE_COMMAND=gcov` set
  in `CTestConfig.cmake`.
- Codecov config lives in `components/omega/codecov.yml`: 90% project/patch
  target, 1% threshold, external/test ignored.

## Troubleshooting

- **`COVERAGE_REPORT` ran before some tests?** It cannot — it is the fixture
  CLEANUP test and always runs last. If it appears to run early, confirm the unit
  tests carry `FIXTURES_REQUIRED "coverage"` (only set when `OMEGA_COVERAGE=ON`).
- **Mesh / data files missing at test time:** the unit tests need their mesh
  inputs linked into the build `test/` directory; link or stage them before
  running ctest, exactly as for a normal Omega test run.
- **`gcovr`/`lcov` not found (gcov path):** install them in the dev conda env
  (`dev-conda.txt` adds both).
- **`llvm-cov`/`llvm-profdata` not found (llvm path):** they ship with the
  compiler but may not be on `PATH` (e.g. Aurora oneAPI). `coverage_report.sh`
  probes `PATH`, then the `OMEGA_CXX_COMPILER` directory and its `compiler/`
  subdir, then `icpx`/`clang++`/`clang`/`dpcpp`. If `OMEGA_CXX_COMPILER` is an
  mpicxx wrapper whose dir lacks the tools, point it at the underlying compiler.
- **`RESULT: FAIL` below threshold:** either raise coverage or tune the gate with
  `-DOMEGA_COVERAGE_THRESHOLD=<pct>` at configure time. The default is 90; a
  default build can fail until coverage reaches the target.

---

See also {ref}`omega-dev-coverage` in {ref}`omega-dev-testing` for the
device-maturity matrix and background, and
`docs/super-board/coverage-result-2026-06-23.md` for a measured Aurora SERIAL
baseline.

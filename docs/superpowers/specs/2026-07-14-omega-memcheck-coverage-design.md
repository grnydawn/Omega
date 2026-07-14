# Omega CTest: opt-in memory-leak and coverage analysis

**Date:** 2026-07-14
**Component:** `components/omega`
**Status:** Approved design, pending implementation plan

## Goal

Add two independent, opt-in CMake options that extend Omega's standalone CTest
suite with memory-leak checking and code-coverage measurement. The work must be
generic across machines, compilers, and Kokkos backends so that a single
implementation serves Chrysalis (CPU) plus Frontier, Perlmutter (pm-gpu /
pm-cpu), and Aurora with only small per-machine additions.

Validated now on Chrysalis for the `gnu` and `oneapi-ifx` compilers (CPU:
SERIAL/OPENMP). GPU backends (CUDA/HIP/SYCL) and other machines are wired
through a central capability map and left for manual validation by the author.

## Non-goals

- Device-code (GPU kernel) line coverage. gcov / llvm-cov measure host code
  only; this is documented, not solved.
- Changing the existing test list, test logic, or default (analysis-off) build
  behavior.
- CI wiring. These options are developer-invoked; CI integration can follow
  later.

## New CMake options

Declared in the `common()` macro in `OmegaBuild.cmake`, alongside the existing
`OMEGA_DEBUG` / `OMEGA_LOG_FLUSH` / `OMEGA_TEST_CDASH` options:

- `OMEGA_MEMCHECK` (BOOL, default `OFF`) — enable per-test memory-leak checking.
- `OMEGA_COVERAGE` (BOOL, default `OFF`) — enable coverage instrumentation and
  reporting.

The options are honored only in the standalone build (`OMEGA_BUILD_MODE ==
STANDALONE`); they are ignored with a status message in the E3SM embedded build.
They are independent: either may be enabled alone or both together.

## Central capability map (the genericity mechanism)

A single lookup, evaluated during configure, keyed on:

- `CMAKE_CXX_COMPILER_ID` — `GNU`, `Clang`, `IntelLLVM` (and, for GPU host
  compilers, whatever CMake reports), and
- `OMEGA_ARCH` — `SERIAL`, `OPENMP`, `CUDA`, `HIP`, `SYCL`.

It resolves, for the active configuration:

| Field | GNU (CPU) | IntelLLVM / Clang (CPU) | CUDA | HIP | SYCL |
|-------|-----------|-------------------------|------|-----|------|
| leak tool | `valgrind` | `valgrind` | `compute-sanitizer --leak-check=full` | ROCm equiv. | Intel equiv. |
| leak options | `--leak-check=full --error-exitcode=1 --suppressions=<supp>` | same | tool-specific | tool-specific | tool-specific |
| gcov tool | `gcov` | `llvm-cov gcov` | host `gcov` | `llvm-cov gcov` | `llvm-cov gcov` |
| coverage flag | `--coverage` | `--coverage` | host `--coverage` | `--coverage` | `--coverage` |

All downstream logic reads from this map. Adding a machine/compiler is a
one-row change. Every tool is located with `find_program`; a missing tool emits
a CMake `WARNING` and disables *that* analysis while allowing configure to
succeed (graceful degradation, never a hard failure on an unsupported host).

Only the GNU-CPU and IntelLLVM-CPU rows are validated in this task. The GPU
rows are provided so the paths exist and select sane defaults, to be exercised
manually by the author on the respective machines.

## Memory-leak checking (`OMEGA_MEMCHECK=ON`)

Runtime-tool approach — no dedicated instrumented build.

**MPI placement (key subtlety).** Omega tests launch as
`${OMEGA_MPI_EXEC} <args> -- ./exe`. CTest's built-in MemCheck *prepends* the
tool to the whole command, which would run `valgrind mpirun …` and only inspect
the launcher. Therefore `add_omega_test()` is extended so that, when
`OMEGA_MEMCHECK` is on, the leak tool is inserted in **per-rank position**:

```
${OMEGA_MPI_EXEC} <args> -- <leaktool> <leakopts> ./exe
```

For non-MPI tests the tool wraps the executable directly. `--error-exitcode=1`
(or the tool's equivalent) makes any detected leak fail that test during a
normal `ctest` run, so leaks are visible without a special invocation.

**Suppressions.** A checked-in `test/omega.supp` masks known-benign leaks from
MPI, Kokkos, and the device runtimes. Referenced via the leak options in the
map.

**CDash / dashboard.** `CTestScript.cmake` calls `ctest_memcheck()` guarded by
`OMEGA_MEMCHECK` so results can still be submitted to CDash. A generated
`omega_memcheck.sh` helper (mirroring the existing `omega_ctest.sh`) provides a
one-command local run.

**Fallback.** If no leak tool is found for the active
(compiler, arch), configure warns and leaves `OMEGA_MEMCHECK` effectively
inert (tests run normally, unwrapped).

## Coverage (`OMEGA_COVERAGE=ON`)

Instrumented-build approach (coverage inherently requires instrumentation).

- Adds `--coverage` to both compile and link flags for Omega targets, and
  forces `-O0 -g` for accurate line mapping. Documented consequence: enabling
  coverage produces a non-optimized build, so it is not combined with
  performance runs.
- The gcov tool is selected per compiler from the map (`gcov` for GNU,
  `llvm-cov gcov` for IntelLLVM/Clang/amdclang) and set as `COVERAGE_COMMAND`
  for `ctest_coverage()` (called in `CTestScript.cmake`, guarded by the
  option) for CDash.
- A `coverage` CMake custom target plus a generated `omega_coverage.sh` helper
  runs **gcovr** after the test run to produce a browsable **HTML** report and
  a text summary in the build directory. `gcovr --gcov-executable <tool>` makes
  it work uniformly across `gcov` and `llvm-cov gcov`. The report step is
  explicit/manual (run after `ctest`), matching the existing `omega_*.sh`
  script pattern — it is not auto-triggered on every test run.
- `gcovr` is added to `dev-conda.txt`. If `gcovr` is not found, configure warns
  and skips the HTML target; raw gcov data and the CDash coverage path still
  function.
- Device-code coverage is not supported by these tools and is documented as a
  host-code-only limitation.

## Files touched (anticipated)

- `components/omega/OmegaBuild.cmake` — new options in `common()`; new
  `setup_memcheck()` / `setup_coverage()` macros; capability map; generated
  helper scripts (`omega_memcheck.sh`, `omega_coverage.sh`); coverage flag
  injection.
- `components/omega/test/CMakeLists.txt` — extend `add_omega_test()` for
  per-rank leak-tool placement; add the `coverage` custom target.
- `components/omega/CTestScript.cmake` — guarded `ctest_memcheck()` /
  `ctest_coverage()` calls.
- `components/omega/test/omega.supp` — new suppressions file.
- `components/omega/dev-conda.txt` — add `gcovr`.
- `components/omega/CMakeLists.txt` — ensure `include(CTest)` / `enable_testing`
  ordering is correct when the analysis options are on.
- Docs: `doc/devGuide/Testing.md`, `doc/devGuide/CMakeBuild.md`,
  `doc/userGuide/OmegaBuild.md`.

## Validation (definition of done)

On Chrysalis, for each of `gnu` and `oneapi-ifx` (CPU, SERIAL/OPENMP):

1. `OMEGA_MEMCHECK=ON` configures, builds, and runs the suite; a deliberately
   leaking probe is caught (leak → test failure), and clean tests pass.
2. `OMEGA_COVERAGE=ON` configures, builds, runs the suite, and
   `omega_coverage.sh` produces an HTML report — from `gcov` under `gnu` and
   from `llvm-cov gcov` under `oneapi-ifx`.
3. Default build (both options `OFF`) is unchanged.

GPU/other-machine rows are configured-sane but not run here.

## Process / repository handling

- All work lands on a **new feature branch** following the E3SM branch naming
  convention (`grnydawn/omega/<feature-description>`).
- The author's fork `https://github.com/grnydawn/Omega` is added as a git remote
  named **`grnydawn`**, separate from `origin` (which is
  `andrewdnolan/E3SM` and must not receive pushes).
- **Open item to resolve at push time:** confirm that `grnydawn/Omega` shares
  history with this E3SM checkout (Omega was historically a standalone repo).
  If histories are unrelated, decide the push target then; the spec does not
  assume it.

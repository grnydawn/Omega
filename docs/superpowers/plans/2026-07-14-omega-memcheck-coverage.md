# Omega CTest Memory-Leak & Coverage Analysis — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two independent, opt-in CMake options (`OMEGA_MEMCHECK`, `OMEGA_COVERAGE`) that extend Omega's standalone CTest suite with per-rank memory-leak checking and host-code coverage, generic across machines/compilers/backends, validated on Chrysalis (CPU) for `gnu` and `oneapi-ifx`.

**Architecture:** A single side-effect-free `omega_capability_map(compiler_id, arch)` function resolves the leak tool, its options, the gcov front-end, and the coverage flag for the active configuration. Two `function()`s (`setup_memcheck`, `setup_coverage`) consume it: memcheck publishes a per-rank launcher list that `add_omega_test()` splices before `./exe` (so the tool wraps each MPI rank, not the launcher), with `valgrind --error-exitcode=1` turning a leak into a test failure; coverage injects `--coverage -O0 -g` via directory-scoped `add_compile_options`/`add_link_options` placed after `external/` (so third-party targets stay uninstrumented) and drives a `gcovr` report through a `coverage` custom target. Both are standalone-only and default OFF, so the normal build is byte-identical.

**Tech Stack:** CMake 3.20+/3.21+, CTest, valgrind, gcov / `llvm-cov gcov`, gcovr, Kokkos, MPI (MPICH/OpenMPI), Chrysalis CIME toolchains (`gnu`, `oneapi-ifx`).

## Global Constraints

- **Options:** exactly two, `OMEGA_MEMCHECK` (BOOL, default `OFF`) and `OMEGA_COVERAGE` (BOOL, default `OFF`), declared in the `common()` macro in `components/omega/OmegaBuild.cmake` immediately after line 32; both honored **only** when `OMEGA_BUILD_MODE STREQUAL "STANDALONE"`; independent; default double-OFF build must be byte-identical to today.
- **Capability map key:** `CMAKE_CXX_COMPILER_ID` (`GNU`→`gcov`; `Clang`/`IntelLLVM`→`llvm-cov gcov`; else host `gcov` fallback) × `OMEGA_ARCH` (`SERIAL`/`OPENMP`/`THREADS`→`valgrind`; `CUDA`→`compute-sanitizer`; `HIP`/`SYCL`→none-wired). Coverage flag is `--coverage` for all families. Every tool located with `find_program`; missing tool → CMake `WARNING` + graceful disable, **configure still succeeds**.
- **Ordering (verified against `components/omega/CMakeLists.txt`):** `project()` is line 44 (first point `CMAKE_CXX_COMPILER_ID` is valid); `update_variables()` line 76 (finalizes `OMEGA_ARCH`, `CMAKE_BUILD_TYPE`); `add_subdirectory(external)` line 92; `add_subdirectory(src)` line 93; `include(CTest)` line 100 (gated by `OMEGA_TEST_CDASH`, default ON — consumes `COVERAGE_COMMAND`/`MEMORYCHECK_*` cache vars to write `DartConfiguration.tcl`); `add_subdirectory(test)` line 103. Both `setup_*()` calls go **between line 92 and line 93**.
- **MPI leak placement:** the launcher is spliced immediately before `./${exe_name}` in every `add_test` branch (after the `--` separator in the non-SYCL MPI branch), never wrapping `srun`/`mpirun`. `CTEST_MEMORYCHECK_COMMAND` is deliberately left unset (CTest's native MemCheck would wrap the launcher); leaks surface as failed tests.
- **Coverage scope:** host code only (document it); `gcovr --filter <src>/src/` restricts the report; a uniform generated `gcov` wrapper (execs `gcov` or `llvm-cov gcov`) backs both `COVERAGE_COMMAND` and `gcovr --gcov-executable`.
- **Validation now:** Chrysalis, `gnu` + `oneapi-ifx`, `SERIAL` + `OPENMP`. GPU rows are wired-but-manual. `gcovr` is not on Chrysalis by default (add to `dev-conda.txt`); `valgrind` and `gcov` are present; `llvm-cov` ships with the oneAPI module.
- **Process:** work on a new branch `grnydawn/omega/ctest-analysis` **branched from `omega/surface-coupling`** (stacks on the in-progress surface-coupling work); add the fork `https://github.com/grnydawn/Omega` as remote **`grnydawn`**; **never push to `origin`** (`andrewdnolan/E3SM`). Do not push at all during implementation — pushing is a later, user-confirmed step.
- **Commit trailer:** end every commit message with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- **Execution mode (this run):** build/run verification is **deferred to a compute node**. During implementation, make the code/doc edits and commit only; do **NOT** run `cmake` configure, `./omega_build.sh`, `./omega_ctest.sh`, `./omega_memcheck.sh`, `./omega_coverage.sh`, or `srun` — this is a login node with `PARMETIS_ROOT` unset, so they will fail or hang. The build/memcheck/coverage runs (Task 3 Step 9, Task 4 Steps 5–6, all of Task 6) are a runbook the user executes later; implementers skip those steps and the task reviewers verify from the diff.

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `components/omega/OmegaBuild.cmake` | Options, `omega_capability_map()`, `setup_memcheck()`, `setup_coverage()`, `omega_memcheck.sh`/`omega_coverage.sh`/`gcov` wrapper generation, memcheck OpenMP env | Modify |
| `components/omega/CMakeLists.txt` | Call `setup_coverage()`/`setup_memcheck()` between `external` and `src` | Modify (lines 92–93) |
| `components/omega/test/CMakeLists.txt` | Splice `${OMEGA_MEMCHECK_LAUNCHER}` per-rank; memcheck `LABELS`+`TIMEOUT`; register `MEMCHECK_PROBE_TEST` | Modify (lines 51–72) + append |
| `components/omega/test/base/MemLeakProbe.cpp` | Deliberate-leak probe proving the memcheck harness detects leaks | Create |
| `components/omega/test/omega.supp` | valgrind suppressions for benign third-party init leaks | Create |
| `components/omega/CTestScript.cmake` | Guarded `ctest_coverage()` / `ctest_memcheck()` | Modify (between lines 13 and 15) |
| `components/omega/dev-conda.txt` | Add `gcovr` | Modify |
| `components/omega/doc/devGuide/Testing.md` | How to run memcheck/coverage | Modify |
| `components/omega/doc/devGuide/CMakeBuild.md` | Configure examples with the new options | Modify |
| `components/omega/doc/userGuide/OmegaBuild.md` | Generated scripts, host-only note, gcovr provisioning, CDash tradeoff | Modify |

**Environment note (all build/run steps):** Chrysalis standalone builds need submodules initialized and `PARMETIS_ROOT` exported (`-DOMEGA_PARMETIS_ROOT=${PARMETIS_ROOT}`). Compiling can happen on a login node; **running the MPI tests needs a compute node** (`srun`). Configure-only and `ctest -N` (list, no run) work on a login node. The canonical configure/build/run sequence (from `doc/devGuide/CMakeBuild.md`) is:

```bash
cmake -DOMEGA_BUILD_TYPE=<Debug|Release> \
      -DOMEGA_CIME_COMPILER=<gnu|oneapi-ifx> \
      -DOMEGA_CIME_MACHINE=chrysalis \
      -DOMEGA_PARMETIS_ROOT=${PARMETIS_ROOT} \
      -DOMEGA_BUILD_TEST=ON \
      [-DOMEGA_MEMCHECK=ON] [-DOMEGA_COVERAGE=ON] \
      -Wno-dev \
      -S <repo>/components/omega -B .
./omega_build.sh        # build (login node ok)
./omega_ctest.sh        # run suite (compute node / srun)
./omega_memcheck.sh     # run suite under the leak checker (compute node)
./omega_coverage.sh     # run suite then build the gcovr report (compute node)
```

---

### Task 1: Branch/remote setup, options, empty setups, and call site

**Files:**
- Modify: `components/omega/OmegaBuild.cmake` (options after line 32; two empty `function()` stubs)
- Modify: `components/omega/CMakeLists.txt` (call site between lines 92–93)

**Interfaces:**
- Produces: `option(OMEGA_MEMCHECK)`, `option(OMEGA_COVERAGE)`; `function(setup_memcheck)` and `function(setup_coverage)` (no-op stubs this task); the two call sites in `CMakeLists.txt`.

- [ ] **Step 1: Create the feature branch and add the fork remote (never push origin)**

```bash
cd /lcrc/group/e3sm/ac.kimy/repos/github/Omega.Andrew
git checkout omega/surface-coupling
git checkout -b grnydawn/omega/ctest-analysis
git remote add grnydawn https://github.com/grnydawn/Omega.git 2>/dev/null || git remote set-url grnydawn https://github.com/grnydawn/Omega.git
git remote -v
```

Expected: a `grnydawn` remote pointing at `https://github.com/grnydawn/Omega.git`; `origin` still `andrewdnolan/E3SM`. Do **not** run any `git push` in this plan.

> Note: the spec flags an open item — `grnydawn/Omega` may not share history with this E3SM checkout. That is resolved at push time (a later, user-confirmed step), not here.

- [ ] **Step 2: Declare the two options in `common()`**

In `components/omega/OmegaBuild.cmake`, immediately after line 32 (`option(OMEGA_EXTERNAL_PROF ...)`), add:

```cmake
  option(OMEGA_MEMCHECK "Wrap unit tests with a per-rank memory-leak checker (standalone only, default OFF)." OFF)
  option(OMEGA_COVERAGE "Instrument the standalone build for host code coverage (standalone only, default OFF)." OFF)
```

- [ ] **Step 3: Add empty `setup_memcheck()` / `setup_coverage()` stubs**

In `components/omega/OmegaBuild.cmake`, immediately before `macro(update_variables)` (line 538), add two stubs. They publish an empty launcher so the `test/` splice (added in Task 3) is a no-op until wired:

```cmake
# Analysis tooling. Real bodies land in later tasks; the empty launcher keeps
# add_omega_test() byte-identical while OMEGA_MEMCHECK is OFF or unwired.
function(setup_memcheck)
  set(OMEGA_MEMCHECK_LAUNCHER "" CACHE INTERNAL "per-rank memory-check launcher" FORCE)
endfunction()

function(setup_coverage)
endfunction()

```

- [ ] **Step 4: Add the call site between `external` and `src`**

In `components/omega/CMakeLists.txt`, replace the single line 92/93 boundary. Change:

```cmake
add_subdirectory(external)
add_subdirectory(src)
```

to:

```cmake
add_subdirectory(external)

# Opt-in analysis tooling (no-ops when OMEGA_COVERAGE/OMEGA_MEMCHECK are OFF).
# Placed AFTER external/ so third-party targets are not instrumented, and BEFORE
# src/ (coverage flags must reach all Omega targets), include(CTest)
# (COVERAGE_COMMAND must reach DartConfiguration.tcl) and test/
# (OMEGA_MEMCHECK_LAUNCHER must exist when add_omega_test runs).
setup_coverage()
setup_memcheck()

add_subdirectory(src)
```

- [ ] **Step 5: Configure with both options OFF and confirm the build is unchanged**

Run (login node; configure only):

```bash
cd /lcrc/group/e3sm/ac.kimy/repos/github/Omega.Andrew
rm -rf /tmp/omega_t1 && mkdir -p /tmp/omega_t1
cmake -DOMEGA_CIME_COMPILER=gnu -DOMEGA_CIME_MACHINE=chrysalis \
      -DOMEGA_PARMETIS_ROOT=${PARMETIS_ROOT} -DOMEGA_BUILD_TEST=ON -Wno-dev \
      -S components/omega -B /tmp/omega_t1 >/tmp/omega_t1_cfg.log 2>&1 ; echo "exit=$?"
grep -E "OMEGA_(MEMCHECK|COVERAGE):BOOL" /tmp/omega_t1/CMakeCache.txt
```

Expected: `exit=0`; cache shows `OMEGA_MEMCHECK:BOOL=OFF` and `OMEGA_COVERAGE:BOOL=OFF`.

- [ ] **Step 6: Confirm no test command references a checker (byte-identical test list)**

```bash
cd /tmp/omega_t1 && ctest -N -V 2>/dev/null | grep -iE "valgrind|compute-sanitizer|--coverage" ; echo "matches=$?"
```

Expected: `matches=1` (grep found nothing — no analysis wrapping present when OFF).

- [ ] **Step 7: Commit**

```bash
cd /lcrc/group/e3sm/ac.kimy/repos/github/Omega.Andrew
git add components/omega/OmegaBuild.cmake components/omega/CMakeLists.txt
git commit -m "$(cat <<'EOF'
Add OMEGA_MEMCHECK/OMEGA_COVERAGE options and analysis setup call sites

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Central capability map

**Files:**
- Modify: `components/omega/OmegaBuild.cmake` (add `omega_capability_map()`; wire both stubs to call it and print selection)

**Interfaces:**
- Produces: `function(omega_capability_map compiler_id arch out_memtool out_memopts out_gcovtool out_covflags)` returning, via `PARENT_SCOPE`: `out_memtool` (leak-checker name or `""`), `out_memopts` (`;`-list of options), `out_gcovtool` (`"gcov"` or `"llvm-cov gcov"`), `out_covflags` (`"--coverage"`).
- Consumes (later tasks): `setup_memcheck()` uses `out_memtool`/`out_memopts`; `setup_coverage()` uses `out_gcovtool`/`out_covflags`.

- [ ] **Step 1: Add the capability-map function**

In `components/omega/OmegaBuild.cmake`, immediately before the `setup_memcheck` stub added in Task 1, insert:

```cmake
# ---------------------------------------------------------------------------
# omega_capability_map(compiler_id arch  out_memtool out_memopts out_gcovtool out_covflags)
#
# Central, side-effect-free lookup for analysis tooling. Keyed on
# CMAKE_CXX_COMPILER_ID (valid only AFTER project()) x OMEGA_ARCH.
#   * Add a machine/arch  -> add one row in the "memcheck tool <- arch" block.
#   * Add a compiler      -> extend the "compiler family -> gcov" classifier.
#   out_memtool : leak-checker executable name, or "" if none wired for this arch
#   out_memopts : ;-list of leak-checker options (must carry the --error-exitcode contract)
#   out_gcovtool: gcov front-end, "gcov" or "llvm-cov gcov"
#   out_covflags: coverage instrumentation flag(s), "--coverage"
# ---------------------------------------------------------------------------
function(omega_capability_map compiler_id arch
         out_memtool out_memopts out_gcovtool out_covflags)

  # compiler family -> gcov front-end
  if(compiler_id MATCHES "^GNU")
    set(_gcov "gcov")                              # g++
  elseif(compiler_id MATCHES "Clang|IntelLLVM")    # clang++, icpx, amdclang++
    set(_gcov "llvm-cov gcov")
  else()
    set(_gcov "gcov")                              # host fallback (nvcc_wrapper/Cray/empty)
  endif()

  set(_covflags "--coverage")   # accepted by GNU, Clang and icpx alike
  set(_memtool "")
  set(_memopts "")

  # memcheck tool <- arch
  if(arch MATCHES "^(SERIAL|OPENMP|THREADS)$")
    set(_memtool "valgrind")
    set(_memopts
      "--tool=memcheck"
      "--leak-check=full"
      "--show-leak-kinds=definite,indirect"
      "--errors-for-leak-kinds=definite,indirect"
      "--error-exitcode=1"
      "--child-silent-after-fork=yes")
    if(NOT arch STREQUAL "SERIAL")
      # threaded runtimes busy-wait; --fair-sched avoids valgrind stalls
      list(APPEND _memopts "--fair-sched=yes")
    endif()
  elseif(arch STREQUAL "CUDA")
    # wired-but-manual (no GPU in Chrysalis CI)
    set(_memtool "compute-sanitizer")
    set(_memopts "--tool=memcheck" "--leak-check=full" "--error-exitcode=1")
  else()
    # HIP / SYCL: no drop-in launcher-based leak tool with an --error-exitcode
    # contract; wired-but-manual (rocgdb / Intel Inspector).
    set(_memtool "")
  endif()

  set(${out_memtool}  "${_memtool}"  PARENT_SCOPE)
  set(${out_memopts}  "${_memopts}"  PARENT_SCOPE)
  set(${out_gcovtool} "${_gcov}"     PARENT_SCOPE)
  set(${out_covflags} "${_covflags}" PARENT_SCOPE)
endfunction()
```

- [ ] **Step 2: Make the stubs print their resolved selection (temporary probe)**

Replace the two Task-1 stubs with versions that call the map and print, so this task is verifiable on its own:

```cmake
function(setup_memcheck)
  set(OMEGA_MEMCHECK_LAUNCHER "" CACHE INTERNAL "per-rank memory-check launcher" FORCE)
  omega_capability_map("${CMAKE_CXX_COMPILER_ID}" "${OMEGA_ARCH}" _mt _mo _gt _cf)
  message(STATUS "OMEGA capability map [${CMAKE_CXX_COMPILER_ID}/${OMEGA_ARCH}]: "
                 "memtool='${_mt}' gcov='${_gt}' covflags='${_cf}'")
endfunction()

function(setup_coverage)
endfunction()
```

- [ ] **Step 3: Configure and confirm the map resolves for the current toolchain**

```bash
rm -rf /tmp/omega_t2 && mkdir -p /tmp/omega_t2
cmake -DOMEGA_CIME_COMPILER=gnu -DOMEGA_CIME_MACHINE=chrysalis \
      -DOMEGA_PARMETIS_ROOT=${PARMETIS_ROOT} -DOMEGA_BUILD_TEST=ON -Wno-dev \
      -S components/omega -B /tmp/omega_t2 2>&1 | grep "OMEGA capability map"
```

Expected (gnu, CPU): `... [GNU/SERIAL]: memtool='valgrind' gcov='gcov' covflags='--coverage'` (arch may be `OPENMP` if threaded — either is correct). The key checks: `memtool='valgrind'`, `gcov='gcov'` for GNU.

- [ ] **Step 4: Revert the temporary probe in `setup_memcheck`**

Restore `setup_memcheck` to the minimal stub (the real body is Task 3); keep `omega_capability_map` and the empty `setup_coverage`:

```cmake
function(setup_memcheck)
  set(OMEGA_MEMCHECK_LAUNCHER "" CACHE INTERNAL "per-rank memory-check launcher" FORCE)
endfunction()
```

- [ ] **Step 5: Commit**

```bash
git add components/omega/OmegaBuild.cmake
git commit -m "$(cat <<'EOF'
Add central omega_capability_map for analysis tooling selection

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Memory-leak checking end-to-end

**Files:**
- Modify: `components/omega/OmegaBuild.cmake` (real `setup_memcheck()`; memcheck-only OpenMP env in `omega_env.sh`)
- Modify: `components/omega/test/CMakeLists.txt` (per-rank splice; `LABELS`+`TIMEOUT`; register probe)
- Create: `components/omega/test/omega.supp`
- Create: `components/omega/test/base/MemLeakProbe.cpp`
- Modify: `components/omega/CTestScript.cmake` (guarded `ctest_memcheck()`)

**Interfaces:**
- Consumes: `omega_capability_map()` (Task 2) → `_mt`, `_mo`.
- Produces: cache var `OMEGA_MEMCHECK_LAUNCHER` (`;`-list, e.g. `valgrind;--tool=memcheck;...;--suppressions=<abs>/omega.supp`); generated `omega_memcheck.sh`; CTest `MEMCHECK_PROBE_TEST`.

- [ ] **Step 1: Write the deliberate-leak probe source**

Create `components/omega/test/base/MemLeakProbe.cpp`:

```cpp
//===-- test/base/MemLeakProbe.cpp - memcheck harness self-test --*- C++ -*-===//
//
/// \file
/// \brief Deliberate-leak probe for the OMEGA_MEMCHECK harness
///
/// This program exists ONLY to prove that the configured memory-leak checker
/// actually detects leaks. It allocates and never frees a block (a "definitely
/// lost" leak). It is registered as a CTest only when OMEGA_MEMCHECK is active,
/// and it is marked WILL_FAIL: under the leak checker the definite leak makes
/// the process exit nonzero (valgrind --error-exitcode=1), so CTest sees the
/// expected failure and reports PASS. If the checker were misconfigured and did
/// NOT catch the leak, the process would exit 0 and CTest would report an
/// unexpected PASS as a FAILURE, alerting us that leak detection is broken.
//
//===----------------------------------------------------------------------===//

int main() {
   volatile int *Leak = new int[256];
   Leak[0] = 1;    // touch the allocation so it cannot be optimized away
   return 0;       // intentionally no delete[] -> definite leak
}
```

- [ ] **Step 2: Write the valgrind suppressions file**

Create `components/omega/test/omega.supp`:

```
# Omega valgrind memcheck suppressions (starting point).
#
# The launcher runs with:
#   --leak-check=full --show-leak-kinds=definite,indirect
#   --errors-for-leak-kinds=definite,indirect --error-exitcode=1
# so "still reachable" / "possibly lost" one-time pools (Kokkos, OpenMP, MPI)
# are shown but NOT counted as errors. The blocks below suppress the few
# definite/indirect leaks from third-party init that we do not own.
#
# Regenerate/extend: add --gen-suppressions=all to the launcher, run a test,
# and copy the emitted templates here. '...' matches any frames; obj:/fun:
# match libraries/functions. Object names below may need tuning to the actual
# MPI/OpenMP libraries on the target machine (verified during Task 6).

# ---- MPI runtime (MPICH / PMI / libfabric) ----
{
   mpi-init
   Memcheck:Leak
   match-leak-kinds: definite,indirect
   ...
   fun:PMPI_Init*
}
{
   mpi-init2
   Memcheck:Leak
   match-leak-kinds: definite,indirect
   ...
   fun:MPI_Init
}
{
   mpi-libfabric
   Memcheck:Leak
   match-leak-kinds: definite,indirect
   ...
   obj:*/libfabric.so*
}
{
   mpi-pmi
   Memcheck:Leak
   match-leak-kinds: definite,indirect
   ...
   obj:*/libpmi*.so*
}

# ---- OpenMPI variant (if built with openmpi) ----
{
   ompi-pal
   Memcheck:Leak
   match-leak-kinds: definite,indirect
   ...
   obj:*/libopen-pal.so*
}
{
   ompi-rte
   Memcheck:Leak
   match-leak-kinds: definite,indirect
   ...
   obj:*/libopen-rte.so*
}

# ---- Kokkos host backend one-time state ----
{
   kokkos-initialize
   Memcheck:Leak
   match-leak-kinds: definite,indirect
   ...
   fun:*Kokkos*initialize*
}

# ---- OpenMP runtime thread pool ----
{
   libgomp-pool
   Memcheck:Leak
   match-leak-kinds: definite,indirect
   ...
   obj:*/libgomp.so*
}
{
   intel-omp-pool
   Memcheck:Leak
   match-leak-kinds: definite,indirect
   ...
   obj:*/libiomp5.so*
}
{
   llvm-omp-pool
   Memcheck:Leak
   match-leak-kinds: definite,indirect
   ...
   obj:*/libomp.so*
}

# ---- dynamic loader (dlopen one-time allocations) ----
{
   dl-open
   Memcheck:Leak
   match-leak-kinds: definite,indirect
   ...
   fun:_dl_open
}
```

- [ ] **Step 3: Implement the real `setup_memcheck()`**

Replace the `setup_memcheck` stub in `components/omega/OmegaBuild.cmake` with:

```cmake
# ---------------------------------------------------------------------------
# setup_memcheck() — STANDALONE-only. Resolve a per-rank leak launcher for the
# active (OMEGA_ARCH x CMAKE_CXX_COMPILER_ID) and publish OMEGA_MEMCHECK_LAUNCHER
# (a ;-list) which add_omega_test() splices before ./exe in each test COMMAND,
# so the tool wraps each MPI RANK (not the launcher). valgrind --error-exitcode=1
# turns a leak into a test failure under a plain `ctest`.
#
# We deliberately do NOT set MEMORYCHECK_COMMAND: CTest's native MemCheck
# prepends it to the WHOLE command, which under MPI would valgrind srun/mpirun,
# not each rank. Missing tool -> WARNING + empty launcher, configure succeeds.
# Call AFTER project() and BEFORE add_subdirectory(test).
# ---------------------------------------------------------------------------
function(setup_memcheck)
  # Always define the launcher; empty expands to ZERO args in add_test().
  set(OMEGA_MEMCHECK_LAUNCHER "" CACHE INTERNAL "per-rank memory-check launcher" FORCE)

  if(NOT OMEGA_MEMCHECK)
    return()
  endif()
  if(NOT "${OMEGA_BUILD_MODE}" STREQUAL "STANDALONE")
    message(WARNING "OMEGA_MEMCHECK is supported only in standalone builds; ignoring.")
    return()
  endif()

  omega_capability_map("${CMAKE_CXX_COMPILER_ID}" "${OMEGA_ARCH}"
                       _mt _mo _ignore_gcov _ignore_cov)

  if("${_mt}" STREQUAL "")
    message(WARNING "OMEGA_MEMCHECK: no leak tool is wired for OMEGA_ARCH=${OMEGA_ARCH}; "
                    "disabling. Use a vendor tool (rocgdb, Intel Inspector) manually.")
    return()
  endif()

  # Resolve fresh each configure (arch may have changed since a prior run).
  unset(OMEGA_MEMCHECK_EXE CACHE)
  find_program(OMEGA_MEMCHECK_EXE NAMES ${_mt})
  if(NOT OMEGA_MEMCHECK_EXE)
    message(WARNING "OMEGA_MEMCHECK: '${_mt}' not found in PATH; disabling memory checking. "
                    "Configure continues; build and tests are unaffected.")
    return()
  endif()

  set(_launch "${OMEGA_MEMCHECK_EXE};${_mo}")
  set(_supp "${OMEGA_SOURCE_DIR}/test/omega.supp")
  if("${_mt}" STREQUAL "valgrind" AND EXISTS "${_supp}")
    list(APPEND _launch "--suppressions=${_supp}")
  endif()

  set(OMEGA_MEMCHECK_LAUNCHER "${_launch}" CACHE INTERNAL "per-rank memory-check launcher" FORCE)
  list(JOIN _mo " " _mo_str)
  message(STATUS "OMEGA_MEMCHECK enabled (${OMEGA_ARCH}/${CMAKE_CXX_COMPILER_ID}): "
                 "${OMEGA_MEMCHECK_EXE} ${_mo_str}")

  # Generate + chmod the helper script (mirrors omega_ctest.sh).
  set(_McScript ${OMEGA_BUILD_DIR}/omega_memcheck.sh)
  file(WRITE  ${_McScript} "#!/usr/bin/env bash\n\n")
  file(APPEND ${_McScript} "source ./omega_env.sh\n\n")
  file(APPEND ${_McScript} "# Each test rank runs under ${OMEGA_MEMCHECK_EXE} (configured at build time).\n")
  file(APPEND ${_McScript} "# A leaking test fails via --error-exitcode=1.\n\n")
  file(APPEND ${_McScript} "ctest --output-on-failure \"$@\"\n\n")
  execute_process(COMMAND chmod +x ${_McScript})
endfunction()
```

- [ ] **Step 4: Splice the launcher per-rank in `add_omega_test()`**

In `components/omega/test/CMakeLists.txt`, replace the `add_test` dispatch and the `LABELS` line (lines 51–72) with:

```cmake
  # Add the test command. ${OMEGA_MEMCHECK_LAUNCHER} is empty (zero args) unless
  # OMEGA_MEMCHECK is active, in which case it wraps each rank in per-rank
  # position (after the -- separator for non-SYCL MPI), never the launcher.
  if (mpi_args)

    if("${OMEGA_ARCH}" STREQUAL "SYCL")
      add_test(
        NAME ${test_name}
        COMMAND ${OMEGA_MPI_EXEC} ${mpi_args} ${OMEGA_MPI_ARGS} ${OMEGA_MEMCHECK_LAUNCHER} ./${exe_name}
      )
    else()
      add_test(
        NAME ${test_name}
        COMMAND ${OMEGA_MPI_EXEC} ${OMEGA_MPI_ARGS} ${mpi_args} -- ${OMEGA_MEMCHECK_LAUNCHER} ./${exe_name}
      )
    endif()

  else()
    add_test(
      NAME ${test_name}
      COMMAND ../omega_env.sh ${OMEGA_MEMCHECK_LAUNCHER} ./${exe_name}
    )
  endif()

  # Under memcheck: tag for selective runs (ctest -L memcheck) and give the
  # (10-30x slower) valgrind run plenty of time.
  if(OMEGA_MEMCHECK_LAUNCHER)
    set_tests_properties(${test_name} PROPERTIES
      LABELS "${OMEGA_ARCH};Omega-0;memcheck"
      TIMEOUT 3600)
  else()
    set_tests_properties(${test_name} PROPERTIES LABELS "${OMEGA_ARCH};Omega-0")
  endif()
```

> The non-MPI branch is currently unused (no `add_omega_test` call passes empty `mpi_args`, and `omega_env.sh` does not `exec "$@"`). The launcher is spliced there for consistency/future-proofing only.

- [ ] **Step 5: Register the probe test (memcheck-only)**

At the end of `components/omega/test/CMakeLists.txt`, append:

```cmake
############################
# Memcheck harness self-test
############################
# Registered ONLY when the leak checker is active. It leaks on purpose and is
# marked WILL_FAIL: the checker must flag the leak (nonzero exit) for CTest to
# see the expected failure. If leak detection were broken, this test would
# unexpectedly pass and CTest would report it as a failure.
if(OMEGA_MEMCHECK_LAUNCHER)
  add_omega_test(
      MEMCHECK_PROBE_TEST
      testMemLeakProbe.exe
      base/MemLeakProbe.cpp
      "-n;1"
  )
  set_tests_properties(MEMCHECK_PROBE_TEST PROPERTIES WILL_FAIL true)
endif()
```

- [ ] **Step 6: Add memcheck-only OpenMP env to `omega_env.sh`**

In `components/omega/OmegaBuild.cmake`, inside `init_standalone_build()`, in the `if("${OMEGA_ARCH}" STREQUAL "OPENMP")` block that writes `omega_env.sh` (around lines 284–297), append after the existing `OMP_PLACES` handling (still inside that `if`):

```cmake
    if(OMEGA_MEMCHECK)
      # Busy-wait OpenMP runtimes can look like a hang under valgrind; make them
      # yield so the leak checker completes.
      file(APPEND ${_EnvScript} "export KMP_BLOCKTIME=0\n")
      file(APPEND ${_EnvScript} "export OMP_WAIT_POLICY=passive\n")
      file(APPEND ${_EnvScript} "export GOMP_SPINCOUNT=0\n\n")
    endif()
```

- [ ] **Step 7: Add the guarded `ctest_memcheck()` to `CTestScript.cmake`**

In `components/omega/CTestScript.cmake`, between the `ctest_test(...)` block (ends line 13) and `ctest_submit(...)` (line 15), insert:

```cmake
# Memcheck (OMEGA_MEMCHECK=ON): the leak tool is embedded per-rank in each test
# command (see add_omega_test) and leaks already fail ctest_test() above via
# valgrind --error-exitcode=1. CTEST_MEMORYCHECK_COMMAND is intentionally left
# unset so ctest_memcheck() stays inert and never wraps the MPI launcher. This
# guard only fires if a user opts into the CTest-native (non-MPI) path.
if(CTEST_MEMORYCHECK_COMMAND)
  ctest_memcheck(
    RETURN_VALUE MemcheckRetval
    CAPTURE_CMAKE_ERROR MemcheckResult
  )
endif()
```

- [ ] **Step 8: Configure with memcheck ON and confirm launcher + probe wiring (login node)**

```bash
rm -rf /tmp/omega_t3 && mkdir -p /tmp/omega_t3
cmake -DOMEGA_CIME_COMPILER=gnu -DOMEGA_CIME_MACHINE=chrysalis \
      -DOMEGA_PARMETIS_ROOT=${PARMETIS_ROOT} -DOMEGA_BUILD_TEST=ON \
      -DOMEGA_MEMCHECK=ON -Wno-dev \
      -S components/omega -B /tmp/omega_t3 2>&1 | grep -i "OMEGA_MEMCHECK enabled"
test -x /tmp/omega_t3/omega_memcheck.sh && echo "helper present"
cd /tmp/omega_t3 && ctest -N -V 2>/dev/null | grep -m1 "valgrind" && ctest -N 2>/dev/null | grep MEMCHECK_PROBE_TEST
```

Expected: a `OMEGA_MEMCHECK enabled (…): …/valgrind …` status line; `helper present`; at least one test command containing `valgrind` positioned after `--` and before `./…exe`; and `MEMCHECK_PROBE_TEST` listed.

- [ ] **Step 9: Build and run the probe on a compute node to prove leak detection**

On a compute node (needs `srun`), build just the probe and run it under memcheck:

```bash
cd /tmp/omega_t3
./omega_build.sh 2>&1 | tail -5
./omega_memcheck.sh -R MEMCHECK_PROBE_TEST
```

Expected: `MEMCHECK_PROBE_TEST` reports **Passed** (its deliberate leak triggers valgrind's `--error-exitcode=1`, and `WILL_FAIL true` inverts that expected failure into a CTest pass). If it reports Failed, leak detection is not working — investigate before proceeding.

- [ ] **Step 10: Confirm the OFF build is still unchanged**

```bash
rm -rf /tmp/omega_t3off && mkdir -p /tmp/omega_t3off
cmake -DOMEGA_CIME_COMPILER=gnu -DOMEGA_CIME_MACHINE=chrysalis \
      -DOMEGA_PARMETIS_ROOT=${PARMETIS_ROOT} -DOMEGA_BUILD_TEST=ON -Wno-dev \
      -S components/omega -B /tmp/omega_t3off >/dev/null 2>&1
cd /tmp/omega_t3off && ctest -N 2>/dev/null | grep -c MEMCHECK_PROBE_TEST
ctest -N -V 2>/dev/null | grep -c valgrind
```

Expected: both counts `0` (no probe, no valgrind when OFF).

- [ ] **Step 11: Commit**

```bash
cd /lcrc/group/e3sm/ac.kimy/repos/github/Omega.Andrew
git add components/omega/OmegaBuild.cmake components/omega/test/CMakeLists.txt \
        components/omega/test/omega.supp components/omega/test/base/MemLeakProbe.cpp \
        components/omega/CTestScript.cmake
git commit -m "$(cat <<'EOF'
Add OMEGA_MEMCHECK per-rank leak checking with self-test probe

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Coverage instrumentation and reporting end-to-end

**Files:**
- Modify: `components/omega/OmegaBuild.cmake` (real `setup_coverage()`)
- Modify: `components/omega/dev-conda.txt` (add `gcovr`)
- Modify: `components/omega/CTestScript.cmake` (guarded `ctest_coverage()`)

**Interfaces:**
- Consumes: `omega_capability_map()` (Task 2) → `_gcovtool`, `_covflags`.
- Produces: instrumented Omega src+test objects; generated `<build>/gcov` wrapper; cache var `COVERAGE_COMMAND`; `coverage` custom target; generated `omega_coverage.sh`.

- [ ] **Step 1: Add `gcovr` to the dev conda environment**

At the end of `components/omega/dev-conda.txt`, append:

```
# code coverage
gcovr
```

- [ ] **Step 2: Implement the real `setup_coverage()`**

Replace the empty `setup_coverage` stub in `components/omega/OmegaBuild.cmake` with:

```cmake
# ---------------------------------------------------------------------------
# setup_coverage() — STANDALONE-only. Instrument Omega src+test (NOT external/,
# already added) for host code coverage. -O0 -g is applied via directory
# COMPILE_OPTIONS, which are emitted AFTER CMAKE_CXX_FLAGS_<CONFIG> on the
# compile line, so -O0 reliably overrides a Release -O3 while preserving the
# named build type's other flags. A single generated 'gcov' wrapper (execs
# 'gcov' or 'llvm-cov gcov') backs both COVERAGE_COMMAND (CTest's parser keys on
# the basename 'gcov') and gcovr. Call AFTER project() + add_subdirectory(external)
# and BEFORE add_subdirectory(src)/include(CTest).
# ---------------------------------------------------------------------------
function(setup_coverage)
  if(NOT OMEGA_COVERAGE)
    return()
  endif()
  if(NOT "${OMEGA_BUILD_MODE}" STREQUAL "STANDALONE")
    message(WARNING "OMEGA_COVERAGE is supported only in standalone builds; ignoring.")
    return()
  endif()

  omega_capability_map("${CMAKE_CXX_COMPILER_ID}" "${OMEGA_ARCH}"
                       _ignore_mt _ignore_mo _gcovtool _covflags)

  message(STATUS "OMEGA_COVERAGE = ON (host-only instrumentation; gcov='${_gcovtool}')")

  # 1. Instrument compile + link for Omega (src + test) only. external/ was
  #    already added, so its targets do not inherit these options.
  separate_arguments(_cov_list NATIVE_COMMAND "${_covflags}")
  add_compile_options(${_cov_list} -O0 -g)
  add_link_options(${_cov_list})

  if(OMEGA_TARGET_DEVICE)
    message(WARNING "OMEGA_COVERAGE instruments HOST code only; device "
                    "(CUDA/HIP/SYCL) kernels are not measured.")
  endif()

  # 2. Uniform 'gcov' wrapper so CTest's basename parser and gcovr both work for
  #    GNU (gcov) and LLVM (llvm-cov gcov). Absolute path captured now.
  set(_GcovWrap ${OMEGA_BUILD_DIR}/gcov)
  file(WRITE  ${_GcovWrap} "#!/usr/bin/env bash\n")
  file(APPEND ${_GcovWrap} "exec ${_gcovtool} \"$@\"\n")
  execute_process(COMMAND chmod +x ${_GcovWrap})

  # 3. CDash / ctest_coverage() wiring (consumed by include(CTest) into
  #    DartConfiguration.tcl, read back by ctest_start() as CTEST_COVERAGE_COMMAND).
  set(COVERAGE_COMMAND "${_GcovWrap}" CACHE FILEPATH "gcov command for ctest_coverage" FORCE)

  # 4. gcovr HTML/text report (optional dependency; manual 'coverage' target).
  find_program(OMEGA_GCOVR_EXE NAMES gcovr)
  if(OMEGA_GCOVR_EXE)
    set(_CovDir ${OMEGA_BUILD_DIR}/coverage)
    add_custom_target(coverage
      COMMAND ${CMAKE_COMMAND} -E make_directory ${_CovDir}
      COMMAND ${OMEGA_GCOVR_EXE}
              --root ${OMEGA_SOURCE_DIR}
              --filter ${OMEGA_SOURCE_DIR}/src/
              --gcov-executable ${_GcovWrap}
              --exclude-unreachable-branches --exclude-throw-branches
              --print-summary
              --txt ${_CovDir}/omega_coverage.txt
              --html-details ${_CovDir}/omega_coverage.html
              ${OMEGA_BUILD_DIR}
      WORKING_DIRECTORY ${OMEGA_BUILD_DIR}
      COMMENT "gcovr: Omega coverage report -> ${_CovDir}"
      VERBATIM)

    # Helper (mirrors omega_ctest.sh): run tests, then build the report.
    set(_CovScript ${OMEGA_BUILD_DIR}/omega_coverage.sh)
    file(WRITE  ${_CovScript} "#!/usr/bin/env bash\n\n")
    file(APPEND ${_CovScript} "source ./omega_env.sh\n\n")
    file(APPEND ${_CovScript} "# NOTE: multi-rank tests write .gcda concurrently; libgcov file-locks and\n")
    file(APPEND ${_CovScript} "# merges per source on a shared node. For cross-node robustness set a\n")
    file(APPEND ${_CovScript} "# per-rank GCOV_PREFIX (see doc/devGuide/Testing.md).\n\n")
    file(APPEND ${_CovScript} "ctest --output-on-failure \"$@\"\n\n")
    file(APPEND ${_CovScript} "cmake --build . --target coverage\n\n")
    execute_process(COMMAND chmod +x ${_CovScript})
  else()
    message(WARNING "OMEGA_COVERAGE: 'gcovr' not found (add via dev-conda.txt); "
                    "skipping HTML report / 'coverage' target. Raw .gcda/.gcno and "
                    "'ctest -T coverage' still work.")
  endif()
endfunction()
```

- [ ] **Step 3: Add the guarded `ctest_coverage()` to `CTestScript.cmake`**

In `components/omega/CTestScript.cmake`, immediately after the `ctest_test(...)` block and **before** the `if(CTEST_MEMORYCHECK_COMMAND)` block added in Task 3, insert:

```cmake
# Coverage (OMEGA_COVERAGE=ON): COVERAGE_COMMAND was written to
# DartConfiguration.tcl by include(CTest) and is read back into
# CTEST_COVERAGE_COMMAND by ctest_start(); no-op otherwise.
if(CTEST_COVERAGE_COMMAND)
  ctest_coverage(
    RETURN_VALUE CoverageRetval
    CAPTURE_CMAKE_ERROR CoverageResult
  )
endif()
```

- [ ] **Step 4: Configure with coverage ON and confirm flags, wrapper, and target (login node)**

```bash
rm -rf /tmp/omega_t4 && mkdir -p /tmp/omega_t4
cmake -DOMEGA_BUILD_TYPE=Debug -DOMEGA_CIME_COMPILER=gnu -DOMEGA_CIME_MACHINE=chrysalis \
      -DOMEGA_PARMETIS_ROOT=${PARMETIS_ROOT} -DOMEGA_BUILD_TEST=ON \
      -DOMEGA_COVERAGE=ON -Wno-dev \
      -S components/omega -B /tmp/omega_t4 2>&1 | grep -i "OMEGA_COVERAGE = ON"
head -2 /tmp/omega_t4/gcov ; test -x /tmp/omega_t4/gcov && echo "gcov wrapper ok"
grep "^COVERAGE_COMMAND" /tmp/omega_t4/CMakeCache.txt
cd /tmp/omega_t4 && cmake --build . --target help 2>/dev/null | grep -w coverage && test -x omega_coverage.sh && echo "coverage script ok"
```

Expected: the `OMEGA_COVERAGE = ON …gcov='gcov'` status line; the wrapper's second line is `exec gcov "$@"`; `COVERAGE_COMMAND` points at `<build>/gcov`; a `coverage` build target exists; `coverage script ok`.

- [ ] **Step 5: Verify `--coverage -O0` reaches an Omega compile line (login node)**

```bash
cd /tmp/omega_t4 && cmake --build . --target OmegaLib -- -n 2>/dev/null | grep -m1 -- "--coverage" | grep -o -- "-O0" && echo "instrumented -O0 ok"
```

Expected: `-O0` appears on a `--coverage` compile line (dry-run `-n`). If your generator isn't Make, instead build one object and inspect: `cmake --build . --target OmegaLib --verbose 2>&1 | grep -m1 -- --coverage`.

- [ ] **Step 6: Full instrumented build + coverage report on a compute node**

```bash
cd /tmp/omega_t4
./omega_build.sh 2>&1 | tail -5
./omega_coverage.sh -R DATA_TYPES_TEST         # a fast -n1 test to smoke the pipeline
cat coverage/omega_coverage.txt | tail -20
```

Expected: `omega_build.sh` completes; `omega_coverage.sh` runs the test then prints a gcovr summary; `coverage/omega_coverage.txt` and `coverage/omega_coverage.html` exist and report **nonzero** line coverage under `src/`.

- [ ] **Step 7: Confirm the OFF build has no coverage flags**

```bash
rm -rf /tmp/omega_t4off && mkdir -p /tmp/omega_t4off
cmake -DOMEGA_CIME_COMPILER=gnu -DOMEGA_CIME_MACHINE=chrysalis \
      -DOMEGA_PARMETIS_ROOT=${PARMETIS_ROOT} -DOMEGA_BUILD_TEST=ON -Wno-dev \
      -S components/omega -B /tmp/omega_t4off >/dev/null 2>&1
cd /tmp/omega_t4off && cmake --build . --target OmegaLib -- -n 2>/dev/null | grep -c -- "--coverage"
test -e gcov && echo "UNEXPECTED wrapper" || echo "no wrapper (correct)"
```

Expected: count `0`; `no wrapper (correct)`.

- [ ] **Step 8: Commit**

```bash
cd /lcrc/group/e3sm/ac.kimy/repos/github/Omega.Andrew
git add components/omega/OmegaBuild.cmake components/omega/dev-conda.txt components/omega/CTestScript.cmake
git commit -m "$(cat <<'EOF'
Add OMEGA_COVERAGE host instrumentation and gcovr reporting

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Documentation

**Files:**
- Modify: `components/omega/doc/devGuide/Testing.md`
- Modify: `components/omega/doc/devGuide/CMakeBuild.md`
- Modify: `components/omega/doc/userGuide/OmegaBuild.md`

**Interfaces:** none (docs only). Content must match the exact option names, script names, and behaviors shipped in Tasks 1–4.

- [ ] **Step 1: Document running the analyses in `doc/devGuide/Testing.md`**

Append a new section to `components/omega/doc/devGuide/Testing.md`:

```markdown
## Memory-leak and coverage analysis (standalone)

Two opt-in CMake options add analysis to the standalone CTest suite. Both
default OFF, are standalone-only, and are independent.

### Memory-leak checking — `-DOMEGA_MEMCHECK=ON`

Each test rank runs under an auto-selected leak checker (valgrind on CPU;
`compute-sanitizer` on CUDA). A leak makes the test fail (`--error-exitcode=1`),
so leaks appear as failed tests in a normal run. Configure, build, then:

    ./omega_memcheck.sh                 # whole suite (compute node)
    ./omega_memcheck.sh -L memcheck     # only memcheck-labelled tests
    ./omega_memcheck.sh -R HALO_TEST    # one test

valgrind is 10–30× slower, so tests get a 3600 s timeout under memcheck. Benign
third-party init leaks are masked by `test/omega.supp`; regenerate/extend it by
adding `--gen-suppressions=all` to the launcher. A `MEMCHECK_PROBE_TEST` (a
deliberate leak, `WILL_FAIL`) is registered only under memcheck to prove the
checker actually detects leaks.

CDash note: because tests launch through MPI, the checker is embedded per rank
rather than via CTest's `MEMORYCHECK_COMMAND` (which would wrap `srun`). Leaks
therefore appear on CDash as failed tests, not as a Dynamic Analysis widget.

### Coverage — `-DOMEGA_COVERAGE=ON`

Instruments Omega host code (`--coverage -O0 -g`; third-party deps excluded),
selects the gcov front-end per compiler (`gcov` for GNU; `llvm-cov gcov` for
`oneapi-ifx`/Clang), and builds a gcovr HTML+text report. Requires `gcovr`
(in `dev-conda.txt`). Recommended with `-DOMEGA_BUILD_TYPE=Debug`.

    ./omega_coverage.sh                 # run suite, then write coverage/
    # -> coverage/omega_coverage.html, coverage/omega_coverage.txt

Coverage is host-only: device (CUDA/HIP/SYCL) kernels are not measured.

Multi-rank note: an N-rank test runs N processes that write the same `.gcda`
files. libgcov file-locks and merges per source, which is correct for ranks on
one node. For cross-node robustness, export a per-rank prefix before the run:

    export GCOV_PREFIX=$PWD/coverage/gcda/rank_${SLURM_PROCID:-0}
    export GCOV_PREFIX_STRIP=0

then point gcovr at `coverage/gcda` in addition to the build tree.
```

- [ ] **Step 2: Add configure examples to `doc/devGuide/CMakeBuild.md`**

In `components/omega/doc/devGuide/CMakeBuild.md`, under the standalone build-commands section (around line 117/151), add:

```markdown
### Analysis builds (Chrysalis examples)

Memory-leak checking with GNU:

    cmake -DOMEGA_BUILD_TYPE=Debug -DOMEGA_CIME_COMPILER=gnu \
          -DOMEGA_CIME_MACHINE=chrysalis -DOMEGA_PARMETIS_ROOT=${PARMETIS_ROOT} \
          -DOMEGA_BUILD_TEST=ON -DOMEGA_MEMCHECK=ON -Wno-dev \
          -S <omega_branch>/components/omega -B .

Coverage with oneAPI (`oneapi-ifx`, uses `llvm-cov gcov` automatically):

    cmake -DOMEGA_BUILD_TYPE=Debug -DOMEGA_CIME_COMPILER=oneapi-ifx \
          -DOMEGA_CIME_MACHINE=chrysalis -DOMEGA_PARMETIS_ROOT=${PARMETIS_ROOT} \
          -DOMEGA_BUILD_TEST=ON -DOMEGA_COVERAGE=ON -Wno-dev \
          -S <omega_branch>/components/omega -B .

`OMEGA_MEMCHECK` and `OMEGA_COVERAGE` are independent and both default OFF; a
missing tool (valgrind / gcovr / llvm-cov) yields a configure WARNING and the
analysis is skipped, never a hard failure.
```

- [ ] **Step 3: Document the generated scripts in `doc/userGuide/OmegaBuild.md`**

In `components/omega/doc/userGuide/OmegaBuild.md`, in the generated-scripts list (around lines 76–79), add entries and a short note:

```markdown
- `omega_memcheck.sh` — run the CTest suite under the per-rank memory-leak
  checker (present only when configured with `-DOMEGA_MEMCHECK=ON`).
- `omega_coverage.sh` — run the CTest suite and generate the gcovr coverage
  report under `coverage/` (present only when configured with
  `-DOMEGA_COVERAGE=ON` and `gcovr` is available).

Coverage is host-only and needs `gcovr` (see `dev-conda.txt`). Under memory
checking, leaks surface as failed tests (not a CDash Dynamic Analysis widget)
because tests launch through MPI.
```

- [ ] **Step 4: Verify docs build (if the docs toolchain is available)**

```bash
cd /lcrc/group/e3sm/ac.kimy/repos/github/Omega.Andrew/components/omega/doc
make html 2>&1 | tail -5 || echo "docs build skipped (sphinx not in env)"
```

Expected: docs build cleanly, or a clear "skipped" message if sphinx isn't installed (acceptable — content is plain Markdown).

- [ ] **Step 5: Commit**

```bash
cd /lcrc/group/e3sm/ac.kimy/repos/github/Omega.Andrew
git add components/omega/doc/devGuide/Testing.md components/omega/doc/devGuide/CMakeBuild.md \
        components/omega/doc/userGuide/OmegaBuild.md
git commit -m "$(cat <<'EOF'
Document OMEGA_MEMCHECK and OMEGA_COVERAGE analysis builds

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Chrysalis validation (gnu + oneapi-ifx, SERIAL + OPENMP)

**Files:** none created; may edit `components/omega/test/omega.supp` if suppression globs need tuning, and fold any fixes back into earlier tasks' files.

**Interfaces:** none. This task exercises the shipped features on the real machine and hardens the suppressions.

**Prereqs:** `gcovr` installed (activate the omega_dev conda env, or `pip install gcovr`), submodules initialized, `PARMETIS_ROOT` exported, an interactive/batch compute node for `srun`.

- [ ] **Step 1: Baseline — default OFF build is byte-identical for both compilers**

For `CC in gnu oneapi-ifx`, configure with both options OFF and confirm no analysis wrapping/flags:

```bash
for CC in gnu oneapi-ifx; do
  rm -rf /tmp/omega_off_$CC && mkdir -p /tmp/omega_off_$CC
  cmake -DOMEGA_CIME_COMPILER=$CC -DOMEGA_CIME_MACHINE=chrysalis \
        -DOMEGA_PARMETIS_ROOT=${PARMETIS_ROOT} -DOMEGA_BUILD_TEST=ON -Wno-dev \
        -S components/omega -B /tmp/omega_off_$CC >/dev/null 2>&1
  echo "$CC valgrind refs: $(cd /tmp/omega_off_$CC && ctest -N -V 2>/dev/null | grep -c valgrind)"
  echo "$CC coverage flags: $(cd /tmp/omega_off_$CC && cmake --build . --target OmegaLib -- -n 2>/dev/null | grep -c -- --coverage)"
done
```

Expected: all four counts `0`.

- [ ] **Step 2: Memcheck — build, run suite, confirm probe passes and clean tests pass (both compilers, SERIAL)**

```bash
for CC in gnu oneapi-ifx; do
  rm -rf /tmp/omega_mc_$CC && mkdir -p /tmp/omega_mc_$CC
  cmake -DOMEGA_BUILD_TYPE=Debug -DOMEGA_CIME_COMPILER=$CC -DOMEGA_CIME_MACHINE=chrysalis \
        -DOMEGA_PARMETIS_ROOT=${PARMETIS_ROOT} -DOMEGA_BUILD_TEST=ON -DOMEGA_MEMCHECK=ON -Wno-dev \
        -S components/omega -B /tmp/omega_mc_$CC >/dev/null 2>&1
  ( cd /tmp/omega_mc_$CC && ./omega_build.sh >build.log 2>&1 && \
    ./omega_memcheck.sh -R "MEMCHECK_PROBE_TEST|DATA_TYPES_TEST|HALO_TEST" 2>&1 | tail -15 )
done
```

Expected (each compiler): `MEMCHECK_PROBE_TEST` **Passed** (leak detected), `DATA_TYPES_TEST` and `HALO_TEST` **Passed** (no false-positive leaks). If a clean test fails on a benign third-party leak, capture it with `--gen-suppressions=all` and add a block to `test/omega.supp`, then re-run.

- [ ] **Step 3: Memcheck — OPENMP arch smoke (gnu)**

If an OpenMP toolchain is available, repeat Step 2 for a threaded build (the CIME case selects OpenMP), confirming the `--fair-sched=yes` + `KMP_BLOCKTIME=0`/`OMP_WAIT_POLICY=passive` env keep valgrind from stalling. Confirm `HALO_TEST` completes within the 3600 s timeout.

Expected: OPENMP tests complete under valgrind without hanging; probe still passes.

- [ ] **Step 4: Coverage — gnu (`gcov`) produces nonzero src coverage**

```bash
rm -rf /tmp/omega_cov_gnu && mkdir -p /tmp/omega_cov_gnu
cmake -DOMEGA_BUILD_TYPE=Debug -DOMEGA_CIME_COMPILER=gnu -DOMEGA_CIME_MACHINE=chrysalis \
      -DOMEGA_PARMETIS_ROOT=${PARMETIS_ROOT} -DOMEGA_BUILD_TEST=ON -DOMEGA_COVERAGE=ON -Wno-dev \
      -S components/omega -B /tmp/omega_cov_gnu >/dev/null 2>&1
( cd /tmp/omega_cov_gnu && ./omega_build.sh >build.log 2>&1 && \
  ./omega_coverage.sh -R "DATA_TYPES_TEST|DECOMP_NTASK1_TEST" 2>&1 | tail -10 )
tail -5 /tmp/omega_cov_gnu/coverage/omega_coverage.txt
```

Expected: gcovr summary reports nonzero line coverage; `coverage/omega_coverage.html` exists.

- [ ] **Step 5: Coverage — oneapi-ifx (`llvm-cov gcov`) produces nonzero src coverage**

Repeat Step 4 with `-DOMEGA_CIME_COMPILER=oneapi-ifx`. Confirm the generated `gcov` wrapper's second line is `exec llvm-cov gcov "$@"` and that gcovr still reports nonzero coverage.

```bash
grep -H "exec" /tmp/omega_cov_oneapi-ifx/gcov
```

Expected: `exec llvm-cov gcov "$@"`; nonzero coverage in the report. If CDash `ctest -T coverage` shows empty for oneapi-ifx (a known llvm-cov quirk), the gcovr report remains the source of truth — note it; the `gcov` wrapper already normalizes the basename for CTest.

- [ ] **Step 6: If multi-rank coverage counts look wrong, apply per-rank `GCOV_PREFIX`**

Run a coverage report over an 8-rank test (e.g. `HALO_TEST`). If gcovr errors on corrupt `.gcda` or counts look implausible, re-run with the documented per-rank prefix and confirm the report is sane:

```bash
cd /tmp/omega_cov_gnu
GCOV_PREFIX=$PWD/coverage/gcda/rank GCOV_PREFIX_STRIP=0 ./omega_ctest.sh -R HALO_TEST
gcovr --root $(git -C /lcrc/group/e3sm/ac.kimy/repos/github/Omega.Andrew rev-parse --show-toplevel)/components/omega \
      --filter '.*/src/' --gcov-executable ./gcov coverage/gcda . --print-summary
```

Expected: a sane merged summary. Record the outcome in `doc/devGuide/Testing.md` if the prefix approach is needed by default.

- [ ] **Step 7: Fold fixes back and final commit**

If Steps 2–6 required changes (suppression blocks, doc clarifications), commit them:

```bash
cd /lcrc/group/e3sm/ac.kimy/repos/github/Omega.Andrew
git add -A components/omega
git commit -m "$(cat <<'EOF'
Tune Omega memcheck suppressions and docs from Chrysalis validation

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

> After this task, the branch is ready. Integrating/pushing to the `grnydawn` fork is a separate, user-confirmed step (use superpowers:finishing-a-development-branch) — do not push here.

---

## Post-plan notes

- **Deferred / accepted trade-offs (from design validation):** CDash gets leaks as failed tests, not a Dynamic Analysis widget (MPI per-rank embed); coverage is host-only; multi-rank `.gcda` correctness relies on libgcov locking with a documented `GCOV_PREFIX` fallback; GPU (CUDA/HIP/SYCL) rows are wired-but-manual.
- **Genericity:** adding a machine/compiler is a one-row edit in `omega_capability_map()`; everything downstream reads from it.

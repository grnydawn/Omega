# Omega — Project Context

Omega (Ocean Model for E3SM Global Applications) is the next-generation ocean component planned for v4 of the **E3SM** (Energy Exascale Earth System Model). It is a modern, performance-portable C++ rewrite of the earlier MPAS-Ocean model, redesigned for GPU/CPU portability via Kokkos and intended to run both standalone and as a coupled E3SM component. Source lives under `components/omega/` (README: `components/omega/README.md`). This repo is the `grnydawn/Omega` fork of E3SM; active work is on branch `ykim/omega/coverage` (test-coverage). The Omega component has its own CMake build and `ctest` suite, **separate from the CIME case-control system** used by the rest of E3SM.

## Stack

- **Language:** C++17 (`set(CMAKE_CXX_STANDARD 17)` in `components/omega/CMakeLists.txt`; some Fortran/C interop and Python for tooling/docs).
- **On-node parallelism:** **Kokkos** (CUDA / HIP / SYCL / OpenMP / Serial backends, selected by `OMEGA_ARCH`). Kokkos is sourced from the E3SM external `externals/ekat/extern/kokkos`. (Note: YAKL is an E3SM submodule required by the build but Omega's own kernels use Kokkos, not YAKL.)
- **Distributed parallelism:** MPI.
- **I/O:** SCORPIO / PIO (`pioc`) from `externals/scorpio`; NetCDF / PNetCDF underneath.
- **Mesh partitioning:** ParMETIS / METIS / GKlib (located via `FindParmetis.cmake`, `OMEGA_PARMETIS_ROOT`).
- **Vendored libs** (in `components/omega/external/`): `spdlog` (logging), `yaml-cpp` (config), `cpptrace` (stack traces), `GSW-C` (TEOS-10 equation of state).
- **Timing/util:** E3SM `share/timing` (gptl), `share/pacer`, `share/util_cxx`.
- **Build system:** CMake ≥ 3.21 (HIP support). Entry point `components/omega/CMakeLists.txt` includes `OmegaBuild.cmake`, which drives a 4-step build (Setup → Update → Build → Output) and auto-detects standalone vs. E3SM-embedded mode.

## Repository Layout

All paths under `components/omega/`:

- `src/` — model source, grouped into:
  - `base/` — lowest-level infra: `DataTypes.h` (Real/Array Kokkos aliases), `Decomp`, `Halo`, `MachEnv`, `Broadcast`, `IO`, `Reductions`, `TriDiagSolvers`.
  - `infra/` — shared infra: `Config` (YAML), `TimeMgr`, `Field`/`Dimension`/`Metadata`, `IOStream`, `Logging`, `Error`, and the Kokkos wrappers `OmegaKokkos*.h`.
  - `ocn/` — ocean physics/numerics: tendencies, operators, EOS, vertical coord/advection/mixing, tracers, state.
  - `timeStepping/` — time integrators.
  - `drivers/` — standalone and coupled drivers (`drivers/standalone/OceanDriver.cpp`).
  - `analysis/` — ocean analyses.
- `test/` — unit tests mirroring `src/` (`base/`, `infra/`, `ocn/`, `timeStepping/`, `drivers/`); registered in `test/CMakeLists.txt`.
- `doc/` — Sphinx docs split into `design/`, `devGuide/`, `userGuide/`. Key dev pages: `devGuide/CMakeBuild.md`, `devGuide/QuickStart.md`, `devGuide/Testing.md`, `devGuide/Linting.md`, `devGuide/ParallelLoops.md`, `devGuide/Error.md`.
- `configs/Default.yml` — default runtime config.
- `external/` — vendored third-party libs (spdlog, yaml-cpp, cpptrace, GSW-C).
- Build/config files: `CMakeLists.txt`, `OmegaBuild.cmake`, `FindParmetis.cmake`, `CTestScript.cmake`, `.clang-format`, `dev-conda.txt`.

## How to Build & Test

Standalone build/test (from `doc/devGuide/CMakeBuild.md` and `QuickStart.md`). There is **no checked-in build shell script**; CMake *generates* `omega_build.sh` and `omega_ctest.sh` into the build directory.

1. Initialize the required submodules (from repo root):
   ```sh
   git submodule update --init --recursive \
       externals/YAKL externals/ekat externals/scorpio \
       components/omega/external cime
   ```
2. Configure in an out-of-source build dir (`$BUILD_DIR`), pointing at ParMETIS:
   ```sh
   cmake \
     -DOMEGA_BUILD_TYPE=<Debug|Release> \
     -DOMEGA_CIME_COMPILER=<compiler> \
     -DOMEGA_CIME_MACHINE=<machine> \
     -DOMEGA_PARMETIS_ROOT=${PARMETIS_ROOT} \
     -DOMEGA_BUILD_TEST=ON \
     -Wno-dev \
     -S <repo>/components/omega \
     -B .
   ```
   `<machine>`/`<compiler>` must be E3SM-supported on that host (the build pulls compiler/flags from CIME). Note: `<machine>` is a required input — a fully unattended build needs a supported machine name; on an unsupported host this configure step will not succeed.
3. Build: `./omega_build.sh`
4. Run unit tests (CTest; tests run on a compute node, many are MPI tests with `-n 8`):
   ```sh
   ./omega_ctest.sh
   ```
   Some tests need mesh files linked into `test/` as `OmegaMesh.nc`, `OmegaSphereMesh.nc`, `OmegaPlanarMesh.nc` (download URLs in `QuickStart.md`). CTest failures are logged to `$BUILD_DIR/Testing/Temporary/LastTest.log`.

Notes:
- The Polaris `omega_pr` regression suite (`doc/devGuide/Testing.md`) is the *additional* full-physics regression gate expected for non-trivial PRs, but it requires the Polaris framework and HPC scheduler — CTest is the in-repo gate the pipeline can run directly.
- Tests are added in `test/CMakeLists.txt` via the `add_omega_test(<TEST_NAME> <exe> <source.cpp> "<mpi_args>")` helper. There are ~40 registered CTest tests (e.g. `DATA_TYPES_TEST`, `DECOMP_NTASK8_TEST`, `HALO_TEST`, `EOS_TEST`, `TIMESTEPPER_TEST`). `ERROR_TEST` is intentionally marked `WILL_FAIL`.

## Conventions

- **C++ standard:** C++17.
- **Formatting:** `clang-format` with **LLVM base style, 3-space indent** (`components/omega/.clang-format`); aligned consecutive assignments/macros/comments. LLVM coding style is required. Enforced via **pre-commit** (`dev-conda.txt` provides the `omega_dev` conda env; `pre-commit install`). Run `pre-commit run --all-files` or `pre-commit run clang-format`. CI lints modified files on every PR. clang-tidy/cppcheck/IWYU are configured but currently disabled.
- **Naming:** `UpperCamelCase` for types and variables/members (e.g. `NCells`, `Team`, `MyError`); macros are `UPPER_SNAKE`. Namespace `OMEGA`. Header guards `OMEGA_<NAME>_H`.
- **Data types:** use the aliases in `base/DataTypes.h` — `I4`/`I8`, `R4`/`R8`, generic `Real` (double by default, float under `OMEGA_SINGLE_PRECISION`), and Kokkos array aliases `Array1DReal … Array5DReal`, `ArrayND<T>`, plus `Host*` and scratch variants. Do not use raw Kokkos `View` declarations directly.
- **Kokkos patterns:** use Omega's loop wrappers, not raw Kokkos policies (`doc/devGuide/ParallelLoops.md`): `parallelFor` / `parallelReduce` for flat multi-dim loops; hierarchical `parallelForOuter`/`parallelForInner` (+ `parallelReduce/Scan/SearchInner`) with `KOKKOS_LAMBDA` (outer) and the `INNER_LAMBDA` macro (inner); `TeamMember Team`, `teamBarrier`, `Kokkos::single`, `LaunchConfig`/`TeamScratch` for scratch memory. Respect `OMEGA_MEMORY_LAYOUT` (default RIGHT).
- **Error handling / logging:** use the `Error.h` facility (`doc/devGuide/Error.md`) — `ABORT_ERROR(msg, args…)`, `OMEGA_ASSERT`/`OMEGA_REQUIRE`, the `Error` class with `ErrorCode` enum (`Success`/`Warn`/`Fail`/`Critical`), accumulation via `+`/`+=`, and `RETURN_ERROR` / `CHECK_ERROR[_WARN|_ABORT]`. It builds on the spdlog-based **Logging** facility (`Logging.h`, fmt-style `{}` placeholders) and emits cpptrace stack traces in debug builds. Prefer the macros over calling `Error::abort()` directly.

## Success Criteria for Tickets

A build ticket on Omega is "done" when:
1. **Compiles** in the standalone Omega CMake build (`./omega_build.sh`) with `-DOMEGA_BUILD_TEST=ON`, in the relevant build type (Debug and/or Release) on a supported machine/compiler.
2. **All existing CTest tests pass** (`./omega_ctest.sh`) — no newly failing or skipped tests; `ERROR_TEST` remains the only intended `WILL_FAIL`.
3. **New functionality has a unit test** under `components/omega/test/` (mirroring the `src/` subdir) and registered via `add_omega_test(...)` in `test/CMakeLists.txt`.
4. **Passes clang-format / pre-commit** (`pre-commit run --all-files`) and follows the LLVM style and Omega naming/data-type/Kokkos/error conventions above.
5. **No regressions** — existing behavior unchanged; uses Omega data-type aliases and loop/error wrappers rather than raw Kokkos/MPI calls. For non-trivial / physics-affecting changes, the Polaris `omega_pr` regression suite is the upstream gate (may require HPC and is documented in `doc/devGuide/Testing.md`); the in-repo CTest suite is the gate the autonomous pipeline runs directly.

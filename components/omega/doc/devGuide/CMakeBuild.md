(omega-dev-cmake-build)=

# Omega Build with CMake

The Omega build system utilizes CMake, a widely-used build tool,
to facilitate the build process.

The build process is defined in the CMakeLists.txt file located
in the top-level directory of Omega. It consists of six consecutive
phases: Setup, Toolchain, Update, Validate, Build, and Output.

```{note}
This was originally documented as four steps (Setup, Update, Build, Output).
Two phases were added because the original four could not describe what the
code actually had to do. `project()` must be called only after the compilers
are final, and that ordering constraint is also the only axis on which the two
build modes differ, so committing the toolchain is now its own phase. And the
"integrity of the build setup" that Step 2 promised to verify was never
actually checked, so validation is now a phase of its own with a defined place
in the order, rather than a single line at the end of Update.
```

The phases are enforced, not merely described. `OmegaBuild.cmake` keeps the
current phase in a global property, every phase macro opens with
`omega_require_phase()`, and `omega_phase_produces()` asserts that a phase
really produced the variables later phases depend on. Calling a macro from the
wrong place, or letting a required variable go unset, is a configure-time error
that names the macro and the phase, instead of a build that succeeds and is
wrong.

The build phase consists of adding three subdirectories that drive builds
for external libraries, the Omega model, and optional tests.

Python is required to use this build system.
The version of CMake should be 3.21 or later for supporting HIP.

## Phase 1: Setup

During this step, the build-controlling variables are configured.
The Omega build system supports two modes: standalone and E3SM
component. The build mode is automatically detected, and any
mode-specific differences are resolved to ensure a consistent
build process.

By default, Omega assumes it is part of the E3SM code distribution.
In both build modes, the build system collects build-controlling
parameters, such as compiler paths and flags, from the E3SM build system.

There are three types of Omega build-controlling variables: Omega,
E3SM, and CMake. The names of these variables start with "OMEGA_",
"E3SM_", and "CMAKE", respectively. Temporary variables are prefixed
with an underscore ("_").

The following is a list of Omega-specific variables available in
this version:

```
OMEGA_PROJECT_NAME: Name of the project ("OmegaOceanModel")
OMEGA_EXE_NAME: Name of the executable ("omega.exe")
OMEGA_LIB_NAME: Name of the library ("OmegaLib")
OMEGA_BUILD_MODES: List of build modes ("E3SM", "STANDALONE", "NOT_DEFINED")
OMEGA_BUILD_MODE: Selected build mode
OMEGA_BUILD_DIR: Omega top-level build directory
OMEGA_SOURCE_DIR: Directory where the top-level Omega CMakeLists.txt is located
OMEGA_DEFAULT_BUILD_TYPE: Default build type ("Release")
OMEGA_INSTALL_PREFIX: User-defined output directory for the library and executable
OMEGA_ARCH: Target architecture. One of "CUDA", "HIP", "SYCL", "OPENMP", "THREADS", "SERIAL", or "" to detect it. The value is upper-cased and validated, so an unknown name is an error rather than a silent fallback to SERIAL.
OMEGA_CXX_COMPILER: C++ compiler
OMEGA_C_COMPILER: C compiler
OMEGA_Fortran_COMPILER: Fortran compiler
OMEGA_CIME_COMPILER: E3SM compiler name defined in config_machines.xml
OMEGA_CIME_MACHINE: E3SM machine name defined in config_machines.xml
OMEGA_CIME_PROJECT: Slurm account passed to CIME during Omega build
OMEGA_CXX_FLAGS: a list for C++ compiler flags
OMEGA_LINK_OPTIONS: a list for linker flags
OMEGA_BUILD_EXECUTABLE: Enable building the Omega executable
OMEGA_BUILD_TEST: Enable building Omega tests
OMEGA_PARMETIS_ROOT: Parmetis installtion directory
OMEGA_METIS_ROOT: Metis installtion directory
OMEGA_GKLIB_ROOT: GKlib installtion directory
OMEGA_HIP_COMPILER: HIP compiler (e.g., hipcc)
OMEGA_HIP_FLAGS: HIP compiler flags
OMEGA_MEMORY_LAYOUT: Kokkos memory layout ("LEFT" or "RIGHT"). "RIGHT" is a default value.
OMEGA_TILE_LENGTH: a length of one "side" of a Kokkos tile. Unset by default, in which case OMEGA_TILE_LENGTH is not defined for the compiler.
OMEGA_LOG_LEVEL: a default logging level, one of "TRACE", "DEBUG", "INFO", "WARN", "ERROR", "CRITICAL", "OFF". "INFO" is the default value.
OMEGA_LOG_FLUSH: turn on the unbuffered logging. "OFF" is a default value.
OMEGA_VECTOR_LENGTH: Vector length used for blocking inner loops for vectorization. "1" is a default value.
```

E3SM-specific variables

```
E3SM_SOURCE_DIR: E3SM component directory (${E3SM_ROOT}/components)
E3SM_CIME_ROOT: CIME root directory
E3SM_CIMECONFIG_ROOT: E3SM CIME config directory
E3SM_EXTERNALS_ROOT: E3SM externals directory
E3SM_DEFAULT_BUILD_TYPE: E3SM build type (Release or Debug)
```

CMake variables

```
CMAKE_CURRENT_SOURCE_DIR
CMAKE_CURRENT_BINARY_DIR
CMAKE_CURRENT_LIST_DIR
CMAKE_CXX_STANDARD
CMAKE_CXX_COMPILER
CMAKE_CXX_FLAGS
CMAKE_CURRENT_LIST_DIR
CMAKE_BUILD_TYPE
CMAKE_INSTALL_PREFIX
CMAKE_VERSION
```

## Phase 2: Toolchain

This phase commits the compilers and flags that Setup discovered, and then
calls `project()`. It exists because CMake requires the toolchain to be final
before `project()` runs, which is why compiler selection cannot simply live in
Update alongside the other derived variables.

It is a no-op in E3SM mode: the parent build
(`components/CMakeLists.txt`) has already selected the compilers and called
`project()`, and Omega adopts what it chose.

## Phase 3: Update

In this phase, CMake is configured, and external library variables,
such as Kokkos, MPI, NetCDF, and PNetCDF, are set based on the settings
defined in the Setup phase.

## Phase 4: Validate

`omega_check_setup()` asserts that everything the first three phases
produced is self-consistent, before anything is built. It checks, among other things, that
`OMEGA_ARCH` is a known architecture and agrees with `OMEGA_TARGET_DEVICE` and
with the Kokkos backend that is actually enabled; that `OMEGA_BUILD_TYPE` is
recognized and that `OMEGA_DEBUG` agrees with it; that ParMETIS and METIS were
found; and that the enumerated and numeric options hold values Omega can use.

All failures are collected and reported together, so a misconfigured build
lists every problem in one pass instead of one per re-run.

Because this runs before `add_subdirectory(external)`, it can only assert about
variables. Assertions about *targets* belong in
`omega_check_dependencies()`, which runs in the Build phase as soon as the
externals have been added, and verifies
that every library `src/CMakeLists.txt` links by bare name is a real CMake
target rather than a raw `-l` flag.

## Phase 5: Build

During this phase, the build process is configured. It includes building
external libraries, followed by building the Omega main model from source
files. Optionally, tests can also be built.

## Phase 6: Output

The final phase, which is optional, involves copying a subset of the build
artifacts to designated locations or generating dynamic outputs as needed.
This is where the standalone developer scripts (`omega_env.sh`,
`omega_build.sh`, `omega_run.sh`, `omega_ctest.sh`, `omega_profile.sh`) and the
configuration file copies are generated. They are emitted last so that every
value they embed - the architecture, the build type, whether the build targets
a device - is final by the time it is written.

## Omega in a coupled case

E3SM calls `build_omega()` unconditionally from `components/CMakeLists.txt`,
unlike `build_mpas_models()`, which only runs for the MPAS cores the case
actually contains. Omega's top-level `CMakeLists.txt` therefore returns
immediately when `COMP_OCN` is set to anything other than `omega`.

Without that guard, a case with a different ocean - MPAS-Ocean, a data ocean,
or a stub ocean - configures the whole of Omega, including its own spdlog,
yaml-cpp, Kokkos and SCORPIO, and then fails outright:

```
add_library cannot create target "ocn" because another target with the same
name already exists.  The existing target is a static library created in
source directory ".../components/omega/src".
```

`src/CMakeLists.txt` creates an `ocn` library whenever `OMEGA_BUILD_MODE` is
`E3SM`, and so does the real ocean component -
`mpas-framework/src/build_core.cmake` for MPAS-Ocean,
`components/cmake/build_model.cmake` for a data or stub ocean.

```{note}
The guard in Omega's `CMakeLists.txt` is the fix that can live inside Omega.
A cleaner fix is still wanted **upstream**: give `build_omega()` in
`components/cmake/build_omega.cmake` the same participation guard that
`build_mpas_models()` has, so that E3SM does not descend into Omega at all for
cases that do not use it. That would also stop these cases from configuring
Omega's external libraries needlessly. It is deliberately not made here because
it changes E3SM code outside the Omega component.
```

## Standalone Build Commands

In the Omega branch you would like to build, first update the submodules that
Omega requires:

```sh
git submodule update --init --recursive \
	externals/YAKL \
	externals/ekat \
	externals/scorpio \
	components/omega/external \
	cime
```

Since some systems require tests to be run in a scratch space, it is a good
idea to build the code somewhere in your scratch space. We will refer to the
build directory as `$BUILD_DIR`:

```sh
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
```

Set `$PARMETIS_ROOT` to the location for Metis and Parmetis libraries built
for your machine and compiler:

```sh
export PARMETIS_ROOT=<parmetis_root>
```

Run CMake for the build type, machine, and compiler you want:

```sh
cmake \
   -DOMEGA_BUILD_TYPE=<build_type> \
   -DOMEGA_CIME_COMPILER=<compiler> \
   -DOMEGA_CIME_MACHINE=<machine> \
   -DOMEGA_PARMETIS_ROOT=${PARMETIS_ROOT} \
   -DOMEGA_BUILD_TEST=ON \
   -Wno-dev \
   -S <omega_branch>/components/omega \
   -B .
```

Where `<build_type>` is either `Debug` or `Release`, `<omega_branch>` is the
path to the base of the Omega branch you want to build, and `<machine>` and
`<compiler>` are supported by E3SM on that machine.

The command above configures Omega to build CTests
(`-DOMEGA_BUILD_TEST=ON`), which is recommended.

If CMake configuration runs correctly, you should have an `omega_build.sh`
script that you can run to build Omega:

```sh
./omega_build.sh
```

When `OMEGA_BUILD_MODE` is `E3SM`, `src/CMakeLists.txt` additionally
builds an `ocn` library from the sources in `src/drivers/coupled/`
(the Fortran/C++ bridge and MCT cap), linked against `OmegaLib` and
`csm_share`. This is the library CIME builds and links into the E3SM
executable when Omega is used as the ocean component of a coupled case.

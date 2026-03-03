# OMEGA CMake Build System Refactoring

This document describes the OMEGA CMake build system architecture after refactoring. The guiding principle is to **strictly follow E3SM's build system** while maintaining a clean and minimal CMake structure.

## Table of Contents
1. [Design Philosophy](#design-philosophy)
2. [Current File Structure](#current-file-structure)
3. [Build Process Overview](#build-process-overview)
4. [E3SM Integration Strategy](#e3sm-integration-strategy)
5. [Key Components](#key-components)
6. [Macros vs Functions](#macros-vs-functions)
7. [Configuration Options](#configuration-options)
8. [Future Work](#future-work)

---

## Design Philosophy

### Core Principles

1. **Strict E3SM Build System Alignment**: Omega is a component of E3SM. Rather than reinventing build configurations, Omega leverages E3SM's mature build system wherever possible.

2. **E3SM Case as Configuration Source**: Create a simple E3SM case in the Omega build directory and use its output files for:
   - Environment settings (compilers, flags, modules)
   - MPI/job launch configurations
   - Machine-specific settings

3. **GLOB_RECURSE for Source Files**: Use `file(GLOB_RECURSE ... CONFIGURE_DEPENDS ...)` to automatically pick up new source files. This reduces maintenance overhead—developers can add new `.cpp` files without manually updating CMakeLists.txt.

4. **Minimal CMake File Splitting**: Split only when a clear separation of concerns justifies it.

5. **Modern CMake Patterns**: Use target-based commands (`target_compile_definitions()`, `add_compile_options()`) instead of deprecated global commands.

---

## Current File Structure

```
omega/
├── CMakeLists.txt              # Top-level: 4-step build orchestration
├── cmake/
│   ├── OmegaBuild.cmake        # Main build macros and variables (~400 lines)
│   ├── OmegaE3SMCase.cmake     # E3SM case creation and configuration (~370 lines)
│   ├── OmegaScripts.cmake      # Template-based script generation (~120 lines)
│   ├── FindParmetis.cmake      # Modern find module with imported targets (~160 lines)
│   ├── CTestScript.cmake       # CDash integration script
│   └── templates/
│       ├── omega_env.sh.in     # Environment setup template
│       ├── omega_build.sh.in   # Build script template
│       ├── omega_run.sh.in     # Run script template
│       ├── omega_ctest.sh.in   # CTest runner template
│       └── omega_profile.sh.in # Profiler setup template
├── src/
│   └── CMakeLists.txt          # Library and executable building
├── test/
│   └── CMakeLists.txt          # Data-driven test registration (~200 lines)
└── external/
    └── CMakeLists.txt          # External dependencies
```

### Key Improvements Over Original

| Aspect | Before | After |
|--------|--------|-------|
| OmegaBuild.cmake | 741 lines monolithic | Split into 3 focused modules |
| test/CMakeLists.txt | 496 lines repetitive | ~200 lines data-driven |
| Script generation | 40+ file(APPEND) calls | Template-based with configure_file() |
| add_definitions() | 12 global calls | target_compile_definitions() |
| CMAKE_CXX_FLAGS | Direct manipulation | add_compile_options() |
| FindParmetis.cmake | Basic variables | Modern imported targets |
| Error handling | Minimal | Comprehensive with clear messages |

---

## Build Process Overview

The build follows a **4-step process** defined in `CMakeLists.txt`:

### Step 1: Setup
```cmake
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/OmegaBuild.cmake)
common()

if (NOT DEFINED PROJECT_NAME)
  # Standalone build
  init_standalone_build()
  project(${OMEGA_PROJECT_NAME} VERSION ${OMEGA_VERSION} LANGUAGES C CXX)
  setup_standalone_build()
else()
  # E3SM embedded build
  setup_e3sm_build()
endif()
```

### Step 2: Update
```cmake
update_variables()  # Set CMake/Kokkos variables
check_setup()       # Verify configuration integrity
```

### Step 3: Build
```cmake
add_subdirectory(external)
add_subdirectory(src)
if(OMEGA_BUILD_TEST)
  add_subdirectory(test)
endif()
```

### Step 4: Output
```cmake
wrap_outputs()              # Installation rules
organize_omega_options()    # Cache variable organization
```

---

## E3SM Integration Strategy

### Overview

Omega creates a minimal E3SM case and extracts build settings from its output files:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Omega Build Process                             │
├─────────────────────────────────────────────────────────────────────┤
│  1. User runs cmake with E3SM source directory                      │
│                           ↓                                         │
│  2. Omega creates E3SM case in build/e3smcase                       │
│     (create_newcase --compset CMPASO-NYF --res T62_oQU120)          │
│                           ↓                                         │
│  3. case.setup generates environment files                          │
│     (.env_mach_specific.sh, .case.run.sh, Macros.cmake)             │
│                           ↓                                         │
│  4. Omega CMake extracts settings:                                  │
│     - Compiler paths and flags                                      │
│     - MPI configuration                                             │
│     - Architecture detection                                        │
│                           ↓                                         │
│  5. Build Omega using extracted E3SM settings                       │
└─────────────────────────────────────────────────────────────────────┘
```

### E3SM Case Files Used

| File | Purpose |
|------|---------|
| `.env_mach_specific.sh` | Environment variables, module loads, compiler paths |
| `.case.run.sh` | MPI launch command and arguments |
| `Macros.cmake` | CMake variables for compilers and flags |

---

## Key Components

### cmake/OmegaBuild.cmake

Main build configuration file containing:

- **Build control variables**: `OMEGA_PROJECT_NAME`, `OMEGA_BUILD_MODE`, `OMEGA_ARCH`
- **Core macros**:
  - `common()` - Initialize options and variables
  - `init_standalone_build()` - Configure standalone build from E3SM case
  - `setup_standalone_build()` / `setup_e3sm_build()` - Mode-specific setup
  - `update_variables()` - Configure Kokkos and compile definitions
  - `check_setup()` - Verify configuration integrity (function)
  - `wrap_outputs()` - Installation rules (function)
  - `organize_omega_options()` - Cache variable organization (function)

### cmake/OmegaE3SMCase.cmake

E3SM case handling with improved error checking:

- `run_bash_command()` - Execute bash with warning on failure
- `run_bash_command_required()` - Execute bash with fatal error on failure
- `read_cime_config()` - Create E3SM case and extract settings
- `detect_compilers_from_e3sm()` - Find C/C++/Fortran compilers
- `detect_omega_arch()` - Detect CUDA/HIP/SYCL/OpenMP/Serial
- `configure_cxx_compiler_for_arch()` - Architecture-specific compiler setup

### cmake/OmegaScripts.cmake

Template-based script generation:

- `generate_omega_scripts()` - Generate all helper scripts using templates
- `copy_omega_config_files()` - Copy YAML configs to build directory

### cmake/FindParmetis.cmake

Modern CMake find module with:

- **Imported targets**: `Parmetis::parmetis`, `Metis::metis`, `GKlib::gklib`
- **Backward-compatible targets**: `parmetis`, `metis`, `gklib`
- Proper dependency handling between libraries
- `find_package_handle_standard_args()` integration

### test/CMakeLists.txt

Data-driven test registration using helper macros:

```cmake
# Helper macros
omega_register_test(TEST_NAME exe_name source_file mpi_args)
omega_register_test_with_def(TEST_NAME exe_name source_file mpi_args DEFINITION)
omega_register_sp_test(TEST_NAME exe_name source_file mpi_args DEFINITION)

# Example usage
omega_register_test(DATA_TYPES_TEST testDataTypes.exe base/DataTypesTest.cpp "-n;1")
omega_register_test_with_def(HORZOPERATORS_PLANE_TEST testHorzOperatorsPlane.exe
                             ocn/HorzOperatorsTest.cpp "-n;8" HORZOPERATORS_TEST_PLANE)
```

---

## Macros vs Functions

The build system uses both macros and functions appropriately:

| Type | When to Use | Examples |
|------|-------------|----------|
| **Macro** | Must set variables in caller's scope | `common()`, `init_standalone_build()`, `update_variables()` |
| **Function** | Self-contained, no caller variables needed | `check_setup()`, `wrap_outputs()`, `generate_omega_scripts()` |

### Functions (scope-isolated)
- `check_setup()` - Only reads variables and prints messages
- `wrap_outputs()` - Only calls install()
- `generate_omega_scripts()` - Only generates files
- `copy_omega_config_files()` - Only copies files
- `organize_omega_options()` - Only marks options as advanced

### Macros (sets caller variables)
- `common()` - Sets `OMEGA_DEBUG`, `OMEGA_CXX_FLAGS`, etc.
- `init_standalone_build()` - Calls other macros that set variables
- `read_cime_config()` - Sets `OMEGA_MPI_EXEC`, environment variables
- `detect_omega_arch()` - Sets `OMEGA_ARCH`
- `configure_cxx_compiler_for_arch()` - Sets `CMAKE_CXX_COMPILER`

---

## Configuration Options

### Primary Options (visible in ccmake/cmake-gui)

| Option | Type | Description |
|--------|------|-------------|
| `OMEGA_BUILD_TYPE` | STRING | Build type (Release/Debug) |
| `OMEGA_BUILD_MODE` | STRING | Build mode (STANDALONE/E3SM) |
| `OMEGA_ARCH` | STRING | Architecture (CUDA/HIP/SYCL/OPENMP/SERIAL) |
| `OMEGA_BUILD_TEST` | BOOL | Build tests |
| `OMEGA_DEBUG` | BOOL | Enable debug mode |
| `OMEGA_PARMETIS_ROOT` | PATH | ParMETIS installation path |
| `OMEGA_INSTALL_PREFIX` | PATH | Installation prefix |

### Advanced Options (hidden by default)

| Option | Type | Description |
|--------|------|-------------|
| `OMEGA_TILE_LENGTH` | STRING | Kokkos tile length |
| `OMEGA_VECTOR_LENGTH` | STRING | Vector length for loop blocking |
| `OMEGA_MEMORY_LAYOUT` | STRING | Kokkos memory layout (LEFT/RIGHT) |
| `OMEGA_LOG_LEVEL` | STRING | Logging level |
| `OMEGA_LOG_TASKS` | STRING | Tasks that generate log files |
| `OMEGA_MPI_ON_DEVICE` | BOOL | Allow device buffers in MPI |
| `OMEGA_CUDA_MALLOC_ASYNC` | BOOL | CUDA async malloc support |

### Version Information

```cmake
OMEGA_VERSION_MAJOR = 0
OMEGA_VERSION_MINOR = 1
OMEGA_VERSION_PATCH = 0
OMEGA_VERSION = "0.1.0"
```

---

## Future Work

### Phase 2: Enhanced E3SM Integration

- [ ] Task 2.1: Improve architecture detection from E3SM case (GPU_TYPE, GPU_OFFLOAD)
- [ ] Task 2.2: Fix `setup_e3sm_build()` TODO for OMEGA_ARCH
- [ ] Task 2.3: Add CMakePresets.json for standardized configurations
- [ ] Task 2.4: Create omega-config.cmake for external package use

### Potential Improvements

1. **CMakePresets.json** - Standardized build configurations for common scenarios
2. **Package Export** - Allow `find_package(Omega)` from external projects
3. **Better E3SM Variable Mapping** - More comprehensive extraction from E3SM case

---

## Configuration Summary Output

When cmake configures successfully, it prints:

```
=== Omega Configuration Summary ===
  Build Mode:        STANDALONE
  Build Type:        Release
  Architecture:      CUDA
  Target Device:     TRUE
  CXX Compiler:      /path/to/nvcc_wrapper
  Debug Mode:        OFF
  Build Tests:       ON
  Build Executable:  ON
  MPI Exec:          srun
  E3SM Case:         /path/to/build/e3smcase
===================================

Omega 0.1.0 configuration complete.
```

---

## References

- [E3SM CIME Documentation](https://esmci.github.io/cime/versions/master/html/)
- [Modern CMake Best Practices](https://cliutils.gitlab.io/modern-cmake/)
- [CMake configure_file()](https://cmake.org/cmake/help/latest/command/configure_file.html)
- [CMake Imported Targets](https://cmake.org/cmake/help/latest/guide/importing-exporting/index.html)

# OMEGA CMake Build System Refactoring Plan

This document outlines the analysis and refactoring plan for the OMEGA CMake build system to make it cleaner, more maintainable, and ready for E3SM integration.

## Table of Contents
1. [Current State Analysis](#current-state-analysis)
2. [Identified Problems](#identified-problems)
3. [Phase 1: Clean Up Current Build System](#phase-1-clean-up-current-build-system)
4. [Phase 2: Prepare for E3SM Integration](#phase-2-prepare-for-e3sm-integration)
5. [Detailed File-by-File Analysis](#detailed-file-by-file-analysis)
6. [Code Examples](#code-examples)

---

## Current State Analysis

### File Structure and Line Counts

| File | Lines | Role | Status |
|------|-------|------|--------|
| `CMakeLists.txt` | 114 | Top-level orchestration | Good - clean 4-step approach |
| `cmake/OmegaBuild.cmake` | 741 | Macros, compiler detection, scripts | **Needs major refactoring** |
| `src/CMakeLists.txt` | 128 | Library/executable building | Minor issues |
| `test/CMakeLists.txt` | 496 | Test registration | **Highly repetitive** |
| `external/CMakeLists.txt` | 142 | External dependencies | Medium issues |
| `cmake/FindParmetis.cmake` | 111 | Parmetis discovery | Outdated but functional |

**Total: 1,621 lines**

### Current Strengths

1. **Clear top-level structure**: 4-step build process (Setup, Update, Build, Output)
2. **Dual build mode support**: Standalone and E3SM-embedded builds
3. **Modern target-based linking**: Uses `target_link_libraries()`, `target_include_directories()`
4. **Multi-architecture support**: CUDA, HIP, SYCL, OpenMP, Serial
5. **C++17 enforcement**: Proper standard requirements
6. **Helper script generation**: Creates omega_build.sh, omega_ctest.sh, etc.

---

## Identified Problems

### Problem 1: OmegaBuild.cmake is a Monolithic "God File"

**Location**: `cmake/OmegaBuild.cmake` (741 lines)

**Issue**: Single file contains 10 unrelated macros mixing:
- Variable declarations (122 variables)
- Compiler detection by parsing `--version` output
- MPI configuration extraction from shell scripts
- Shell script generation via 40+ `file(APPEND)` calls
- Build configuration logic

**Impact**: Hard to maintain, test, or modify individual components.

### Problem 2: Deprecated CMake Patterns

#### 2a. Direct CMAKE_CXX_FLAGS Manipulation (6 instances)

**Locations in OmegaBuild.cmake**:
- Line 360
- Line 388
- Line 393
- Line 396
- Line 455
- Line 458

**Current Pattern**:
```cmake
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OMEGA_CXX_FLAGS}")
```

**Problems**:
- Bypasses CMake's flag deduplication
- Makes cross-platform builds fragile
- Ignores target-specific requirements

**Solution**:
```cmake
add_compile_options(${OMEGA_CXX_FLAGS})
# Or for target-specific:
target_compile_options(OmegaLib PRIVATE ${OMEGA_CXX_FLAGS})
```

#### 2b. Global add_definitions() Usage (12 instances)

**Locations in OmegaBuild.cmake**:
- Line 543
- Line 551
- Lines 555-567
- Line 574
- Line 579
- Line 584
- Line 586
- Line 590

**Current Pattern**:
```cmake
add_definitions(-DOMEGA_DEBUG)
add_definitions(-DOMEGA_LOG_LEVEL=${OMEGA_LOG_LEVEL})
```

**Problems**:
- Applies globally to all targets
- Hard to control scope and visibility
- Inconsistent with modern CMake

**Solution**:
```cmake
target_compile_definitions(OmegaLib PRIVATE OMEGA_DEBUG)
target_compile_definitions(OmegaLib PRIVATE OMEGA_LOG_LEVEL=${OMEGA_LOG_LEVEL})
```

### Problem 3: Fragile Shell Script Parsing

#### 3a. Environment Variable Extraction

**Location**: OmegaBuild.cmake lines 93-107

**Current Pattern**:
```cmake
file(READ "${CASEROOT}/.env_mach_specific.sh" ENV_OUTPUT)
string(REPLACE "\n" ";" lines ${ENV_OUTPUT})
foreach(line ${lines})
    string(REGEX MATCH "([A-Za-z_][A-Za-z0-9_]*)=(.*)" ENV_LINE ${line})
    # ... complex parsing logic
endforeach()
```

**Problems**:
- Fragile regex parsing of bash syntax
- Breaks if shell script format changes
- Hard to debug

#### 3b. MPI Configuration Extraction

**Location**: OmegaBuild.cmake lines 116-144

**Current Pattern**: Parses `.case.run.sh` to extract MPI arguments (-n, -N, -c, -np)

**Problems**:
- Hardcoded MPI flag patterns
- Breaks with different MPI implementations
- Should use CMake's FindMPI module

#### 3c. Compiler Detection by Version String

**Location**: OmegaBuild.cmake lines 226-275

**Current Pattern**:
```cmake
execute_process(COMMAND ${OMEGA_CXX_COMPILER} --version ...)
string(REGEX MATCH "HIP|hip" _HIP_CHECK "${_CXX_VER_OUTPUT}")
string(REGEX MATCH "AMD|amd" _AMD_CHECK "${_CXX_VER_OUTPUT}")
```

**Problems**:
- Compiler-specific string matching
- Breaks with new compiler versions
- Should use FindCUDA, FindHIP, etc.

### Problem 4: Excessive execute_process Calls (41 instances)

**Locations throughout OmegaBuild.cmake**:
- Line 52: xmlquery calls
- Line 240: Compiler version detection
- Line 604+: chmod operations for generated scripts

**Impact**: Slow configuration, potential race conditions, hard to debug.

### Problem 5: Repetitive Test Registration (496 lines)

**Location**: `test/CMakeLists.txt`

**Current Pattern**:
```cmake
add_omega_test(DATA_TYPES_TEST testDataTypes.exe base/DataTypesTest.cpp "-n;1")
add_omega_test(MACHINE_ENV_TEST testMachEnv.exe base/MachEnvTest.cpp "-n;8")
add_omega_test(BROADCAST_TEST testBroadcast.exe base/BroadcastTest.cpp "-n;8")
# ... 37+ more identical patterns
```

**Problems**:
- 40+ nearly identical function calls
- 90% boilerplate
- Adding tests requires copy-paste

**Solution**: Data-driven test registration (see Phase 1).

### Problem 6: GLOB_RECURSE Usage

**Location**: `src/CMakeLists.txt` line 75

**Current Pattern**:
```cmake
file(GLOB_RECURSE _LIBSRC_FILES infra/*.cpp base/*.cpp ocn/*.cpp timeStepping/*.cpp)
```

**Problems**:
- Files globbed at configuration time only
- Adding new .cpp files requires CMake reconfiguration
- Can cause stale builds if files are deleted

**Solution**: Explicit file lists or `GLOB` with `CONFIGURE_DEPENDS`.

### Problem 7: Missing Modern CMake Features

| Feature | Status | Impact |
|---------|--------|--------|
| CMakePresets.json | Missing | No standardized configurations |
| Package config files | Missing | Can't integrate as CMake package |
| Target namespace (Omega::) | Missing | No clear exported interface |
| Version management | Missing | No compatibility checking |
| install(EXPORT) | Missing | Libraries not exported |
| FetchContent | Not used | Manual external dependency handling |
| Generator expressions | Minimal | Many if() blocks could be simplified |

### Problem 8: E3SM Integration Blockers

| Issue | Location | Description |
|-------|----------|-------------|
| Hard-coded paths | Lines 16, 502, 507 | `../..` assumes directory structure |
| Implicit detection | Lines 56-59, 502-507 | E3SM_SOURCE_DIR detected implicitly |
| Unimplemented TODO | Line 527 | "set OMEGA_ARCH according to E3SM variables" |
| Embedded CIME logic | init_standalone_build | Should be modular |
| No package export | N/A | E3SM can't link OMEGA as package |

---

## Phase 1: Clean Up Current Build System

### Task 1.1: Split OmegaBuild.cmake into Modules

Create the following structure:
```
cmake/
├── OmegaBuild.cmake        # Reduced to include() calls only (~50 lines)
├── OmegaVariables.cmake    # Variable/option declarations (~150 lines)
├── OmegaCompilers.cmake    # Compiler detection and setup (~100 lines)
├── OmegaMPI.cmake          # MPI configuration (~50 lines)
├── OmegaExternals.cmake    # External dependency logic (~100 lines)
├── OmegaUtils.cmake        # Helper macros (~100 lines)
├── OmegaScripts.cmake      # Script generation (~150 lines)
└── FindParmetis.cmake      # Keep existing (modernize later)
```

**New OmegaBuild.cmake structure**:
```cmake
# cmake/OmegaBuild.cmake - Orchestration only
include(${CMAKE_CURRENT_LIST_DIR}/OmegaVariables.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/OmegaUtils.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/OmegaCompilers.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/OmegaMPI.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/OmegaExternals.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/OmegaScripts.cmake)

# Main macros that call into the modules
macro(setup_common)
    omega_setup_variables()
    omega_setup_compilers()
endmacro()

macro(init_standalone_build)
    omega_init_cime()
    omega_configure_mpi()
    omega_generate_scripts()
endmacro()
# ...
```

### Task 1.2: Replace add_definitions() with target_compile_definitions()

**Files to modify**: `cmake/OmegaBuild.cmake`

**Changes**:
```cmake
# Before (OmegaBuild.cmake ~line 543-590):
add_definitions(-DOMEGA_DEBUG)
add_definitions(-DOMEGA_LOG_LEVEL=${OMEGA_LOG_LEVEL})

# After:
# Create an interface library for compile definitions
add_library(OmegaCompileDefinitions INTERFACE)

if(OMEGA_DEBUG)
    target_compile_definitions(OmegaCompileDefinitions INTERFACE OMEGA_DEBUG)
endif()
target_compile_definitions(OmegaCompileDefinitions INTERFACE
    OMEGA_LOG_LEVEL=${OMEGA_LOG_LEVEL}
)

# Then in src/CMakeLists.txt:
target_link_libraries(OmegaLib PRIVATE OmegaCompileDefinitions)
```

### Task 1.3: Replace CMAKE_CXX_FLAGS with add_compile_options()

**Files to modify**: `cmake/OmegaBuild.cmake`

**Changes**:
```cmake
# Before (lines 360, 388, 393, 396, 455, 458):
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OMEGA_CXX_FLAGS}")

# After:
# Collect flags in a list
list(APPEND OMEGA_COMPILE_OPTIONS ${OMEGA_CXX_FLAGS})

# Apply at the end of configuration
add_compile_options(${OMEGA_COMPILE_OPTIONS})
```

### Task 1.4: Convert Test Registration to Data-Driven Loop

**File to modify**: `test/CMakeLists.txt`

**Before** (496 lines):
```cmake
add_omega_test(DATA_TYPES_TEST testDataTypes.exe base/DataTypesTest.cpp "-n;1")
add_omega_test(MACHINE_ENV_TEST testMachEnv.exe base/MachEnvTest.cpp "-n;8")
# ... 38 more
```

**After** (~80 lines):
```cmake
# Test configuration data: NAME;EXECUTABLE;SOURCE;MPI_ARGS
set(OMEGA_UNIT_TESTS
    # Base tests
    "DATA_TYPES_TEST;testDataTypes.exe;base/DataTypesTest.cpp;-n;1"
    "MACHINE_ENV_TEST;testMachEnv.exe;base/MachEnvTest.cpp;-n;8"
    "BROADCAST_TEST;testBroadcast.exe;base/BroadcastTest.cpp;-n;8"
    "HALO_TEST;testHalo.exe;base/HaloTest.cpp;-n;4"

    # Infra tests
    "LOGGING_TEST;testLogging.exe;infra/LoggingTest.cpp;-n;1"
    "CONFIG_TEST;testConfig.exe;infra/ConfigTest.cpp;-n;1"
    "IO_TEST;testIO.exe;infra/IOTest.cpp;-n;4"
    "FIELD_TEST;testField.exe;infra/FieldTest.cpp;-n;4"

    # Add remaining tests here...
)

# Register all tests via loop
foreach(test_spec ${OMEGA_UNIT_TESTS})
    # Parse the semicolon-separated spec
    string(REPLACE ";" ";" test_parts "${test_spec}")
    list(GET test_parts 0 test_name)
    list(GET test_parts 1 test_exe)
    list(GET test_parts 2 test_src)
    list(SUBLIST test_parts 3 -1 test_mpi_args)

    add_omega_test(${test_name} ${test_exe} ${test_src} "${test_mpi_args}")
endforeach()
```

### Task 1.5: Replace GLOB_RECURSE with Explicit File Lists

**File to modify**: `src/CMakeLists.txt`

**Option A**: Explicit file lists (preferred for stability)
```cmake
set(OMEGA_BASE_SOURCES
    base/DataTypes.cpp
    base/MachEnv.cpp
    base/Decomp.cpp
    base/Halo.cpp
    # ... list all files
)

set(OMEGA_INFRA_SOURCES
    infra/Logging.cpp
    infra/Config.cpp
    infra/Field.cpp
    # ... list all files
)

set(OMEGA_OCN_SOURCES
    ocn/HorzMesh.cpp
    ocn/State.cpp
    # ... list all files
)

set(_LIBSRC_FILES
    ${OMEGA_BASE_SOURCES}
    ${OMEGA_INFRA_SOURCES}
    ${OMEGA_OCN_SOURCES}
    ${OMEGA_TIMESTEPPING_SOURCES}
)
```

**Option B**: GLOB with CONFIGURE_DEPENDS (less maintenance, but rebuilds more)
```cmake
file(GLOB_RECURSE _LIBSRC_FILES CONFIGURE_DEPENDS
    infra/*.cpp
    base/*.cpp
    ocn/*.cpp
    timeStepping/*.cpp
)
```

### Task 1.6: Move Script Generation Out of CMake

**Current**: Shell scripts generated via 40+ `file(APPEND)` calls in OmegaBuild.cmake

**Proposed**: Use CMake's `configure_file()` with templates

Create template files:
```
cmake/templates/
├── omega_env.sh.in
├── omega_build.sh.in
├── omega_ctest.sh.in
├── omega_run.sh.in
└── omega_profile.sh.in
```

**Example template** (`cmake/templates/omega_build.sh.in`):
```bash
#!/bin/bash
# Generated by OMEGA CMake build system
# Build directory: @OMEGA_BUILD_DIR@
# Source directory: @OMEGA_SOURCE_DIR@

cd "@OMEGA_BUILD_DIR@"
make -j @OMEGA_BUILD_JOBS@ @OMEGA_BUILD_TARGET@
```

**CMake usage**:
```cmake
set(OMEGA_BUILD_JOBS 8)
set(OMEGA_BUILD_TARGET "all")
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/templates/omega_build.sh.in
    ${OMEGA_BUILD_DIR}/omega_build.sh
    @ONLY
)
```

### Task 1.7: Modernize Compiler Detection

**Current**: Parse compiler `--version` output with regex

**Proposed**: Use CMake's built-in detection

```cmake
# cmake/OmegaCompilers.cmake

# Detect CUDA
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(OMEGA_HAS_CUDA TRUE)
endif()

# Detect HIP (CMake 3.21+)
check_language(HIP)
if(CMAKE_HIP_COMPILER)
    enable_language(HIP)
    set(OMEGA_HAS_HIP TRUE)
endif()

# For SYCL, use FindSYCL or compiler-specific detection
find_package(SYCL QUIET)
if(SYCL_FOUND)
    set(OMEGA_HAS_SYCL TRUE)
endif()

# Set OMEGA_ARCH based on detection (if not explicitly set)
if(NOT OMEGA_ARCH)
    if(OMEGA_HAS_CUDA)
        set(OMEGA_ARCH "CUDA")
    elseif(OMEGA_HAS_HIP)
        set(OMEGA_ARCH "HIP")
    elseif(OMEGA_HAS_SYCL)
        set(OMEGA_ARCH "SYCL")
    elseif(OpenMP_CXX_FOUND)
        set(OMEGA_ARCH "OPENMP")
    else()
        set(OMEGA_ARCH "SERIAL")
    endif()
endif()
```

---

## Phase 2: Prepare for E3SM Integration

### Task 2.1: Create CMake Package Configuration Files

Create `cmake/OmegaConfig.cmake.in`:
```cmake
@PACKAGE_INIT@

include(CMakeFindDependencyMacro)

# Find required dependencies
find_dependency(MPI REQUIRED)
find_dependency(Kokkos REQUIRED)
find_dependency(NetCDF REQUIRED)

# Include targets
include("${CMAKE_CURRENT_LIST_DIR}/OmegaTargets.cmake")

# Provide version info
set(OMEGA_VERSION @OMEGA_VERSION@)
set(OMEGA_VERSION_MAJOR @OMEGA_VERSION_MAJOR@)
set(OMEGA_VERSION_MINOR @OMEGA_VERSION_MINOR@)

check_required_components(Omega)
```

Create `cmake/OmegaConfigVersion.cmake.in`:
```cmake
set(PACKAGE_VERSION "@OMEGA_VERSION@")

if(PACKAGE_VERSION VERSION_LESS PACKAGE_FIND_VERSION)
    set(PACKAGE_VERSION_COMPATIBLE FALSE)
else()
    set(PACKAGE_VERSION_COMPATIBLE TRUE)
    if(PACKAGE_VERSION VERSION_EQUAL PACKAGE_FIND_VERSION)
        set(PACKAGE_VERSION_EXACT TRUE)
    endif()
endif()
```

### Task 2.2: Export Targets with Namespace

Add to `src/CMakeLists.txt`:
```cmake
# Create alias with namespace
add_library(Omega::OmegaLib ALIAS OmegaLib)

# Install library
install(TARGETS OmegaLib OmegaLibFlags
    EXPORT OmegaTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# Install headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION include/omega
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Export targets
install(EXPORT OmegaTargets
    FILE OmegaTargets.cmake
    NAMESPACE Omega::
    DESTINATION lib/cmake/Omega
)
```

### Task 2.3: Add Version Management

Add to top-level `CMakeLists.txt`:
```cmake
# Version definition
set(OMEGA_VERSION_MAJOR 0)
set(OMEGA_VERSION_MINOR 1)
set(OMEGA_VERSION_PATCH 0)
set(OMEGA_VERSION "${OMEGA_VERSION_MAJOR}.${OMEGA_VERSION_MINOR}.${OMEGA_VERSION_PATCH}")

project(OmegaOceanModel VERSION ${OMEGA_VERSION} LANGUAGES CXX)
```

Generate version header:
```cmake
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/omega/version.h
)
```

### Task 2.4: Make E3SM Integration Opt-In

Create `cmake/OmegaE3SMIntegration.cmake`:
```cmake
# E3SM Integration Module
# Include this when building OMEGA as part of E3SM

option(OMEGA_E3SM_INTEGRATION "Build as E3SM component" OFF)

if(OMEGA_E3SM_INTEGRATION)
    # Expect these variables from E3SM
    if(NOT DEFINED E3SM_SOURCE_DIR)
        message(FATAL_ERROR "E3SM_SOURCE_DIR must be set for E3SM integration")
    endif()

    # Use E3SM's compiler settings
    if(DEFINED E3SM_CXX_COMPILER)
        set(CMAKE_CXX_COMPILER ${E3SM_CXX_COMPILER})
    endif()

    # Use E3SM's architecture detection
    if(DEFINED E3SM_ARCH)
        set(OMEGA_ARCH ${E3SM_ARCH})
    endif()

    # Skip OMEGA's own compiler detection
    set(OMEGA_SKIP_COMPILER_DETECTION TRUE)
endif()
```

### Task 2.5: Document Expected E3SM Variables

Create `cmake/E3SM_INTEGRATION.md`:
```markdown
# E3SM Integration Guide

## Required Variables from E3SM

When building OMEGA as an E3SM component, the following variables must be set:

| Variable | Type | Description |
|----------|------|-------------|
| E3SM_SOURCE_DIR | PATH | Root of E3SM source tree |
| E3SM_CXX_COMPILER | FILEPATH | C++ compiler path |
| E3SM_C_COMPILER | FILEPATH | C compiler path |
| E3SM_Fortran_COMPILER | FILEPATH | Fortran compiler path |
| E3SM_ARCH | STRING | Target architecture (CUDA/HIP/OPENMP/SERIAL) |

## Optional Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| E3SM_BUILD_TYPE | STRING | Release | Debug or Release |
| E3SM_MPI_EXEC | FILEPATH | mpiexec | MPI launcher |
| E3SM_MPI_ARGS | STRING | "" | Additional MPI arguments |

## Linking OMEGA in E3SM

```cmake
find_package(Omega REQUIRED)
target_link_libraries(e3sm_ocean PRIVATE Omega::OmegaLib)
```
```

### Task 2.6: Create CMakePresets.json

```json
{
    "version": 6,
    "cmakeMinimumRequired": {
        "major": 3,
        "minor": 21,
        "patch": 0
    },
    "configurePresets": [
        {
            "name": "base",
            "hidden": true,
            "binaryDir": "${sourceDir}/build/${presetName}",
            "cacheVariables": {
                "CMAKE_EXPORT_COMPILE_COMMANDS": "ON"
            }
        },
        {
            "name": "debug",
            "inherits": "base",
            "cacheVariables": {
                "OMEGA_BUILD_TYPE": "Debug",
                "OMEGA_BUILD_TEST": "ON"
            }
        },
        {
            "name": "release",
            "inherits": "base",
            "cacheVariables": {
                "OMEGA_BUILD_TYPE": "Release"
            }
        },
        {
            "name": "cuda",
            "inherits": "release",
            "cacheVariables": {
                "OMEGA_ARCH": "CUDA"
            }
        },
        {
            "name": "hip",
            "inherits": "release",
            "cacheVariables": {
                "OMEGA_ARCH": "HIP"
            }
        },
        {
            "name": "openmp",
            "inherits": "release",
            "cacheVariables": {
                "OMEGA_ARCH": "OPENMP"
            }
        },
        {
            "name": "serial",
            "inherits": "release",
            "cacheVariables": {
                "OMEGA_ARCH": "SERIAL"
            }
        }
    ],
    "buildPresets": [
        {
            "name": "debug",
            "configurePreset": "debug"
        },
        {
            "name": "release",
            "configurePreset": "release"
        }
    ],
    "testPresets": [
        {
            "name": "debug",
            "configurePreset": "debug",
            "output": {
                "outputOnFailure": true
            }
        }
    ]
}
```

---

## Detailed File-by-File Analysis

### CMakeLists.txt (114 lines)

**Status**: Good overall structure

**Minor Issues**:
- Line 41: `cmake_minimum_required(VERSION 3.21)` - document why 3.21 (HIP support)
- Missing: Project version specification

**Recommended Changes**:
```cmake
# Add at top
cmake_minimum_required(VERSION 3.21)  # Required for HIP language support

# Add version
set(OMEGA_VERSION "0.1.0")
project(OmegaOceanModel VERSION ${OMEGA_VERSION} LANGUAGES CXX)
```

### cmake/OmegaBuild.cmake (741 lines)

**Status**: Needs major refactoring

**Macro Inventory**:
| Macro | Lines | Purpose | Action |
|-------|-------|---------|--------|
| `run_bash_command` | ~20 | Execute bash | Keep in OmegaUtils.cmake |
| `cime_xmlquery` | ~30 | Query CIME XML | Keep in OmegaUtils.cmake |
| `setup_common` | ~100 | Initialize variables | Split to OmegaVariables.cmake |
| `init_standalone_build` | ~300 | CIME/compiler setup | Split to multiple modules |
| `update_variables` | ~50 | Configure dependencies | Move to OmegaExternals.cmake |
| `build_model` | ~30 | Build configuration | Keep |
| `add_external_libraries` | ~20 | External deps | Move to OmegaExternals.cmake |
| `add_omega_source` | ~20 | Source configuration | Keep |
| `add_omega_test` | ~30 | Test registration | Move to OmegaTests.cmake |
| `wrap_outputs` | ~40 | Installation | Keep |

### src/CMakeLists.txt (128 lines)

**Status**: Good, minor improvements needed

**Issues**:
- Line 75: `GLOB_RECURSE` usage
- Missing: Install targets
- Missing: Target export

### test/CMakeLists.txt (496 lines)

**Status**: Needs refactoring to data-driven approach

**Current test count**: 40+ tests
**Target line count**: ~80 lines (with data-driven approach)

### external/CMakeLists.txt (142 lines)

**Status**: Medium priority improvements

**Issues**:
- Line 6-17: GSW-C uses hardcoded `make` command
- Lines 52-67: Manual environment variable forwarding to Scorpio
- Lines 119-134: Redundant IMPORTED library declarations

---

## Code Examples

### Example: New OmegaVariables.cmake

```cmake
# cmake/OmegaVariables.cmake
# Variable and option declarations for OMEGA build system

#------------------------------------------------------------------------------
# Project identification
#------------------------------------------------------------------------------
set(OMEGA_PROJECT_NAME "OmegaOceanModel")
set(OMEGA_EXE_NAME "omega.exe")
set(OMEGA_LIB_NAME "OmegaLib")

#------------------------------------------------------------------------------
# Build mode options
#------------------------------------------------------------------------------
set(OMEGA_BUILD_MODES "E3SM" "STANDALONE" "NOT_DEFINED")
set(OMEGA_BUILD_MODE "NOT_DEFINED" CACHE STRING "Build mode")
set_property(CACHE OMEGA_BUILD_MODE PROPERTY STRINGS ${OMEGA_BUILD_MODES})

#------------------------------------------------------------------------------
# Build type
#------------------------------------------------------------------------------
set(OMEGA_DEFAULT_BUILD_TYPE "Release")
if(NOT OMEGA_BUILD_TYPE)
    set(OMEGA_BUILD_TYPE ${OMEGA_DEFAULT_BUILD_TYPE})
endif()

#------------------------------------------------------------------------------
# Architecture options
#------------------------------------------------------------------------------
set(OMEGA_ARCH_OPTIONS "CUDA" "HIP" "OPENMP" "SYCL" "SERIAL")
set(OMEGA_ARCH "" CACHE STRING "Target architecture")
set_property(CACHE OMEGA_ARCH PROPERTY STRINGS ${OMEGA_ARCH_OPTIONS})

#------------------------------------------------------------------------------
# Build options
#------------------------------------------------------------------------------
option(OMEGA_BUILD_EXECUTABLE "Build OMEGA executable" ON)
option(OMEGA_BUILD_TEST "Build OMEGA tests" OFF)
option(OMEGA_DEBUG "Enable debug mode" OFF)

#------------------------------------------------------------------------------
# Logging options
#------------------------------------------------------------------------------
set(OMEGA_LOG_LEVEL "OMEGA_LOG_INFO" CACHE STRING "Default logging level")
option(OMEGA_LOG_FLUSH "Enable unbuffered logging" OFF)
set(OMEGA_LOG_TASKS "0" CACHE STRING "Tasks that generate log files")

#------------------------------------------------------------------------------
# Performance options
#------------------------------------------------------------------------------
set(OMEGA_MEMORY_LAYOUT "RIGHT" CACHE STRING "Kokkos memory layout (LEFT or RIGHT)")
set(OMEGA_TILE_LENGTH 64 CACHE STRING "Kokkos tile length")
set(OMEGA_VECTOR_LENGTH 1 CACHE STRING "Vector length for loop blocking")

#------------------------------------------------------------------------------
# Dependency paths
#------------------------------------------------------------------------------
set(OMEGA_PARMETIS_ROOT "" CACHE PATH "Parmetis installation directory")
set(OMEGA_METIS_ROOT "" CACHE PATH "Metis installation directory")
set(OMEGA_GKLIB_ROOT "" CACHE PATH "GKlib installation directory")
```

### Example: New OmegaUtils.cmake

```cmake
# cmake/OmegaUtils.cmake
# Utility macros and functions for OMEGA build system

#------------------------------------------------------------------------------
# Execute a bash command and capture output
#------------------------------------------------------------------------------
function(omega_run_bash_command COMMAND OUTPUT_VAR)
    execute_process(
        COMMAND bash -c "${COMMAND}"
        OUTPUT_VARIABLE _output
        ERROR_VARIABLE _error
        RESULT_VARIABLE _result
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT _result EQUAL 0)
        message(WARNING "Command failed: ${COMMAND}\nError: ${_error}")
    endif()
    set(${OUTPUT_VAR} "${_output}" PARENT_SCOPE)
endfunction()

#------------------------------------------------------------------------------
# Query CIME XML configuration
#------------------------------------------------------------------------------
function(omega_cime_xmlquery CIMEROOT CASEROOT XMLVAR OUTPUT_VAR)
    execute_process(
        COMMAND ${CIMEROOT}/CIME/Tools/xmlquery
                --caseroot ${CASEROOT}
                --value ${XMLVAR}
        OUTPUT_VARIABLE _output
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _result
    )
    if(_result EQUAL 0)
        set(${OUTPUT_VAR} "${_output}" PARENT_SCOPE)
    else()
        message(WARNING "xmlquery failed for ${XMLVAR}")
        set(${OUTPUT_VAR} "" PARENT_SCOPE)
    endif()
endfunction()

#------------------------------------------------------------------------------
# Add compile definition to target (wrapper for consistency)
#------------------------------------------------------------------------------
function(omega_add_definition TARGET DEFINITION)
    target_compile_definitions(${TARGET} PRIVATE ${DEFINITION})
endfunction()
```

---

## Progress Tracking

### Phase 1 Checklist

- [ ] Task 1.1: Split OmegaBuild.cmake into modules
  - [ ] Create OmegaVariables.cmake
  - [ ] Create OmegaCompilers.cmake
  - [ ] Create OmegaMPI.cmake
  - [ ] Create OmegaExternals.cmake
  - [ ] Create OmegaUtils.cmake
  - [ ] Create OmegaScripts.cmake
  - [ ] Update OmegaBuild.cmake to use includes
- [ ] Task 1.2: Replace add_definitions() with target_compile_definitions()
- [ ] Task 1.3: Replace CMAKE_CXX_FLAGS with add_compile_options()
- [ ] Task 1.4: Convert test registration to data-driven loop
- [ ] Task 1.5: Replace GLOB_RECURSE with explicit file lists
- [ ] Task 1.6: Move script generation to templates
- [ ] Task 1.7: Modernize compiler detection

### Phase 2 Checklist

- [ ] Task 2.1: Create OmegaConfig.cmake.in
- [ ] Task 2.2: Export targets with Omega:: namespace
- [ ] Task 2.3: Add version management
- [ ] Task 2.4: Make E3SM integration opt-in
- [ ] Task 2.5: Document expected E3SM variables
- [ ] Task 2.6: Create CMakePresets.json

---

## References

- [Modern CMake Best Practices](https://cliutils.gitlab.io/modern-cmake/)
- [CMake Documentation](https://cmake.org/cmake/help/latest/)
- [E3SM Build System](https://github.com/E3SM-Project/E3SM)
- [Kokkos CMake Integration](https://kokkos.github.io/kokkos-core-wiki/building.html)

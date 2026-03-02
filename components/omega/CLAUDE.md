# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OMEGA (Ocean Model for E3SM Global Applications) is the ocean component for E3SM v4. It is a complete redesign of MPAS-Ocean, rewritten in C++17 for better computational performance and portability across CPU/GPU architectures. In this project, OMEGA CMake build system will be refactored to make OMEGA build system to be merged into E3SM CMake build system.

To refactor OMEGA build system, there will be two phases: 1) Update current OMEGA build system to be clearner and structured, and 2) modify OMEGA build system to be merged into E3SM CMake build system.

## Build Commands

No local build. OMEGA should be built remotely manually on HPC system.

### Running Tests

No local test. OMEGA should be tested remotely manually on HPC system.

## Code Architecture

### Source Directory Structure (`src/`)
- **base/**: Machine model, domain decomposition, data types, MPI communication (Broadcast, Halo)
- **infra/**: Infrastructure - Logging, Config, Field management, TimeMgr, IOStream
- **ocn/**: Ocean physics - EOS (equation of state), tendencies, tracers, vertical coordinate
  - **auxiliaryVars/**: Derived variables computed from state
- **timeStepping/**: Time integration schemes (RK2, RK4, Forward-Backward)
- **drivers/**: Standalone ocean driver
- **analysis/**: Analysis and diagnostics

### Key Design Patterns
- Uses Kokkos for performance portability across CUDA, HIP, SYCL, OpenMP, and serial backends
- Configuration via YAML files (see `configs/Default.yml`)
- Field management system for ocean state variables
- Horizontal mesh based on unstructured polygonal cells

### External Dependencies (`external/`)
- **GSW-C**: TEOS10 equation of state library
- **yaml-cpp**: YAML configuration parsing
- Links to: Parmetis (domain decomposition), NetCDF/PNetCDF (I/O), GPTL (profiling)

### Test Structure (`test/`)
Tests mirror the `src/` directory structure. Key tests include mesh operations, field management, I/O, tendencies, and time stepping. Tests run with varying MPI process counts (1, 2, 4, 8).

## Build System Variables

Key CMake variables:
- `OMEGA_BUILD_MODE`: STANDALONE or E3SM
- `OMEGA_ARCH`: CUDA, HIP, OPENMP, SYCL, or SERIAL
- `OMEGA_BUILD_TYPE`: Debug or Release
- `OMEGA_CIME_MACHINE`: Machine name from E3SM config_machines.xml
- `OMEGA_CIME_COMPILER`: Compiler name from E3SM config

## Code Style

LLVM coding standards enforced via clang-format. The `.clang-format` file uses LLVM style with 3-space indentation.
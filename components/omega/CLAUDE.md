# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OMEGA (Ocean Model for E3SM Global Applications) is the ocean component for E3SM v4 (Energy Exascale Earth System Model). It is a complete redesign/rewrite of MPAS-Ocean for better computational performance and GPU portability using Kokkos.

## Build System

OMEGA uses CMake and supports two build modes:
1. **Standalone build**: Uses CIME for machine/compiler configuration
2. **E3SM embedded build**: Built as part of the full E3SM system

### Standalone Build Commands

```bash
# From build directory (creates e3smcase for machine config):
cmake /path/to/omega -DOMEGA_CIME_MACHINE=<machine> -DOMEGA_CIME_COMPILER=<compiler>

# Build using generated script:
./omega_build.sh

# Or directly:
source omega_env.sh && make -j

# Run tests:
./omega_ctest.sh
# Or: source omega_env.sh && ctest --output-on-failure

# Run single test:
./omega_ctest.sh -R <TEST_NAME>
```

### Key CMake Variables
- `OMEGA_ARCH`: Target architecture (CUDA, HIP, SYCL, OPENMP, SERIAL) - auto-detected
- `OMEGA_BUILD_TYPE`: Debug or Release (default: Release)
- `OMEGA_DEBUG`: Enable error message throwing
- `OMEGA_SINGLE_PRECISION`: Use 32-bit floats instead of 64-bit doubles
- `OMEGA_CIME_MACHINE`: Target machine name for CIME
- `OMEGA_CIME_COMPILER`: Compiler for CIME

## Architecture

### Source Code Organization (`src/`)
- **base/**: Low-level infrastructure - MPI communication, domain decomposition, Kokkos data types, I/O, halo exchanges
- **infra/**: Intermediate infrastructure - time management, configuration (YAML), logging, fields, I/O streams
- **ocn/**: Ocean physics - tendencies, operators, tracers, equation of state, vertical mixing, mesh handling
- **timeStepping/**: Time integration schemes (Forward-Backward, RK2, RK4)
- **drivers/**: Top-level drivers for standalone and coupled E3SM execution

### Key Patterns

**Kokkos Arrays**: All arrays use Kokkos Views. Aliases defined in `DataTypes.h`:
- `Array1DReal`, `Array2DReal`, etc. - Device arrays
- `HostArray1DReal`, `HostArray2DReal`, etc. - Host arrays
- Types: `I4`, `I8`, `R4`, `R8`, `Real` (precision-dependent)

**Namespace**: All OMEGA code lives in the `OMEGA` namespace.

**Driver Pattern**: Ocean simulation follows init/run/finalize pattern:
- `ocnInit(MPI_Comm)` - Initialize all modules
- `ocnRun(TimeInstant&)` - Advance simulation
- `ocnFinalize(TimeInstant&)` - Cleanup

**Configuration**: Uses YAML files (see `configs/Default.yml`). Access via `Config` class.

### External Dependencies
- **Kokkos**: GPU/CPU portability layer (from E3SM externals)
- **Scorpio (PIO)**: Parallel I/O (from E3SM externals)
- **ParMETIS**: Domain decomposition
- **yaml-cpp**: Configuration parsing
- **spdlog**: Logging
- **GSW-C**: TEOS-10 seawater equation of state

## Testing

Tests are in `test/` organized by source module (base/, infra/, ocn/, etc.). Test names follow pattern like `DATA_TYPES_TEST`, `HALO_TEST`, `HORZMESH_TEST`.

Tests use MPI and are run through ctest. Most tests require 8 MPI ranks. The `add_omega_test()` CMake function handles test setup.

## Memory Layout

Configurable via `OMEGA_MEMORY_LAYOUT`:
- `RIGHT` (default): Row-major (C-style)
- `LEFT`: Column-major (Fortran-style)

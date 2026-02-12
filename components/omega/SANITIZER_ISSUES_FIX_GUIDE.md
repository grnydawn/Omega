# OMEGA GPU Sanitizer Issues - Fix Guide for AI Coding Agent

## Document Purpose
This document provides detailed findings from NVIDIA Compute Sanitizer analysis (initcheck and memcheck tools) of the OMEGA ocean model test suite. Use this guide to locate and fix memory initialization issues in the codebase.

---

## Executive Summary

| Issue Type | Severity | Root Cause | Primary Fix Location |
|-----------|----------|------------|---------------------|
| Uninitialized Memory in Halo Exchange | **HIGH** | Kokkos Views not initialized before MPI halo exchange | `OMEGA::Halo` class and callers |
| Invalid Pointer Query in MPI | LOW | MPI GTL probing non-CUDA pointers | Benign - optional suppression |

---

## Issue #1: Uninitialized Memory Access in Halo Exchange (HIGH PRIORITY)

### Problem Description
The `OMEGA::Halo::exchangeFullArrayHalo<>` template function sends data via GPU-aware MPI where the source buffers contain uninitialized memory. When the Cray MPI GPU Transport Layer attempts to copy this data using `cudaMemcpyAsync`, the sanitizer detects reads from uninitialized memory regions.

### Error Signature
```
========= Host API memory access error at host access to 0x7f... of size N bytes
=========     Uninitialized access at 0x7f... on access by cudaMemcpy source
```

### Technical Details

#### Call Flow Where Error Occurs
```
Application Code (various test functions)
    ↓
OMEGA::Halo::exchangeFullArrayHalo<KokkosViewType>(view, meshElement)
    ↓
OMEGA::Halo::startSends(bool)
    ↓
MPI_Isend (libmpi_gnu_123.so)
    ↓
MPIDI_CRAY_Common_lmt_ctrl_send_rts_cb
    ↓
gtlt_cuda_memcpy_async (libmpi_gtl_cuda.so)
    ↓
cudaMemcpyAsync (libcudart.so) ← ERROR: Source contains uninitialized data
```

#### Affected Kokkos View Types
The following Kokkos View types are being passed with uninitialized data:
- `Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::CudaSpace>`
- `Kokkos::View<double***, Kokkos::LayoutRight, Kokkos::CudaSpace>`
- `Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::CudaSpace>`
- `Kokkos::View<int*, Kokkos::LayoutRight, Kokkos::CudaSpace>`

### Affected Functions and Files

#### 1. Halo Exchange Core (Primary Fix Location)

**File:** Look for `Halo.cpp` or `Halo.hpp` in the OMEGA source tree

**Functions to examine:**
- `OMEGA::Halo::exchangeFullArrayHalo<T>()`
- `OMEGA::Halo::startSends(bool)`

**Issue:** The halo exchange assumes input Views are fully initialized, but callers pass partially initialized or uninitialized Views.

**Recommended Fix Options:**

Option A - Initialize in `exchangeFullArrayHalo` (defensive):
```cpp
template <typename ViewType>
int Halo::exchangeFullArrayHalo(ViewType& view, MeshElement elem) {
    // Add initialization of halo regions before exchange
    // Only initialize the halo portion that will be sent
    Kokkos::deep_copy(Kokkos::subview(view, haloRange), 0.0);

    // ... existing exchange logic
}
```

Option B - Add validation (debug mode):
```cpp
template <typename ViewType>
int Halo::exchangeFullArrayHalo(ViewType& view, MeshElement elem) {
#ifdef OMEGA_DEBUG
    // Validate that send regions are initialized
    validateInitialized(view, sendRegions);
#endif
    // ... existing exchange logic
}
```

#### 2. Test Application Callers (Secondary Fix Locations)

Each of these locations calls `exchangeFullArrayHalo` with potentially uninitialized data:

---

**Test: testAuxiliaryState.exe**
- **Function:** `initState()`
- **Call site:** `auxStateTest()` → `initState()` → `exchangeFullArrayHalo`
- **Fix:** Initialize the state View before calling halo exchange

```cpp
// Before:
// Kokkos::View<double**> stateView(...);
// halo->exchangeFullArrayHalo(stateView, element);

// After:
Kokkos::View<double**> stateView(...);
Kokkos::deep_copy(stateView, 0.0);  // Initialize entire view
// ... populate interior values ...
halo->exchangeFullArrayHalo(stateView, element);
```

---

**Test: testHorzOperatorsPlane.exe / testHorzOperatorsSphere.exe**
- **Function:** `setVectorEdge<>()` template
- **Call chain:** `testDivergence()` → `setVectorEdge()` → `exchangeFullArrayHalo`
- **Fix:** In `setVectorEdge`, ensure the edge View is initialized before halo exchange

```cpp
template <typename SetupType, typename ViewType>
int setVectorEdge(const SetupType& setup, const ViewType& edgeView,
                  EdgeComponent comp, Geometry geom,
                  const HorzMesh* mesh, ExchangeHalos exchange,
                  CartProjection proj) {
    // Ensure view is initialized before any operations
    Kokkos::deep_copy(edgeView, 0.0);

    // ... compute edge values for interior ...

    if (exchange == ExchangeHalos::Yes) {
        halo->exchangeFullArrayHalo(edgeView, MeshElement::Edge);
    }
}
```

---

**Test: testTendencyTermsPlane.exe / testTendencyTermsSphere.exe**
- **Function:** `testThickFluxDiv()`
- **Call chain:** `tendencyTermsTest()` → `testThickFluxDiv()` → `setVectorEdge()` → `exchangeFullArrayHalo`
- **Additional location:** `VertCoord::minMaxLayerEdge()` → `exchangeFullArrayHalo`
- **Fix:** Initialize Views in test setup functions

---

**Test: testHalo.exe**
- **Function:** `haloExchangeTest<ViewType>()`
- **Call site:** `main()` → `haloExchangeTest()` → `exchangeFullArrayHalo`
- **Fix:** This is a halo test - ensure test input Views are initialized

```cpp
template <typename ViewType>
int haloExchangeTest(Halo* halo, ViewType inputView,
                     ViewType& outputView, const char* name,
                     MeshElement elem) {
    // Initialize input view before exchange
    Kokkos::deep_copy(inputView, 0);  // or appropriate initial value

    // ... set up test values in interior ...

    return halo->exchangeFullArrayHalo(inputView, elem);
}
```

---

**Test: testAuxiliaryVarsPlane.exe / testAuxiliaryVarsSphere.exe**
- **Function:** `initState()`
- **Signature:** `initState(const View<double**>& view1, const View<double**>& view2, HorzMesh* mesh)`
- **Fix:** Initialize both Views at start of function

---

**Test: testTracers.exe**
- **Function:** `OMEGA::Tracers::updateTimeLevels()`
- **Call site:** `main()` → `Tracers::updateTimeLevels()` → `exchangeFullArrayHalo`
- **Fix:** Initialize tracer Views before time level update

**File:** Look for `Tracers.cpp` or `Tracers.hpp`
```cpp
void Tracers::updateTimeLevels() {
    // Ensure tracer data is initialized before halo exchange
    // The 3D tracer view should be initialized in constructor or init()

    // If doing time level swap, ensure new level is initialized
    Kokkos::deep_copy(tracerView, 0.0);  // if needed

    halo->exchangeFullArrayHalo(tracerView, MeshElement::Cell);
}
```

---

**Test: testDriver.exe**
- **Function:** `OMEGA::OceanState::exchangeHalo()`
- **Call chain:** `main()` → `ocnInit()` → `initOmegaModules()` → `OceanState::exchangeHalo()` → `exchangeFullArrayHalo`
- **Fix:** Initialize OceanState Views in constructor or init method

**File:** Look for `OceanState.cpp` or `OceanState.hpp`
```cpp
void OceanState::exchangeHalo(int timeLevel) {
    // Ensure state arrays are initialized before exchange
    // This should be done in OceanState constructor/init, not here

    halo->exchangeFullArrayHalo(layerThickness[timeLevel], MeshElement::Cell);
    halo->exchangeFullArrayHalo(normalVelocity[timeLevel], MeshElement::Edge);
    // ... other exchanges
}
```

---

**Test: testState.exe**
- **Function:** `initStateTest()`
- **Call chain:** `main()` → `initStateTest()` → `OceanState::exchangeHalo()` → `exchangeFullArrayHalo`
- **Fix:** Same as testDriver - fix OceanState initialization

---

**Test: testTendencies.exe**
- **Location:** Similar pattern to other tests
- **Fix:** Initialize tendency Views before halo exchange

---

### VertCoord Specific Issue

**Test:** testTendencyTermsSphere.exe
**Function:** `OMEGA::VertCoord::minMaxLayerEdge()`
**View Type:** `Kokkos::View<int*, Kokkos::LayoutRight, Kokkos::CudaSpace>`

```cpp
void VertCoord::minMaxLayerEdge(Halo* halo) {
    // The integer View for layer indices must be initialized
    Kokkos::View<int*> layerEdgeView(...);
    Kokkos::deep_copy(layerEdgeView, 0);  // Initialize

    // ... compute layer edge values ...

    halo->exchangeFullArrayHalo(layerEdgeView, MeshElement::Edge);
}
```

---

## Issue #2: Invalid Pointer Query in MPI GPU Layer (LOW PRIORITY)

### Problem Description
The Cray MPI GPU Transport Layer calls `cuPointerGetAttribute()` to determine if a buffer is GPU memory. When passed host-allocated memory (not from CUDA APIs), this returns `CUDA_ERROR_INVALID_VALUE`. This is expected behavior - the MPI library falls back to CPU communication paths.

### Error Signature
```
========= Program hit CUDA_ERROR_INVALID_VALUE (error 1) due to "invalid argument"
=========     on CUDA API call to cuPointerGetAttribute.
```

### Technical Details

#### When This Occurs
- During `Pacer::initialize()` when calling `MPI_Comm_dup`
- During `OMEGA::MachEnv::init()` / `MachEnv::create()`
- During `OMEGA::Broadcast()` operations
- During collective operations: `MPI_Allreduce`, `MPI_Bcast`, `MPI_Barrier`

#### Call Flow
```
PMPI_Comm_dup / MPI_Bcast / MPI_Allreduce
    ↓
MPIR_CRAY_Allreduce / MPIR_CRAY_Bcast
    ↓
MPIR_gpu_can_use_reduce_kernel
    ↓
mpix_gtl_pointer_type (libmpi_gtl_cuda.so)
    ↓
gtlt_cuda_pointer_type
    ↓
cuPointerGetAttribute (libcuda.so) ← Returns CUDA_ERROR_INVALID_VALUE
```

### Recommended Actions

**Option 1: No Action Required (Recommended)**
These errors are benign. The MPI library handles the error gracefully and uses CPU communication paths. The application continues to function correctly.

**Option 2: Suppress in Sanitizer Output**
If the error volume obscures real issues, use sanitizer options to filter:
```bash
compute-sanitizer --error-exitcode 0 --print-limit 100 ./testApp
```

**Option 3: Use CUDA-Managed Memory for MPI Buffers (Performance Optimization)**
If you want to eliminate these probes and potentially improve performance:

```cpp
// Instead of:
int value;
MPI_Bcast(&value, 1, MPI_INT, root, comm);

// Use CUDA managed memory:
int* value;
cudaMallocManaged(&value, sizeof(int));
*value = initialValue;
MPI_Bcast(value, 1, MPI_INT, root, comm);
cudaFree(value);
```

**Note:** This is only worthwhile for large, frequent communications where GPU-direct would benefit performance.

---

## Implementation Checklist for AI Coding Agent

### Phase 1: Core Halo Exchange Fix
- [ ] Locate `Halo.hpp` and `Halo.cpp` in OMEGA source
- [ ] Review `exchangeFullArrayHalo<T>()` implementation
- [ ] Add defensive initialization or validation option
- [ ] Consider adding a flag to skip initialization for performance-critical paths where caller guarantees initialization

### Phase 2: Fix Callers by Priority

**High Priority (most errors):**
- [ ] Fix `testHalo.exe` - `haloExchangeTest()` function
- [ ] Fix `testDriver.exe` - `OceanState::exchangeHalo()`
- [ ] Fix `testState.exe` - `initStateTest()` setup

**Medium Priority:**
- [ ] Fix `testTendencyTermsSphere.exe` - `VertCoord::minMaxLayerEdge()`
- [ ] Fix `testTendencyTermsPlane.exe` - `testThickFluxDiv()`
- [ ] Fix `testHorzOperatorsPlane.exe` - `setVectorEdge()`
- [ ] Fix `testHorzOperatorsSphere.exe` - `setVectorEdge()`

**Lower Priority:**
- [ ] Fix `testAuxiliaryState.exe` - `initState()`
- [ ] Fix `testAuxiliaryVarsPlane.exe` - `initState()`
- [ ] Fix `testAuxiliaryVarsSphere.exe` - `initState()`
- [ ] Fix `testTracers.exe` - `Tracers::updateTimeLevels()`
- [ ] Fix `testTendencies.exe`

### Phase 3: Validation
- [ ] Re-run sanitizer tests after fixes
- [ ] Verify zero initcheck errors
- [ ] Confirm application correctness with unit tests

---

## Code Pattern Reference

### Pattern 1: Initialize View at Creation
```cpp
// Best practice: Initialize when creating the View
Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::CudaSpace> myView("myView", dim1, dim2);
Kokkos::deep_copy(myView, 0.0);
```

### Pattern 2: Initialize Before Halo Exchange
```cpp
// If View is created elsewhere, initialize before exchange
void setupAndExchange(Halo* halo, ViewType& view) {
    // Initialize the entire view
    Kokkos::deep_copy(view, 0.0);

    // Populate interior points with computed values
    Kokkos::parallel_for("ComputeInterior",
        Kokkos::RangePolicy<>(0, nInterior),
        KOKKOS_LAMBDA(int i) {
            view(i, 0) = computeValue(i);
        });

    // Now safe to exchange - all memory is initialized
    halo->exchangeFullArrayHalo(view, element);
}
```

### Pattern 3: Initialize Only Halo Regions (Performance Optimization)
```cpp
// If interior is already initialized, only initialize halo regions
void initializeHaloRegions(ViewType& view, const HaloInfo& haloInfo) {
    // Initialize only the halo portion that will be sent/received
    for (const auto& region : haloInfo.sendRegions) {
        auto subview = Kokkos::subview(view, region.range);
        Kokkos::deep_copy(subview, 0.0);
    }
}
```

### Pattern 4: Constructor Initialization
```cpp
class OceanState {
public:
    OceanState(int nCells, int nLevels, int nTimeLevels) {
        for (int t = 0; t < nTimeLevels; ++t) {
            layerThickness[t] = View<double**>("thickness", nCells, nLevels);
            Kokkos::deep_copy(layerThickness[t], 0.0);  // Initialize

            normalVelocity[t] = View<double**>("velocity", nEdges, nLevels);
            Kokkos::deep_copy(normalVelocity[t], 0.0);  // Initialize
        }
    }
};
```

---

## File Search Hints

To locate the relevant source files, search for:

```bash
# Find Halo implementation
find . -name "*Halo*" -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" \)

# Find OceanState implementation
find . -name "*OceanState*" -o -name "*State*" | grep -E "\.(cpp|hpp|h)$"

# Find Tracers implementation
find . -name "*Tracer*" -type f \( -name "*.cpp" -o -name "*.hpp" \)

# Find VertCoord implementation
find . -name "*VertCoord*" -type f \( -name "*.cpp" -o -name "*.hpp" \)

# Find test files
find . -path "*/test/*" -name "*.cpp"
```

---

## Testing After Fixes

Run sanitizer checks to verify fixes:

```bash
# Run initcheck on specific test
compute-sanitizer --tool initcheck ./testHalo.exe

# Run with reduced output for quick verification
compute-sanitizer --tool initcheck --print-limit 10 ./testHalo.exe

# Expected output after fix:
# ========= COMPUTE-SANITIZER
# ========= ERROR SUMMARY: 0 errors
```

---

## Appendix: Error Counts by Test (Before Fix)

| Test Application | MPI Rank(s) with Errors | Initcheck Errors | Memcheck Errors |
|-----------------|------------------------|------------------|-----------------|
| testHalo.exe | 1, 2, 5, 6 | 42,420 | 4,654 |
| testDriver.exe | 1, 2, 5, 6 | 40,560 | - |
| testState.exe | 1, 2, 5, 6 | 10,815 | - |
| testTendencyTermsSphere.exe | 0, 2, 3, 4, 6, 7 | 10,517 | 988,620 |
| testTendencyTermsPlane.exe | 0, 1, 3, 4, 5, 7 | 5,432 | - |
| testTracers.exe | 1, 2, 5, 6 | 3,150 | - |
| testAuxiliaryState.exe | 1, 2, 5, 6 | 3,120 | 2,049,022 |
| testHorzOperatorsPlane.exe | 0, 1, 3, 4, 5, 7 | 1,032 | - |
| testHorzOperatorsSphere.exe | 0, 2, 3, 4, 6, 7 | - | 988,332 |
| testAuxiliaryVarsPlane.exe | 0, 1, 3, 4, 5, 7 | 1,252 | - |
| testAuxiliaryVarsSphere.exe | 0, 2, 3, 4, 6, 7 | - | - |
| testBroadcast.exe | 0, 1 | - | 244 |
| testDataTypes.exe | 0-7 | - | 164 |
| testConfig.exe | 0 | - | - |

**Note:** Missing MPI ranks in log files indicate those ranks had zero errors.

---

## Document Version
- **Created:** 2026-02-12
- **Based on:** NVIDIA Compute Sanitizer logs from Feb 9, 2026
- **CUDA Version:** 12.4 (NVIDIA HPC SDK 24.5)
- **MPI:** Cray MPICH with GPU Transport Layer (libmpi_gtl_cuda.so)
- **Build:** GNU GPU build (gnugpu)

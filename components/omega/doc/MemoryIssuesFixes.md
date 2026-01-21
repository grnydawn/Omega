# Memory Issues and Fixes

This document describes the memory issues identified during a code review of the `components/omega` folder and the fixes that were implemented.

## Summary

A comprehensive code review was conducted to identify potential memory leaks and resource management issues. The review identified several issues that have been addressed.

---

## Issue 1: Memory Leaks in IOStream.cpp

### Description
Three convenience overload functions in `IOStream.cpp` allocated `Clock` objects on the heap using `new` but never freed them, causing memory leaks every time these functions were called.

### Affected Locations
| File | Line | Function |
|------|------|----------|
| `src/infra/IOStream.cpp` | 77 | `IOStream::init(void)` |
| `src/infra/IOStream.cpp` | 116 | `IOStream::finalize(void)` |
| `src/infra/IOStream.cpp` | 272 | `IOStream::read(const std::string&)` |

### Original Code
```cpp
void IOStream::init(void) {
   Clock *ModelClock = new Clock;  // Memory leak!
   init(ModelClock);
}
```

### Fix Applied
Changed heap-allocated Clock objects to stack-allocated objects that are automatically destroyed when the function returns:

```cpp
void IOStream::init(void) {
   Clock ModelClock;  // Stack allocated - no leak
   init(&ModelClock);
}
```

### Files Modified
- `src/infra/IOStream.cpp`

---

## Issue 2: Missing Singleton Cleanup in OceanFinal.cpp

### Description
The `ocnFinalize()` function was missing cleanup calls for several singleton instances, which would result in memory not being properly freed during shutdown.

### Missing Cleanup Calls
| Singleton | Destroy Function | Created In |
|-----------|------------------|------------|
| `Eos` | `Eos::destroyInstance()` | `src/ocn/Eos.cpp` |
| `VertMix` | `VertMix::destroyInstance()` | `src/ocn/VertMix.cpp` |

### Fix Applied
Added the missing singleton cleanup calls to `ocnFinalize()`:

```cpp
// Destroy singleton instances
Eos::destroyInstance();
VertMix::destroyInstance();
```

### Files Modified
- `src/ocn/OceanFinal.cpp`

---

## Issue 3: LogFileStream Never Closed

### Description
The `LogFileStream` (a static `std::ofstream`) was opened in `initLogging()` to redirect stdout/stderr to the log file, but there was no corresponding close operation. While the OS reclaims resources on process termination, this is not clean shutdown behavior.

### Affected Location
| File | Line | Operation |
|------|------|-----------|
| `src/infra/Logging.cpp` | 193 | `LogFileStream.open(...)` |

### Fix Applied
Created a new `finalizeLogging()` function that properly closes the log file stream:

```cpp
void finalizeLogging() {
   if (LogFileStream.is_open()) {
      LogFileStream.flush();
      LogFileStream.close();
   }
   // Note: We do NOT call spdlog::shutdown() here because logging may still
   // be needed after ocean model finalization (e.g., in test drivers).
   // spdlog will be cleaned up automatically at program exit.
}
```

Added the function declaration to `Logging.h` and called it from `ocnFinalize()`.

**Note:** We intentionally do not call `spdlog::shutdown()` because test drivers and other code may still need to log messages after the ocean model has finalized.

### Files Modified
- `src/infra/Logging.h` - Added function declaration
- `src/infra/Logging.cpp` - Added function implementation
- `src/ocn/OceanFinal.cpp` - Added call to `finalizeLogging()`

---

## Issue 4: Duplicate VertCoord::clear() Call

### Description
The `ocnFinalize()` function called `VertCoord::clear()` twice (at lines 39 and 43), which is redundant and wasteful.

### Original Code
```cpp
VertCoord::clear();   // First call (line 39)
Dimension::clear();
Field::clear();
HorzMesh::clear();
VertCoord::clear();   // Duplicate call (line 43)
```

### Fix Applied
Removed the duplicate call, keeping only one `VertCoord::clear()` in the proper cleanup sequence.

### Files Modified
- `src/ocn/OceanFinal.cpp`

---

## Issue 5: IOStream Not Finalized

### Description
The `IOStream::finalize()` function was not being called during shutdown, which means I/O streams may not be properly flushed and closed.

### Fix Applied
Added `IOStream::finalize()` call to `ocnFinalize()`:

```cpp
// Clean up I/O streams and logging
IOStream::finalize();
finalizeLogging();
```

### Files Modified
- `src/ocn/OceanFinal.cpp`

---

## Issue 6: Variable Shadowing Bug in IO.cpp

### Description
In `IO::init()`, a local variable `DefaultFileFmt` was declared that shadowed the global variable of the same name. This meant the global `DefaultFileFmt` was never updated from the configuration file.

### Affected Location
| File | Line | Issue |
|------|------|-------|
| `src/base/IO.cpp` | 196 | Local variable shadows global |

### Original Code
```cpp
FileFmt DefaultFileFmt = FileFmtFromString(InFileFmt);  // Creates LOCAL variable
```

### Fix Applied
Removed the type declaration to assign to the global variable instead:

```cpp
DefaultFileFmt = FileFmtFromString(InFileFmt);  // Assigns to GLOBAL variable
```

### Files Modified
- `src/base/IO.cpp`

---

## Issue 7: Missing IO::finalize() Function

### Description
The PIO (Parallel I/O) system was never properly finalized during shutdown. This could cause resource leaks and potential issues if the IO system needed to be re-initialized.

### Fix Applied
Added a new `IO::finalize()` function to properly clean up PIO resources:

```cpp
void finalize() {
   if (SysID > 0) {
      int PIOErr = PIOc_finalize(SysID);
      if (PIOErr != PIO_NOERR)
         LOG_WARN("IO::finalize: Error finalizing PIO system");
      SysID = 0;
   }
}
```

Added the function declaration to `IO.h` and called it from `ocnFinalize()`.

### Files Modified
- `src/base/IO.h` - Added function declaration
- `src/base/IO.cpp` - Added function implementation
- `src/ocn/OceanFinal.cpp` - Added call to `IO::finalize()`

---

## Issue 8: IOStream Uses Wrong File Format Constant

### Description
In `IOStream.cpp`, the `readStream()` and `writeStream()` functions were using `IO::FmtDefault` (a hardcoded enum constant equal to `PIO_IOTYPE_NETCDF4C`) instead of `IO::DefaultFileFmt` (the configurable global variable that gets set from the configuration file).

This meant that even if the user specified a different file format in the configuration (e.g., `IODefaultFormat: NetCDF4`), the streams would always use `NetCDF4c` (compressed).

### Affected Locations
| File | Line | Function |
|------|------|----------|
| `src/infra/IOStream.cpp` | 2290 | `IOStream::readStream()` |
| `src/infra/IOStream.cpp` | 2443 | `IOStream::writeStream()` |

### Original Code
```cpp
IO::openFile(InFileID, InFileName, Mode, IO::FmtDefault, ExistAction);
```

### Fix Applied
Changed to use the configurable global variable:
```cpp
IO::openFile(InFileID, InFileName, Mode, IO::DefaultFileFmt, ExistAction);
```

### Files Modified
- `src/infra/IOStream.cpp`

---

## Issue 9: Missing SysID Validation and Debug Logging

### Description
The PIO system ID (`SysID`) is critical for all PIO operations. If `IO::init()` is not called before other IO functions, or if `IO::finalize()` has already been called, the SysID would be invalid (0 or negative), leading to cryptic PIO errors like "invalid IO type".

### Fix Applied
1. Added debug logging to `IO::init()`, `IO::openFile()`, and `IO::finalize()` to trace SysID values
2. Added SysID validation checks in `IO::openFile()`, `IO::createDecomp()`, and `IO::destroyDecomp()` with clear error messages

Example validation:
```cpp
if (SysID <= 0) {
   ABORT_ERROR("IO::openFile: Invalid SysID ({}) - IO system not initialized "
               "or already finalized", SysID);
}
```

### Files Modified
- `src/base/IO.cpp`

---

## Positive Patterns Observed

The codebase generally follows good memory management practices:

1. **Smart Pointers**: Most container classes use `std::unique_ptr` or `std::shared_ptr`:
   - `AllTimeSteppers`, `AllDecomps`, `AllHalos` use `std::unique_ptr`
   - `AllFields` uses `std::shared_ptr`

2. **RAII File Streams**: Most file operations use `std::ifstream`/`std::ofstream` which are RAII-compliant.

3. **Virtual Destructors**: Base classes like `TimeStepper` properly declare virtual destructors.

4. **Kokkos Views**: The codebase uses Kokkos views (`Array2DReal`, `Array3DReal`) which have automatic memory management through reference counting.

5. **No C-style Allocation**: No `malloc/free` usage found in the Omega source code.

---

## Recommendations for Future Development

1. **Avoid Raw `new`**: Prefer stack allocation or smart pointers over raw `new` for objects with function-local scope.

2. **Singleton Pattern**: Consider using `std::unique_ptr` for singleton instances instead of raw pointers to ensure automatic cleanup even if `destroyInstance()` is not called.

3. **RAII Consistency**: Always ensure resources opened in `init()` functions have corresponding cleanup in `finalize()` or `clear()` functions.

4. **Code Review Checklist**: Add memory leak checks to the code review checklist, specifically looking for:
   - Unmatched `new`/`delete` pairs
   - File streams opened but not closed
   - Singleton instances not destroyed
   - Duplicate cleanup calls

---

## Files Changed Summary

| File | Changes |
|------|---------|
| `src/infra/IOStream.cpp` | Fixed 3 memory leaks (lines 77, 116, 272); Fixed FmtDefault vs DefaultFileFmt bug |
| `src/infra/Logging.h` | Added `finalizeLogging()` declaration |
| `src/infra/Logging.cpp` | Added `finalizeLogging()` implementation |
| `src/ocn/OceanFinal.cpp` | Added singleton cleanup, IOStream/IO/Logging finalize, removed duplicate call |
| `src/base/IO.cpp` | Fixed variable shadowing bug, added `IO::finalize()`, added SysID validation and debug logging |
| `src/base/IO.h` | Added `IO::finalize()` declaration |

---

*Document created: January 2026*
*Code review performed on: components/omega/src*

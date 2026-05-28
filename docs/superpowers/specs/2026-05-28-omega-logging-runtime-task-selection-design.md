# Omega logging: runtime `OMEGA_LOG_TASKS` selection

**Date:** 2026-05-28
**Branch:** `ykim/omega/coupled-logging`
**Target files:** `components/omega/src/infra/Logging.h`, `components/omega/src/infra/Logging.cpp`, `components/omega/test/infra/LoggingTest.cpp`, `components/omega/OmegaBuild.cmake`, `components/omega/doc/design/Logging.md`, `components/omega/doc/devGuide/CMakeBuild.md`

## Goal

Replace the CMake-baked MPI rank selection for logging with a selector that is sourced at runtime and resolved against the Omega MPI sub-communicator (`MachEnv`). `OMEGA_LOG_TASKS` keeps its name but becomes a runtime selection *policy* rather than a build-time fixed rank list.

## Background / current state

- `OMEGA_LOG_TASKS` is baked into the binary by `OmegaBuild.cmake` (`string(TOUPPER ...)` + `add_definitions(-DOMEGA_LOG_TASKS=...)`), then stringified and parsed at runtime by `splitTasks()` in `Logging.cpp`.
- `splitTasks()` already compares against `DefEnv->getMyTask()` / `getNumTasks()`, so rank IDs are already local to the env's communicator. What is genuinely build-time-fixed is the **policy string** (changing it requires a rebuild), and there is **no "master" concept** (it cannot resolve `getMasterTask()`).
- `MachEnv` exposes `getMyTask()`, `getNumTasks()`, `getMasterTask()`, `isMasterTask()`, `getComm()`.
- stdout/stderr redirection (`std::cout`/`std::cerr` rdbuf swap) lives **only** in the `(DefEnv, LogFilePath)` overload, not the `(DefEnv, Logger)` overload the unit test uses.
- Nearly all callers invoke `initLogging(DefEnv)` (file-path overload, default path) and ignore the return value.
- No `getenv` usage exists in `omega/src` today. A `Config` system exists but initializes after logging, so it is not a viable source here.

## Decisions (locked with user)

1. **Value source:** read `OMEGA_LOG_TASKS` from the environment at init via `std::getenv`. If unset, fall back to a compile-time default string macro set by CMake. The name `OMEGA_LOG_TASKS` is kept for both the build option and the env var.
2. **Selector forms:** `*`, `m`/`master`, single `<number>`, comma lists (`0,2,4`), dash ranges (`0-3`), and combinations (`0,2-3`).
3. **Malformed selector:** print a clear warning to stderr **on the master rank only**, then fall back to **master-rank-only** logging and continue.
4. **Out-of-range rank:** allowed; it simply matches no rank (nobody logs for it). Optional single master-side note if the net selection logs on nobody.
5. **Default when env unset:** `master`.
6. **Docs:** update `doc/design/Logging.md` and `doc/devGuide/CMakeBuild.md` to match.

## Selector grammar & semantics

Resolved against the Omega MPI sub-communicator via `MachEnv`:

| Selector | Meaning | Resolved from |
|---|---|---|
| `*` | all ranks | `getNumTasks()` -> `[0, N)` |
| `m` or `master` | master rank only | `getMasterTask()` |
| `<n>` | single rank `n` | literal |
| `0,2,4` | comma list | literal set (deduped) |
| `0-3` | inclusive range | expanded |
| `0,2-3` | list + ranges | union |

Rules:

- Case-insensitive; surrounding whitespace trimmed.
- Keywords (`*`, `master`/`m`) must stand **alone**. Mixing a keyword into a numeric list (e.g. `master,0`, `*,2`) is **malformed**.
- A numeric expression is a non-empty comma list of tokens, each either a single non-negative integer or an inclusive range `a-b` with `a <= b`.
- **Malformed** examples: `foo`, `1,,2` (empty token), `3-` / `-2` (incomplete range), `5-2` (reversed range), negative values, non-numeric tokens. Malformed -> warn on master + fall back to `{MasterTask}`.
- **Out-of-range** (token `>= NumTasks`): kept in the parsed set but matches no rank; treated as valid (not malformed). If the net in-range selection is empty, optionally emit one master-side note that logging is disabled.

## Code structure

### `Logging.h`

- Replace the `#ifndef OMEGA_LOG_TASKS / #define OMEGA_LOG_TASKS 0` block with a quoted-string default macro:
  ```cpp
  #ifndef OMEGA_LOG_TASKS_DEFAULT
  #define OMEGA_LOG_TASKS_DEFAULT "master"
  #endif
  ```
- Declare the parser in namespace `OMEGA` (following the existing `_PackLogMsg` precedent so the unit test can call it directly):
  ```cpp
  std::vector<int> _selectLogTasks(const std::string &Selector, // [in] raw selector
                                   I4 NumTasks,    // [in] sub-communicator size
                                   I4 MasterTask,  // [in] sub-communicator master rank
                                   bool &Valid);   // [out] false if selector malformed
  ```
- Update the `OMEGA_LOG_TASKS` doc comment to describe the runtime/env-var behavior and the new syntax.

### `Logging.cpp`

- Remove `splitTasks()` and the now-unused `_OMEGA_STRINGIFY` / `_OMEGA_TOSTRING` macros.
- Add `_selectLogTasks()` implementing the grammar above. On malformed input it sets `Valid = false` and returns `{MasterTask}`.
- Add a small internal helper to source the raw selector string: `std::getenv("OMEGA_LOG_TASKS")`, else `OMEGA_LOG_TASKS_DEFAULT`.
- Both `initLogging` overloads share the same flow:
  1. Get selector string (env or default).
  2. `Tasks = _selectLogTasks(Selector, NumTasks, MasterTask, Valid)`.
  3. If `!Valid` and this is the master rank, print to stderr: a clear message naming the offending selector and stating the master-only fallback. (`_selectLogTasks` has already returned `{MasterTask}`.)
  4. `ThisTaskLogs = Tasks contains TaskId`.
  5. For the multi-file naming decision, compute the count of selected ranks that are **in range** `[0, NumTasks)`; append `_<TaskId>` only when that count > 1 (preserves current behavior, ignoring out-of-range padding).
- The stderr warning is emitted **before** the stdout/stderr redirect block in the file-path overload, so it reaches the real terminal.
- Preserve unchanged: spdlog logger/pattern/level/flush setup, the `_<TaskId>` filename-suffix scheme, and the `std::cout`/`std::cerr` redirect mechanism in the file-path overload.

### `OmegaBuild.cmake`

Replace the existing block:
```cmake
if(OMEGA_LOG_TASKS)
  string(TOUPPER "${OMEGA_LOG_TASKS}" _LOG_TASKS)
  add_definitions(-DOMEGA_LOG_TASKS=${_LOG_TASKS})
endif()
```
with:
```cmake
if(NOT OMEGA_LOG_TASKS)
  set(OMEGA_LOG_TASKS "master")
endif()
add_definitions(-DOMEGA_LOG_TASKS_DEFAULT="${OMEGA_LOG_TASKS}")
```
The build option name `OMEGA_LOG_TASKS` is unchanged; it now sets the compiled-in default, which the runtime env var of the same name overrides. `TOUPPER` is dropped (case handled at runtime).

## Tests (`LoggingTest.cpp`)

Add direct unit tests for `_selectLogTasks` (no MPI gymnastics; runs on a single rank). For a fixed `NumTasks` and `MasterTask`, assert the resolved rank vector and `Valid` flag:

- `*` -> all ranks `[0, NumTasks)`, valid.
- `master` and `m` -> `{MasterTask}`, valid (verify with a non-zero `MasterTask`).
- `2` -> `{2}`, valid.
- `0,2,4` -> `{0,2,4}`, valid.
- `0-3` -> `{0,1,2,3}`, valid.
- `0,2-3` -> `{0,2,3}`, valid.
- Out-of-range `99` (NumTasks=4) -> parsed but no in-range match (logging would be disabled), valid.
- Malformed: `foo`, `1,,2`, `3-`, `5-2`, `master,0` -> `Valid == false` and returns `{MasterTask}`.

Keep the existing log-level and Kokkos data-type tests intact. Existing call sites that use `initLogging(DefEnv)` keep working: with the env var unset the default is `master`, which in standalone `MPI_COMM_WORLD` is local rank 0 — matching today's default behavior.

## Docs

- `doc/design/Logging.md` §4.1.1: replace the compile-time `-D OMEGA_LOG_TASKS=0,2-3` description with the runtime env-var selector and the `*` / `master` / number / list / range syntax; note the CMake option sets the default and the env var overrides at runtime.
- `doc/devGuide/CMakeBuild.md`: update the `OMEGA_LOG_TASKS` line to state the default is `master` and that it sets the compiled-in default overridable by the `OMEGA_LOG_TASKS` env var.

## Explicitly NOT changing

spdlog configuration, log levels, formatters (`LogFormatters.h`), the stdout/stderr redirect mechanism itself, and the multi-file naming scheme.

## Success criteria

- Rank selection no longer depends on a CMake-baked rank list; it is computed at runtime from `MachEnv` (`getNumTasks`, `getMasterTask`, `getMyTask`) and an env-var-sourced selector.
- `*`, `m`/`master`, single number, comma lists, and ranges all resolve correctly against the sub-communicator.
- Malformed selectors warn on master and fall back to master-only; out-of-range ranks are allowed and simply log on nobody.
- stdout/stderr redirect behavior is unchanged.
- `LoggingTest.cpp` covers the new selector behavior and existing tests still pass.
- If local build/tests require the remote machine, changes are committed for manual verification there.

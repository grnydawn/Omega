# Omega Logging Runtime Task Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace CMake-baked MPI rank selection for Omega logging with a runtime, environment-variable-sourced `OMEGA_LOG_TASKS` selector resolved against the Omega MPI sub-communicator (`MachEnv`).

**Architecture:** A new pure function `OMEGA::_selectLogTasks(Selector, NumTasks, MasterTask, Valid)` parses the selector string (`*`, `m`/`master`, single ranks, comma lists, dash ranges) into a sorted, deduplicated rank vector. Both `initLogging` overloads source the selector at runtime via `std::getenv("OMEGA_LOG_TASKS")` (falling back to the compile-time string macro `OMEGA_LOG_TASKS_DEFAULT`, default `"master"`), then use `_selectLogTasks` to decide whether the local rank logs. Malformed selectors warn on the master rank and fall back to master-only.

**Tech Stack:** C++17, spdlog, MPI, Kokkos, CMake/CTest. Builds and tests run in the configured Omega build environment.

---

## BUILD/TEST ENVIRONMENT NOTE

Omega is built out-of-source via CIME or the standalone Omega CMake build and requires a supported machine (MPI + Kokkos + spdlog). The exact build directory is environment-specific. The Logging unit test is registered as CTest target **`LOGGING_TEST`** (`testLogging.exe`, launched with 8 MPI ranks).

Per the spec's success criteria: **if the build/tests cannot run locally and require the remote machine, complete all edits and commit each task so they can be verified manually on the remote system.** Where a step says "build & run", run it in the Omega build directory:

```bash
# Reconfigure/build (machine-specific build dir), then:
ctest -R LOGGING_TEST --output-on-failure
```

If you cannot build locally, still perform every edit and commit, and record in the commit that remote verification is pending.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `components/omega/src/infra/Logging.h` | Public logging API + macros | Add `OMEGA_LOG_TASKS_DEFAULT` macro, declare `_selectLogTasks`, remove old `OMEGA_LOG_TASKS` macro |
| `components/omega/src/infra/Logging.cpp` | Logging init + selector parsing | Add `_selectLogTasks` + `getLogTaskSelector`; rewire both `initLogging` overloads; remove `splitTasks` + stringify macros |
| `components/omega/test/infra/LoggingTest.cpp` | Logging unit test | Add `testTaskSelection()` covering the selector grammar |
| `components/omega/OmegaBuild.cmake` | Build configuration | Replace `OMEGA_LOG_TASKS` baking with `OMEGA_LOG_TASKS_DEFAULT` string default |
| `components/omega/doc/design/Logging.md` | Design doc | Update §4.1.1 task-selection description |
| `components/omega/doc/devGuide/CMakeBuild.md` | Build doc | Update `OMEGA_LOG_TASKS` line |

All paths below are relative to the repository root `/home/youngsung/repos/github/Omega`.

---

## Task 1: Add and test the `_selectLogTasks` parser

This task adds the parser and its unit tests **without** touching the existing `initLogging` behavior (the old `splitTasks` path stays in place), so the build stays green and behavior is unchanged.

**Files:**
- Modify: `components/omega/src/infra/Logging.h` (add default macro + declaration)
- Modify: `components/omega/src/infra/Logging.cpp` (add parser + includes)
- Test: `components/omega/test/infra/LoggingTest.cpp` (add `testTaskSelection`)

- [ ] **Step 1: Declare the parser and add the default macro in `Logging.h`**

In `components/omega/src/infra/Logging.h`, immediately AFTER the existing `OMEGA_LOG_TASKS` block (the lines ending with `#define OMEGA_LOG_TASKS 0` / `#endif`, around line 143), insert the new default-string macro with its doc comment:

```cpp
/// \def OMEGA_LOG_TASKS_DEFAULT
/// Compile-time default for the logging task selector, set by CMake from the
/// OMEGA_LOG_TASKS build option (default "master"). At runtime this default is
/// overridden by the OMEGA_LOG_TASKS environment variable when it is set. The
/// selector is resolved against the Omega MPI sub-communicator and accepts:
///   "*"            - all ranks in the sub-communicator
///   "m" / "master" - the sub-communicator master rank only
///   "<n>"          - a single rank
///   "0,2,4"        - a comma-separated list of ranks
///   "0-3"          - an inclusive range of ranks
///   "0,2-3"        - any combination of lists and ranges
/// An invalid selector logs a warning on the master rank and falls back to
/// master-rank-only logging.
#ifndef OMEGA_LOG_TASKS_DEFAULT
#define OMEGA_LOG_TASKS_DEFAULT "master"
#endif
```

Then, inside `namespace OMEGA`, immediately AFTER the `_PackLogMsg(...)` declaration (the block ending around line 172, just before `} // namespace OMEGA`), add the parser declaration:

```cpp
/// Resolve a logging task selector string into the sorted, deduplicated list
/// of MPI ranks (relative to the Omega sub-communicator) that should log.
/// Accepts "*", "m"/"master", a single rank, comma lists, dash ranges, or any
/// combination of lists and ranges. On a malformed selector, sets Valid=false
/// and returns {MasterTask}.
std::vector<int>
_selectLogTasks(const std::string &Selector, ///< [in] raw selector string
                I4 NumTasks,   ///< [in] number of tasks in the sub-communicator
                I4 MasterTask, ///< [in] master rank of the sub-communicator
                bool &Valid    ///< [out] false if the selector was malformed
);
```

- [ ] **Step 2: Add includes and the parser implementation in `Logging.cpp`**

In `components/omega/src/infra/Logging.cpp`, add these includes after the existing `#include` lines (after `#include <spdlog/sinks/basic_file_sink.h>`):

```cpp
#include <algorithm>
#include <cctype>
#include <limits>
#include <set>
#include <sstream>
#include <vector>
```

Then, inside `namespace OMEGA`, immediately AFTER the `_PackLogMsg` function definition (around line 39, before the `splitTasks` function), add the two static helpers and the parser:

```cpp
//------------------------------------------------------------------------------
// Trim surrounding whitespace from a string
static std::string trimStr(const std::string &Str) {
   const std::string WhiteSpace = " \t\n\r\f\v";
   std::size_t Begin            = Str.find_first_not_of(WhiteSpace);
   if (Begin == std::string::npos)
      return "";
   std::size_t End = Str.find_last_not_of(WhiteSpace);
   return Str.substr(Begin, End - Begin + 1);
}

//------------------------------------------------------------------------------
// Parse a string as a non-negative int. Returns false if the string is empty,
// contains any non-digit character, or overflows an int.
static bool toNonNegInt(const std::string &Str, int &Out) {
   if (Str.empty())
      return false;
   for (char C : Str) {
      if (!std::isdigit(static_cast<unsigned char>(C)))
         return false;
   }
   try {
      std::size_t Pos = 0;
      long Val        = std::stol(Str, &Pos);
      if (Pos != Str.size() || Val < 0 ||
          Val > static_cast<long>(std::numeric_limits<int>::max()))
         return false;
      Out = static_cast<int>(Val);
      return true;
   } catch (...) {
      return false;
   }
}

//------------------------------------------------------------------------------
// Resolve a logging task selector string into the list of ranks that should log
std::vector<int> _selectLogTasks(const std::string &Selector,
                                 OMEGA::I4 NumTasks, OMEGA::I4 MasterTask,
                                 bool &Valid) {

   Valid = true;

   std::string Sel = trimStr(Selector);

   // Lowercase copy for case-insensitive keyword matching
   std::string Lower = Sel;
   std::transform(Lower.begin(), Lower.end(), Lower.begin(),
                  [](unsigned char C) { return std::tolower(C); });

   // Standalone keyword: all ranks in the sub-communicator
   if (Lower == "*") {
      std::vector<int> All;
      for (int I = 0; I < NumTasks; ++I)
         All.push_back(I);
      return All;
   }

   // Standalone keyword: master rank only
   if (Lower == "m" || Lower == "master")
      return std::vector<int>{MasterTask};

   // Empty selector is malformed
   if (Sel.empty()) {
      Valid = false;
      return std::vector<int>{MasterTask};
   }

   // Numeric expression: comma-separated single ranks and/or "a-b" ranges.
   // A std::set keeps the result sorted and deduplicated.
   std::set<int> Selected;
   std::stringstream Ss(Sel);
   std::string Token;
   while (std::getline(Ss, Token, ',')) {
      Token = trimStr(Token);

      std::size_t DashPos = Token.find('-');
      if (DashPos == std::string::npos) {
         int Rank;
         if (!toNonNegInt(Token, Rank)) {
            Valid = false;
            return std::vector<int>{MasterTask};
         }
         Selected.insert(Rank);
      } else {
         int Lo, Hi;
         if (!toNonNegInt(trimStr(Token.substr(0, DashPos)), Lo) ||
             !toNonNegInt(trimStr(Token.substr(DashPos + 1)), Hi) || Lo > Hi) {
            Valid = false;
            return std::vector<int>{MasterTask};
         }
         for (int I = Lo; I <= Hi; ++I)
            Selected.insert(I);
      }
   }

   return std::vector<int>(Selected.begin(), Selected.end());
}
```

- [ ] **Step 3: Write the parser unit test in `LoggingTest.cpp`**

In `components/omega/test/infra/LoggingTest.cpp`, add this function immediately BEFORE `int main(int argc, char **argv) {` (after `testKokkosDataTypes`). It uses a fixed `NumTasks`/`MasterTask` passed directly to the parser, so it is independent of the MPI launch size, and uses a non-zero `MasterTask` to prove master resolution:

```cpp
int testTaskSelection() {

   int RetVal = 0;

   const OMEGA::I4 NumTasks   = 4;
   const OMEGA::I4 MasterTask = 1; // non-zero to verify master resolution

   auto check = [&](const std::string &Name, const std::string &Selector,
                    const std::vector<int> &Expected, bool ExpectValid) {
      bool Valid;
      std::vector<int> Got =
          _selectLogTasks(Selector, NumTasks, MasterTask, Valid);
      if (Got == Expected && Valid == ExpectValid) {
         std::cout << Name << ": PASS" << std::endl;
      } else {
         std::cout << Name << ": FAIL" << std::endl;
         RetVal += 1;
      }
   };

   // Keywords
   check("Select all", "*", std::vector<int>{0, 1, 2, 3}, true);
   check("Select master", "master", std::vector<int>{1}, true);
   check("Select m", "m", std::vector<int>{1}, true);
   check("Select MASTER (case)", "MASTER", std::vector<int>{1}, true);

   // Numeric forms
   check("Select single", "2", std::vector<int>{2}, true);
   check("Select list", "0,2,4", std::vector<int>{0, 2, 4}, true);
   check("Select range", "0-3", std::vector<int>{0, 1, 2, 3}, true);
   check("Select list+range", "0,2-3", std::vector<int>{0, 2, 3}, true);
   check("Select with spaces", " 0, 2 ", std::vector<int>{0, 2}, true);

   // Out-of-range kept as-is (caller treats as nobody logging)
   check("Out of range", "99", std::vector<int>{99}, true);

   // Malformed -> Valid=false, fall back to master
   check("Malformed text", "foo", std::vector<int>{1}, false);
   check("Malformed empty token", "1,,2", std::vector<int>{1}, false);
   check("Malformed open range", "3-", std::vector<int>{1}, false);
   check("Malformed reversed range", "5-2", std::vector<int>{1}, false);
   check("Malformed keyword mix", "master,0", std::vector<int>{1}, false);

   return RetVal;
}
```

Then, in `main`, add a call to it on the master rank only (to avoid duplicate output across the 8 ranks). Insert it immediately AFTER the line `OMEGA::I4 TaskId = DefEnv->getMyTask();` (around line 249):

```cpp
   if (DefEnv->isMasterTask())
      RetVal += testTaskSelection();
```

- [ ] **Step 4: Build & run the test**

Run in the Omega build directory:

```bash
ctest -R LOGGING_TEST --output-on-failure
```

Expected: the test output (from rank 0) includes a `PASS` line for every `testTaskSelection` case, e.g.:

```
Select all: PASS
Select master: PASS
...
Malformed keyword mix: PASS
```

and the overall test result is **Passed** (existing log-level and Kokkos tests unchanged).

> If the build can only run on the remote machine, skip the local run, proceed to commit, and verify there. Before implementing Step 2, the test would fail to link (`undefined reference to OMEGA::_selectLogTasks`) — that is the expected pre-implementation red state.

- [ ] **Step 5: Commit**

```bash
git add components/omega/src/infra/Logging.h \
        components/omega/src/infra/Logging.cpp \
        components/omega/test/infra/LoggingTest.cpp
git commit -m "$(cat <<'EOF'
Add runtime OMEGA_LOG_TASKS selector parser

Add OMEGA::_selectLogTasks to resolve a selector string (*, m/master,
single ranks, comma lists, dash ranges) into a sorted rank list relative
to the Omega MPI sub-communicator, plus the OMEGA_LOG_TASKS_DEFAULT macro
and unit tests. initLogging still uses the old path; rewired next.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Source the selector at runtime and rewire `initLogging`

Now switch both `initLogging` overloads to source the selector from the environment and use `_selectLogTasks`, then remove the obsolete `splitTasks` function and stringify macros, and remove the old `OMEGA_LOG_TASKS` macro from the header.

**Files:**
- Modify: `components/omega/src/infra/Logging.cpp` (add `getLogTaskSelector`, rewire both overloads, remove `splitTasks` + stringify macros)
- Modify: `components/omega/src/infra/Logging.h` (remove old `OMEGA_LOG_TASKS` macro + doc)

- [ ] **Step 1: Add the runtime selector-source helper in `Logging.cpp`**

Add `#include <cstdlib>` to the includes added in Task 1 (for `std::getenv`).

Immediately AFTER the `_selectLogTasks` definition (and before the first `initLogging` definition), add:

```cpp
//------------------------------------------------------------------------------
// Source the logging task selector from the OMEGA_LOG_TASKS environment
// variable, falling back to the compile-time default when unset/empty.
static std::string getLogTaskSelector() {
   const char *Env = std::getenv("OMEGA_LOG_TASKS");
   if (Env != nullptr && Env[0] != '\0')
      return std::string(Env);
   return std::string(OMEGA_LOG_TASKS_DEFAULT);
}
```

- [ ] **Step 2: Rewire the custom-logger overload**

Replace the entire body of the `initLogging(const OMEGA::MachEnv *DefEnv, std::shared_ptr<spdlog::logger> Logger)` overload (currently the block from `int RetVal = 0;` through `return RetVal;`) with:

```cpp
   int RetVal = 0;

   OMEGA::I4 TaskId     = DefEnv->getMyTask();
   OMEGA::I4 NumTasks   = DefEnv->getNumTasks();
   OMEGA::I4 MasterTask = DefEnv->getMasterTask();

   // Determine which tasks log from the runtime selector
   std::string Selector = getLogTaskSelector();
   bool Valid;
   std::vector<int> Tasks =
       _selectLogTasks(Selector, NumTasks, MasterTask, Valid);

   if (!Valid && DefEnv->isMasterTask()) {
      std::cerr << "[Omega Logging] Invalid OMEGA_LOG_TASKS selector \""
                << Selector << "\"; falling back to master rank only."
                << std::endl;
   }

   // Count selected ranks that actually exist in this communicator
   int NumLogging = 0;
   for (int Rank : Tasks) {
      if (Rank >= 0 && Rank < NumTasks)
         ++NumLogging;
   }
   if (Valid && NumLogging == 0 && DefEnv->isMasterTask()) {
      std::cerr << "[Omega Logging] OMEGA_LOG_TASKS selector \"" << Selector
                << "\" matches no ranks in this communicator; logging disabled."
                << std::endl;
   }

   bool ThisTaskLogs =
       (std::find(Tasks.begin(), Tasks.end(), TaskId) != Tasks.end());

   if (ThisTaskLogs) {

      spdlog::set_default_logger(Logger);

      // set prefix format - here n is logger name, l is level and v is msg txt
      spdlog::set_pattern("[%n %l] %v");
      // set default log level
      spdlog::set_level(
          static_cast<spdlog::level::level_enum>(SPDLOG_ACTIVE_LEVEL));
      // flush output buffers for levels above warn
      spdlog::flush_on(spdlog::level::warn);

      RetVal = 1; // log enabled

   } else {
      spdlog::set_level(spdlog::level::off);
      RetVal = 0; // log disabled
   }

   return RetVal;
```

- [ ] **Step 3: Rewire the file-path overload**

Replace the entire body of the `initLogging(const OMEGA::MachEnv *DefEnv, std::string const &LogFilePath)` overload (from `int RetVal = 0;` through `return RetVal;`) with:

```cpp
   int RetVal = 0;

   OMEGA::I4 TaskId     = DefEnv->getMyTask();
   OMEGA::I4 NumTasks   = DefEnv->getNumTasks();
   OMEGA::I4 MasterTask = DefEnv->getMasterTask();
   std::string NewLogFilePath;

   // Determine which tasks log from the runtime selector
   std::string Selector = getLogTaskSelector();
   bool Valid;
   std::vector<int> Tasks =
       _selectLogTasks(Selector, NumTasks, MasterTask, Valid);

   if (!Valid && DefEnv->isMasterTask()) {
      std::cerr << "[Omega Logging] Invalid OMEGA_LOG_TASKS selector \""
                << Selector << "\"; falling back to master rank only."
                << std::endl;
   }

   // Count selected ranks that actually exist in this communicator
   int NumLogging = 0;
   for (int Rank : Tasks) {
      if (Rank >= 0 && Rank < NumTasks)
         ++NumLogging;
   }
   if (Valid && NumLogging == 0 && DefEnv->isMasterTask()) {
      std::cerr << "[Omega Logging] OMEGA_LOG_TASKS selector \"" << Selector
                << "\" matches no ranks in this communicator; logging disabled."
                << std::endl;
   }

   bool ThisTaskLogs =
       (std::find(Tasks.begin(), Tasks.end(), TaskId) != Tasks.end());

   if (ThisTaskLogs) {

      try {
         std::size_t dotPos = LogFilePath.find_last_of('.');

         // create log file name/path and set default (*) logger. When more than
         // one rank logs, append the rank id to keep per-rank files distinct.
         if (NumLogging > 1 && dotPos != std::string::npos) {
            NewLogFilePath = LogFilePath.substr(0, dotPos) + "_" +
                             std::to_string(TaskId) +
                             LogFilePath.substr(dotPos);
         } else {
            NewLogFilePath = LogFilePath;
         }

         // Create default logger
         spdlog::set_default_logger(
             spdlog::basic_logger_mt("*", NewLogFilePath));

         // Set the prefix for all messages l is log level and v is msg txt
         spdlog::set_pattern("[%l] %v");
         // Set the default log level based on cpp input
         spdlog::set_level(
             static_cast<spdlog::level::level_enum>(SPDLOG_ACTIVE_LEVEL));
         // Flush the message buffer for all messages above warning level
         spdlog::flush_on(spdlog::level::warn);

         RetVal = 1; // log enabled

      } catch (spdlog::spdlog_ex const &Ex) {
         std::cout << "Log init failed: " << Ex.what() << std::endl;
         RetVal = -1; // error occured
      }

   } else {
      spdlog::set_level(spdlog::level::off);
      RetVal = 0; // log disabled
   }

   // If logging is successful, also redirect stdout and stderr to log file
   if (RetVal == 1) {
      // Open an output filestream to the new log file defined above
      LogFileStream.open(NewLogFilePath, std::ios::app);

      // Set the stdout and stderr buffers to the file streambuffer
      std::cout.rdbuf(LogFileStream.rdbuf());
      std::cerr.rdbuf(LogFileStream.rdbuf());
   }

   return RetVal;
```

- [ ] **Step 4: Remove the obsolete `splitTasks` function and stringify macros from `Logging.cpp`**

Delete the entire `splitTasks` function definition (the block starting with its `//---` comment `Utility function to determine which tasks perform logging...` through its closing `}` and trailing `//---` separator).

Delete the now-unused macros near the top of the file:

```cpp
#define _OMEGA_STRINGIFY(x) #x
#define _OMEGA_TOSTRING(x)  _OMEGA_STRINGIFY(x)
```

- [ ] **Step 5: Remove the old `OMEGA_LOG_TASKS` macro from `Logging.h`**

Delete the original `OMEGA_LOG_TASKS` doc comment and macro block (the `/// \def OMEGA_LOG_TASKS` comment through its `#ifndef OMEGA_LOG_TASKS` / `#define OMEGA_LOG_TASKS 0` / `#endif`). Keep the `OMEGA_LOG_TASKS_DEFAULT` block added in Task 1.

- [ ] **Step 6: Build & run the test**

Run in the Omega build directory:

```bash
ctest -R LOGGING_TEST --output-on-failure
```

Expected: **Passed**. With `OMEGA_LOG_TASKS` unset, the default is `master`; in standalone `MPI_COMM_WORLD` the master is rank 0, so only rank 0 logs — identical to the previous default behavior. The `testTaskSelection` PASS lines still appear, and the existing log-level/Kokkos assertions still pass on rank 0.

Optional manual runtime check on the remote machine (exercises env sourcing across the 8 ranks):

```bash
# All ranks log to per-rank files tmplog_<rank>.log
OMEGA_LOG_TASKS='*' ctest -R LOGGING_TEST --output-on-failure
# Invalid selector: warning on master + master-only fallback, still Passed
OMEGA_LOG_TASKS='bogus' ctest -R LOGGING_TEST --output-on-failure
```

> If building only on the remote machine, commit now and verify there.

- [ ] **Step 7: Commit**

```bash
git add components/omega/src/infra/Logging.h \
        components/omega/src/infra/Logging.cpp
git commit -m "$(cat <<'EOF'
Use runtime OMEGA_LOG_TASKS selector in initLogging

Source the selector from the OMEGA_LOG_TASKS environment variable
(falling back to OMEGA_LOG_TASKS_DEFAULT) and resolve it with
_selectLogTasks against the Omega MPI sub-communicator in both
initLogging overloads. Warn on the master rank for invalid selectors and
fall back to master-only. Remove the obsolete splitTasks function and
stringify macros. stdout/stderr redirect behavior is unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Switch CMake to a default-string macro

**Files:**
- Modify: `components/omega/OmegaBuild.cmake:577-580`

- [ ] **Step 1: Replace the `OMEGA_LOG_TASKS` definition block**

In `components/omega/OmegaBuild.cmake`, replace this block (around lines 577-580):

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

This keeps the `OMEGA_LOG_TASKS` build option name; it now sets the compiled-in default string (overridable at runtime by the `OMEGA_LOG_TASKS` environment variable). `TOUPPER` is dropped (case is handled at runtime).

- [ ] **Step 2: Reconfigure, build & run the test**

From the Omega build directory, force a fresh configure so the new define is picked up, then run:

```bash
ctest -R LOGGING_TEST --output-on-failure
```

Expected: **Passed**. Verify the macro reaches the compiler as a string literal (the compile command for `Logging.cpp` should contain `-DOMEGA_LOG_TASKS_DEFAULT="master"` or `=\"master\"`). If a generator mangles the quotes so the macro is not a valid string literal, the symptom is a compile error in `getLogTaskSelector`; in that case switch the CMake line to `add_compile_definitions(OMEGA_LOG_TASKS_DEFAULT="${OMEGA_LOG_TASKS}")`.

Optional: configure with a non-default to confirm wiring:

```bash
# e.g. -DOMEGA_LOG_TASKS=* should make all ranks log by default
```

> Remote-build environments: commit and verify on the remote machine.

- [ ] **Step 3: Commit**

```bash
git add components/omega/OmegaBuild.cmake
git commit -m "$(cat <<'EOF'
Set OMEGA_LOG_TASKS default via string macro in CMake

Define OMEGA_LOG_TASKS_DEFAULT (default "master") instead of baking a
fixed rank list, so the OMEGA_LOG_TASKS build option supplies only the
runtime-overridable default.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Update documentation

**Files:**
- Modify: `components/omega/doc/design/Logging.md:115-120`
- Modify: `components/omega/doc/devGuide/CMakeBuild.md:70`

- [ ] **Step 1: Update the design doc**

In `components/omega/doc/design/Logging.md`, replace this paragraph (around lines 115-120):

```markdown
Users can control which MPI ranks generate log files using
`-D OMEGA_LOG_TASKS=<tasks-pattern>` at compile time. The `<tasks-pattern>` is
either all for all tasks to generate log files, or comma-separated MPI rank
numbers, or a range of MPI ranks with a dash. For example,
`-D OMEGA_LOG_TASKS=0,2-3` indicates that MPI ranks 0, 2, and 3 generate log
files.
```

with:

```markdown
Users can control which MPI ranks generate log files using the
`OMEGA_LOG_TASKS` selector, which is resolved at runtime against the Omega MPI
sub-communicator. The selector is read from the `OMEGA_LOG_TASKS` environment
variable; when it is unset, the compile-time default supplied by the
`-D OMEGA_LOG_TASKS=<selector>` build option is used (default `master`). The
`<selector>` accepts `*` (all ranks), `m` or `master` (the sub-communicator
master rank), a single rank number, a comma-separated list of ranks, an
inclusive dash range, or any combination of lists and ranges. For example,
`OMEGA_LOG_TASKS=0,2-3` makes MPI ranks 0, 2, and 3 generate log files. A
malformed selector logs a warning on the master rank and falls back to
master-rank-only logging; ranks outside the sub-communicator simply produce no
log.
```

- [ ] **Step 2: Update the build-guide doc**

In `components/omega/doc/devGuide/CMakeBuild.md`, replace line 70:

```markdown
OMEGA_LOG_TASKS: set the tasks that generate log file. "0" is a default value.
```

with:

```markdown
OMEGA_LOG_TASKS: sets the compile-time default for which MPI ranks generate log files, overridable at runtime by the OMEGA_LOG_TASKS environment variable. Accepts "*", "m"/"master", a rank number, a comma list, or a dash range (e.g. "0,2-3"). "master" is the default value.
```

- [ ] **Step 3: Commit**

```bash
git add components/omega/doc/design/Logging.md \
        components/omega/doc/devGuide/CMakeBuild.md
git commit -m "$(cat <<'EOF'
Document runtime OMEGA_LOG_TASKS selector

Update the logging design doc and CMake build guide to describe the
env-var-sourced selector, the new */master syntax, and the master-only
fallback for invalid selectors.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- Runtime selection sourced from env var with CMake default fallback → Task 2 (`getLogTaskSelector`), Task 3 (CMake).
- Resolution against Omega sub-communicator (`getNumTasks`/`getMasterTask`/`getMyTask`) → Task 1 (`_selectLogTasks`), Task 2 (overloads).
- Selector forms `*`, `m`/`master`, single, comma list, range, combined → Task 1 parser + tests.
- Malformed → warn on master + master fallback → Task 2 overloads; tested in Task 1.
- Out-of-range allowed, nobody logs (+ optional master note) → Task 2 (`NumLogging` note); tested in Task 1.
- Default `master` → Task 1 macro + Task 3 CMake.
- Preserve stdout/stderr redirect + multi-file naming → Task 2 file-path overload (unchanged redirect; `NumLogging`-based suffix).
- Tests cover new behavior → Task 1 `testTaskSelection`.
- Docs updated → Task 4.

**Placeholder scan:** No TBD/TODO/"handle edge cases"; every code step shows complete code.

**Type consistency:** `_selectLogTasks(const std::string &, I4, I4, bool &) -> std::vector<int>` is declared (Task 1 Step 1), defined (Task 1 Step 2), and called identically in the test (Task 1 Step 3) and both overloads (Task 2). `OMEGA_LOG_TASKS_DEFAULT` is defined in the header (Task 1) and consumed in `getLogTaskSelector` (Task 2) and CMake (Task 3). `getLogTaskSelector()`, `trimStr()`, `toNonNegInt()` signatures are consistent across definition and use.

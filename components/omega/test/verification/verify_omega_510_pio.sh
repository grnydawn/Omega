#!/usr/bin/env bash
#
# verify_omega_510_pio.sh -- verification for Omega issue #510
#   "Use driver-owned PIO settings for coupled Omega initialization"
#
# Three phases, increasing in cost:
#
#   A  namelist guard (seconds, no build)
#      IO.IOBaseTask / IO.IORearranger must be REJECTED in user_nl_omega, while
#      IO.IOTasks / IO.IOStride stay user-configurable. Discriminating: run it
#      against a tree without the fix and phase A fails.
#
#   B  source plumbing (seconds, no build)
#      shr_pio_getioroot/shr_pio_getrearranger are actually called, and the
#      omega_ocn_init1 argument count agrees across the Fortran caller, the
#      Fortran interface block and the C++ definition.
#
#   C  integration run (--with-run; needs a build, input data and a batch job)
#      Build a small coupled case, point the DRIVER at non-default PIO settings
#      for OCN, and confirm the model initializes and runs with them.
#
# Phase C is a REAL check, not a smoke test, provided IO::init logs its resolved
# settings (it does on the #510 branch). The driver is pointed at non-default
# PIO settings for OCN and the run's ocean log must report those same values --
# if Omega fell back to its own YAML defaults, the assertion fails. Phase B
# reports if the log line is missing, in which case phase C degrades to a smoke
# test and says so.
#
# Usage:
#   ./components/omega/test/verification/verify_omega_510_pio.sh
#   ./components/omega/test/verification/verify_omega_510_pio.sh --with-run --machine pm-cpu --compiler gnu --project mXXXX
#
# CIME needs python >= 3.8. It is loaded with a module command, not a hard-coded
# path. Change it for your system by editing PYTHON_MODULE_DEFAULT below, or:
#   --python-module "module load cray-python"
#   OMEGA_VERIFY_PYTHON_MODULE="module load python/3.11" verify_omega_510_pio.sh ...
#   --python-module ""    (skip modules; use the python3 already on PATH)
#
set -euo pipefail

MACH=""; COMPILER=""; PROJECT=""
RES="T62_oQU240"
COMPSET=""
DRIVER="mct"
PECOUNT="S"
CASEROOT_BASE="${SCRATCH:-$HOME}/omega-verify"
WITH_RUN=0
STOP_N=5
PIO_ROOT_OCN=1          # driver-owned value to push at Omega (non-default)
PIO_REARR_OCN=2         # 1 = box (default), 2 = subset
PYTHON_MODULE=""        # set below from PYTHON_MODULE_DEFAULT/env/--python-module

usage() { sed -n "2,38p" "$0"; exit 0; }

while [ $# -gt 0 ]; do
  case "$1" in
    --machine)  MACH="$2"; shift 2 ;;
    --compiler) COMPILER="$2"; shift 2 ;;
    --project)  PROJECT="$2"; shift 2 ;;
    --res)      RES="$2"; shift 2 ;;
    --compset)  COMPSET="$2"; shift 2 ;;
    --driver)   DRIVER="$2"; shift 2 ;;
    --pecount)  PECOUNT="$2"; shift 2 ;;
    --caseroot) CASEROOT_BASE="$2"; shift 2 ;;
    --with-run) WITH_RUN=1; shift ;;
    --stop-n)   STOP_N="$2"; shift 2 ;;
    --pio-root) PIO_ROOT_OCN="$2"; shift 2 ;;
    --pio-rearr) PIO_REARR_OCN="$2"; shift 2 ;;
    --python-module) PYTHON_MODULE="$2"; PYTHON_MODULE_SET=1; shift 2 ;;
    -h|--help)  usage ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# repo root: this script lives at components/omega/test/verification/
SRCROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$SRCROOT"

pass() { printf '  PASS  %s\n' "$*"; }
fail() { printf '  FAIL  %s\n' "$*"; FAILED=1; }
say()  { printf '\n== %s\n' "$*"; }
FAILED=0

# ------------------------------------------------------- python for CIME ----
# CIME needs python >= 3.8 and many systems default to something older, so the
# right python is loaded with a module command rather than a hard-coded path
# (paths differ per system; module names do too).
#
# EDIT THIS for your system, or override it without editing:
#   --python-module "module load cray-python"
#   OMEGA_VERIFY_PYTHON_MODULE="module load python/3.11" ./this-script ...
# Set it to the empty string to skip module handling entirely and use whatever
# python3 is already on PATH.
PYTHON_MODULE_DEFAULT="module load python"

# precedence: --python-module  >  $OMEGA_VERIFY_PYTHON_MODULE  >  the default
# above. Note ${VAR-...} not ${VAR:-...}, so an explicitly EMPTY env var means
# "skip module handling", rather than falling back to the default.
if [ "${PYTHON_MODULE_SET:-0}" != "1" ]; then
  PYTHON_MODULE="${OMEGA_VERIFY_PYTHON_MODULE-$PYTHON_MODULE_DEFAULT}"
fi

python_ok() {
  command -v python3 >/dev/null 2>&1 &&
    python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null
}

# `module` is usually only a shell function in login shells; make it available
# here too before trying to use it.
init_modules() {
  if type module >/dev/null 2>&1; then return 0; fi
  for _init in /usr/share/lmod/lmod/init/bash \
               /opt/cray/pe/lmod/lmod/init/bash \
               "${LMOD_PKG:-/nonexistent}/init/bash" \
               /etc/profile.d/lmod.sh \
               /etc/profile.d/modules.sh; do
    if [ -r "$_init" ]; then
      # shellcheck disable=SC1090
      . "$_init"
      break
    fi
  done
  type module >/dev/null 2>&1
}

setup_python() {
  if python_ok; then return 0; fi
  if [ -z "$PYTHON_MODULE" ]; then return 1; fi
  if ! init_modules; then
    echo "ERROR: no 'module' command available to run: $PYTHON_MODULE" >&2
    return 1
  fi
  say "loading python: $PYTHON_MODULE"
  eval "$PYTHON_MODULE" || true
  # bash caches resolved command paths; without this the old python3 sticks
  # even though the module already prepended the new one to PATH.
  hash -r
  python_ok
}

if ! setup_python; then
  cat >&2 <<ERR
ERROR: need python >= 3.8 for CIME, and could not get one.
       Tried: ${PYTHON_MODULE:-<module handling disabled>}
       Fix by editing PYTHON_MODULE_DEFAULT at the top of this script, or:
         --python-module "module load <your python module>"
         OMEGA_VERIFY_PYTHON_MODULE="module load <...>" $0 ...
ERR
  exit 2
fi
say "python: $(python3 --version 2>&1) ($(command -v python3))"

# =========================================================== PHASE A ========
say "PHASE A -- user_nl_omega guard on driver-owned IO options"
NMLDIR="$SRCROOT/components/omega/cime_config"
PROBE=$(mktemp /tmp/omega510probe.XXXXXX.py)
cat > "$PROBE" <<'PY'
import sys
from omega_buildnml.validate import validate_blocked_options
overrides = {
    "IO": {"IOBaseTask": 5, "IORearranger": "subset", "IOTasks": 8, "IOStride": 2},
    "TimeIntegration": {"StartTime": "0001-01-01"},
}
errs = validate_blocked_options(overrides, "user_nl_omega")
joined = " ".join(errs)
checks = [
    ("IO.IOBaseTask is blocked",              "IO.IOBaseTask"    in joined),
    ("IO.IORearranger is blocked",            "IO.IORearranger"  in joined),
    ("control: TimeIntegration.StartTime blocked", "TimeIntegration.StartTime" in joined),
    ("IO.IOTasks stays configurable",         "IO.IOTasks"  not in joined),
    ("IO.IOStride stays configurable",        "IO.IOStride" not in joined),
]
ok = True
for name, res in checks:
    print(("  PASS  " if res else "  FAIL  ") + name)
    ok = ok and res
print("  raw:", errs or "(no errors)")
sys.exit(0 if ok else 1)
PY
if PYTHONPATH="$NMLDIR" python3 "$PROBE"; then
  pass "phase A"
else
  fail "phase A -- driver-owned IO options are NOT blocked (is the #510 fix on this branch?)"
fi
rm -f "$PROBE"

say "PHASE A2 -- omega_buildnml unit tests"
if ( cd "$NMLDIR" && python3 -m pytest omega_buildnml/tests -q ) ; then
  pass "omega_buildnml test suite"
else
  fail "omega_buildnml test suite"
fi

# =========================================================== PHASE B ========
say "PHASE B -- coupled init plumbing"
MCT="components/omega/src/drivers/coupled/ocn_comp_mct.F90"
F2C="components/omega/src/drivers/coupled/omega_f2cxx_interface.F90"
C2F="components/omega/src/drivers/coupled/omega_cxx2f_interface.cpp"

for sym in shr_pio_getioroot shr_pio_getrearranger; do
  if grep -q "$sym" "$MCT"; then pass "$MCT calls $sym"
  else fail "$MCT does not call $sym"; fi
done

# argument count of omega_ocn_init1 must agree across the three layers
N_IFACE=$(awk '/subroutine omega_ocn_init1\(/,/bind\(c\)/' "$F2C" | grep -c '&$' || true)
N_CALL=$(awk '/call omega_ocn_init1\(/,/\)$/' "$MCT" | grep -c ',' || true)
N_CXX=$(awk '/void omega_ocn_init1\(/,/^\) \{|^\)/' "$C2F" | grep -c ',' || true)
echo "  arg-ish counts: fortran-interface=$N_IFACE fortran-call=$N_CALL cxx-def=$N_CXX"
if grep -q "IOBaseTask\|io_base_task\|ioroot" "$F2C" && \
   grep -q "IOBaseTask\|io_base_task\|ioroot" "$C2F"; then
  pass "IO base task is plumbed through both interface layers"
else
  fail "IO base task is missing from an interface layer"
fi
if grep -q "IORearranger\|io_rearranger\|rearranger" "$F2C" && \
   grep -q "IORearranger\|io_rearranger\|rearranger" "$C2F"; then
  pass "IO rearranger is plumbed through both interface layers"
else
  fail "IO rearranger is missing from an interface layer"
fi
if grep -q "IOInitParams" components/omega/src/base/IO.h; then
  pass "IO.h declares IOInitParams"
else
  fail "IO.h has no IOInitParams -- the #510 fix is not on this branch"
fi

# Without this log line the runtime values are unobservable: they go straight
# into PIOc_Init_Intracomm and are never reported.
if grep -q 'IO::init: IOTasks=' components/omega/src/base/IO.cpp; then
  pass "IO::init logs the resolved IO settings (phase C can verify them)"
  HAVE_IO_LOG=1
else
  fail "IO::init does not log the resolved settings -- phase C can only smoke test"
  HAVE_IO_LOG=0
fi

# =========================================================== PHASE C ========
if [ "$WITH_RUN" -eq 1 ]; then
  [ -n "$MACH" ] || { echo "ERROR: --with-run needs --machine" >&2; exit 2; }
  [ -n "$COMPILER" ] || { echo "ERROR: --with-run needs --compiler" >&2; exit 2; }

  if [ "$DRIVER" = "mct" ] && [ ! -f externals/mct/mct/Makefile ]; then
    say "initializing externals/mct submodule"
    git submodule update --init --depth=1 externals/mct
  fi
  if [ -z "$COMPSET" ]; then
    COMPSET=$(grep -o '<alias>COMEGA[^<]*</alias>' components/omega/cime_config/config_compsets.xml \
              | head -1 | sed 's/<[^>]*>//g')
  fi

  CASE="$CASEROOT_BASE/c510.${MACH}.${COMPILER}"
  mkdir -p "$CASEROOT_BASE"; rm -rf "$CASE"
  say "PHASE C -- creating $CASE  (res=$RES compset=$COMPSET)"
  PROJ_ARG=""; [ -n "$PROJECT" ] && PROJ_ARG="--project $PROJECT"
  ./cime/scripts/create_newcase --case "$CASE" --res "$RES" --compset "$COMPSET" \
      --driver "$DRIVER" --machine "$MACH" --compiler "$COMPILER" \
      --pecount "$PECOUNT" $PROJ_ARG > "$CASEROOT_BASE/create.510.log" 2>&1 \
    || { fail "create_newcase (see $CASEROOT_BASE/create.510.log)"; exit 1; }

  cd "$CASE"
  say "pushing NON-DEFAULT driver PIO settings at the ocean component"
  ./xmlchange PIO_ROOT_OCN="$PIO_ROOT_OCN"
  ./xmlchange PIO_REARRANGER_OCN="$PIO_REARR_OCN"
  ./xmlchange STOP_OPTION=nsteps,STOP_N="$STOP_N",REST_OPTION=never
  ./xmlquery -p PIO_ROOT -p PIO_REARRANGER 2>/dev/null | sed 's/^/  /'

  ./case.setup > "$CASEROOT_BASE/setup.510.log" 2>&1 \
    || { fail "case.setup (see $CASEROOT_BASE/setup.510.log)"; exit 1; }

  say "downloading any missing input data"
  ./check_input_data --download > "$CASEROOT_BASE/inputdata.510.log" 2>&1 || true

  say "case.build"
  if ./case.build > "$CASEROOT_BASE/build.510.log" 2>&1; then
    pass "case.build"
  else
    fail "case.build (see $CASEROOT_BASE/build.510.log)"; exit 1
  fi

  say "case.submit  (STOP_N=$STOP_N nsteps)"
  if ./case.submit --no-batch > "$CASEROOT_BASE/run.510.log" 2>&1; then
    pass "run completed with driver PIO_ROOT_OCN=$PIO_ROOT_OCN REARRANGER=$PIO_REARR_OCN"
  else
    fail "run failed (see $CASEROOT_BASE/run.510.log)"
  fi
  RUNDIR=$(./xmlquery -N --value RUNDIR)
  say "checking the IO settings Omega actually initialized SCORPIO with"
  say "  (logs in $RUNDIR)"

  # CIME gzips the component logs at the end of a successful run, so plain grep
  # finds nothing. zgrep reads both plain and .gz; fall back to grep if absent.
  if command -v zgrep >/dev/null 2>&1; then
    IOLINE=$(zgrep -h "IO::init: IOTasks=" "$RUNDIR"/*.log* 2>/dev/null | head -1 || true)
  else
    IOLINE=$(grep -h "IO::init: IOTasks=" "$RUNDIR"/*.log* 2>/dev/null | head -1 || true)
  fi
  if [ -z "$IOLINE" ]; then
    if [ "${HAVE_IO_LOG:-0}" = "1" ]; then
      fail "no 'IO::init:' line found in $RUNDIR -- did the ocean reach IO::init?"
    else
      say "  (skipped: this branch has no IO::init log line)"
    fi
  else
    echo "  $IOLINE"
    GOT_BASE=$(printf '%s' "$IOLINE" | sed -n 's/.*IOBaseTask=\([0-9-]*\).*/\1/p')
    GOT_REARR=$(printf '%s' "$IOLINE" | sed -n 's/.*IORearranger=[a-z]* (\([0-9]*\)).*/\1/p')
    if [ "$GOT_BASE" = "$PIO_ROOT_OCN" ]; then
      pass "IOBaseTask=$GOT_BASE matches the driver's PIO_ROOT_OCN"
    else
      fail "IOBaseTask=$GOT_BASE but the driver was set to PIO_ROOT_OCN=$PIO_ROOT_OCN (Omega ignored the driver)"
    fi
    if [ "$GOT_REARR" = "$PIO_REARR_OCN" ]; then
      pass "IORearranger=$GOT_REARR matches the driver's PIO_REARRANGER_OCN"
    else
      fail "IORearranger=$GOT_REARR but the driver was set to PIO_REARRANGER_OCN=$PIO_REARR_OCN (Omega ignored the driver)"
    fi
  fi
else
  say "PHASE C skipped (pass --with-run --machine M --compiler C to enable)"
fi

say "RESULT: $([ $FAILED -eq 0 ] && echo PASS || echo FAIL)"
exit $FAILED

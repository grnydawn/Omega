# Coupled build/run verification scripts

These are not ctest unit tests. They create a real CIME case, build it, and in
one case run it, then assert on what the build and run actually produced. They
exist because the behaviour they check -- which GPU architecture Omega's build
selects, and which PIO settings Omega initializes SCORPIO with -- is not
observable from the unit tests.

Run them from anywhere; they locate the repo root themselves.

## verify_omega_508_arch.sh

Checks that a coupled (E3SM) build sets `OMEGA_ARCH` and picks up the machine's
Kokkos architecture.

```bash
./components/omega/test/verification/verify_omega_508_arch.sh \
    --machine frontier --compiler craycray-mphipcc
```

It derives the expected `OMEGA_ARCH` and `Kokkos_ARCH_*` from the machine's
`cime_config/machines/cmake_macros/` files, so no expectation needs to be passed
in (`--expect` overrides if needed). Asserts that `OMEGA_ARCH` is non-empty and
matches, that the Kokkos architecture reached Omega, that the "GPU build but no
Kokkos_ARCH_*" guard did not fire, and that the build succeeded.

**Run this on a GPU machine.** On a CPU machine the check does not discriminate:
without the fix `OMEGA_ARCH` also ends up `SERIAL` and the compile defines are
identical, so only a status line differs. The script warns when it detects this.

## verify_omega_510_pio.sh

Checks that coupled Omega takes its PIO base task and rearranger from the driver
rather than from its own YAML config.

```bash
# phases A and B only -- seconds, no build required
./components/omega/test/verification/verify_omega_510_pio.sh

# add the integration run
./components/omega/test/verification/verify_omega_510_pio.sh \
    --with-run --machine pm-cpu --compiler gnu --project <acct>
```

- **Phase A** -- `IO.IOBaseTask` and `IO.IORearranger` must be rejected in
  `user_nl_omega`, while `IO.IOTasks` and `IO.IOStride` stay user-configurable.
- **Phase B** -- `shr_pio_getioroot`/`shr_pio_getrearranger` are called and the
  values are plumbed through both interface layers.
- **Phase C** -- points the driver at PIO settings that differ from Omega's YAML
  defaults, runs, and asserts the ocean log reports the driver's values. This
  relies on the `IO::init` log line; phase B reports if it is missing, in which
  case phase C degrades to a smoke test.

## Python

CIME needs python >= 3.8. Both scripts load it with a module command rather than
a hard-coded path, since paths and module names differ per system. Edit
`PYTHON_MODULE_DEFAULT` near the top of each script, or override per invocation:

```bash
--python-module "module load cray-python"
OMEGA_VERIFY_PYTHON_MODULE="module load python/3.11" <script> ...
--python-module ""      # skip modules, use the python3 already on PATH
```

## Notes

- A `--driver mct` build needs the `externals/mct` submodule; the scripts
  initialize it if it is missing.
- Cases are created under `$SCRATCH/omega-verify` by default (`--caseroot`).
- Building needs no compute allocation; only `--with-run` does.

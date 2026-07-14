(omega-dev-testing)=

# Testing Code

For instructions on how to build Omega and run CTest unit tests, see
{ref}`omega-dev-quick-start-build-test`.


## Running the Polaris `omega_pr` Test Suite

The CTest unit tests on their own are not sufficient to ensure that changes
to Omega behave as expected.  Developers also need to run the regression tests
in the form of the
[omega_pr](https://docs.e3sm.org/polaris/main/users_guide/ocean/suites.html#omega-pr-suite)
test suite from Polaris.  The tests in the suite are defined in
[omega_pr.txt](https://github.com/E3SM-Project/polaris/blob/main/polaris/suites/ocean/omega_pr.txt)
in the Polaris repository.

You will run the suite twice: once using the Omega version vendored in Polaris
under `e3sm_submodules/Omega` (the official baseline) and once using your
development branch. This means you will need to
{ref}`build Omega <omega-dev-quick-start-build>` from both sources before
running the suite. You may wish to use the {ref}`omega-dev-quick-start-ctest-util`
to automate the build and run the CTests for each.

```{note}
While the Omega `develop` branch often matches the Polaris submodule
bit-for-bit (b4b), the submodule at `e3sm_submodules/Omega` is the
authoritative baseline for regression comparison.
```

### Prerequisites: build Omega for baseline and PR

Before setting up and running the `omega_pr` suite, you must build Omega twice:

- Build the baseline from the Omega sources vendored by Polaris at
  `<path-to-polaris-repo>/e3sm_submodules/Omega`. Record the CMake build
  directory path as `BASELINE_BUILD` (used below with `polaris suite -p`).
- Build from your development (PR) branch. Record the CMake build directory
  path as `PR_BUILD` (also used with `-p`).

The `-p` option passed to `polaris suite` should point to the CMake build
directory for the corresponding branch (the directory where you ran CMake and
invoked `./omega_build.sh`).

```{note}
Run both baseline and PR suite on the same machine, CPU/GPU partition,
compiler, and Omega build type (Debug/Release) to avoid false positive/negative
diffs.
```

### Setting up the suite

If you have not yet set up Polaris, you will need to follow its
[quick start](https://docs.e3sm.org/polaris/main/developers_guide/quick_start.html)
including [cloning the repo](https://docs.e3sm.org/polaris/main/developers_guide/quick_start.html#set-up-a-polaris-repository-for-beginners),
[configuring the software environment](https://docs.e3sm.org/polaris/main/developers_guide/quick_start.html#supported-machines),
and [activating the environment](https://docs.e3sm.org/polaris/main/developers_guide/quick_start.html#activating-the-environment).

Then, with the Polaris environment active, you can set up the baseline
`omega_pr` suite as follows:
``` bash
BASELINE_BUILD=/path/to/build/from/polaris/e3sm_submodules/Omega
POLARIS_SCRATCH=/path/to/scratch/polaris_testing/
polaris suite -c ocean -t omega_pr --model omega \
  -w $POLARIS_SCRATCH/baseline_omega_pr \
  -p $BASELINE_BUILD
```
where `$POLARIS_SCRATCH` is a directory within your scratch space that is
useful for setting up and running the Polaris test suites and
`$BASELINE_BUILD` is the directory where you built Omega from the Omega
sources located at `<polaris-repo>/e3sm_submodules/Omega`, or where the
Polaris CTest utility built it for you (e.g. `build_omega/build_chrysalis_intel`).

Next, you can set up the `omega_pr` for your development branch, using the
baseline set up above:

``` bash
PR_BUILD=/path/to/omega/pr/branch/build
POLARIS_SCRATCH=/path/to/scratch/polaris_testing/
polaris suite -c ocean -t omega_pr --model omega \
    -b $POLARIS_SCRATCH/baseline_omega_pr \
    -w $POLARIS_SCRATCH/my_branch_omega_pr \
  -p $PR_BUILD
```
Here, `$POLARIS_SCRATCH` is the same scratch directory as above,
`$PR_BUILD` is the directory where you build Omega from your
development branch (or where the CTest utility built for you), and
`my_branch_omega_pr` can be any name that helps you keep track of what you
were testing.

### Running the suite

First, run the baseline, e.g. for Slurm systems:
``` bash
cd $POLARIS_SCRATCH/baseline_omega_pr
sbatch job_script.omega_pr.sh
```
Note the job ID for the next step.  Next, run the suite for the test branch,
waiting until the baseline is done:
``` bash
cd $POLARIS_SCRATCH/my_branch_omega_pr
sbatch --kill-on-invalid-dep=yes --dependency=afterok:$JOBID job_script.omega_pr.sh
```
where `$JOBID` is the JOB ID of the baseline slurm job.

Monitor the jobs (e.g. `squeue -j $JOBID`) and take a look at combined
stdout/stderr log in `polaris_omega_pr.o$JOBID` in each work directory to see
whether the tests passed.  Report the results as part of any (non-trivial)
Omega PR.

```{note}
For other schedulers (e.g. PBS on Aurora), take a look at the machine's
documentation for equivalent commands.
```

### Handling test failures

What should you do if you see test failures?

#### Test execution errors for baseline (Polaris `e3sm_submodules/Omega`)

If you see test execution failures in the baseline run from Polaris
`e3sm_submodules/Omega`,
these need to be reported as [issues](https://github.com/E3SM-Project/Omega/issues)
on the Omega repo, preferably mentioning a relevant Omega developer who may
be able to investigate the cause of the failure.  Please provide the contents
of the relevant log file for the failing test(s) and optionally the location of
the tests (and log files) if other Omega developers will have access.

#### Test execution errors for your branch

If you see execution failures on the tests for your development branch, please
take a look at the log files and attempt to debug the issues yourself first,
then reach out to the Omega team if you need help.  In many cases, the test
failures may related to changes that are required in Polaris, see
{ref}`omega-dev-testing-update-polaris` below.

#### Diffs between your branch and baseline

If diffs are expected:
- Document the diffs (in Omega PR comment)
- Update `e3sm_submodules/Omega` in Polaris (after merge of Omega PR)

If *not* expected:
- Debug and eliminate

If you see errors related to comparisons with the baseline, you will need to
determine if these are expected based on the code changes or not.  If they
are expected, you will need to document (in a PR comment) how large the
differences are in the tests that show differences.  In addition, you will need
to make a pull request to the Polaris repository to update the Omega submodule
in `e3sm_submodules/Omega` once your branch has been merged so that the changes
become the new baseline in Polaris.

```{note}
We do not update `e3sm_submodules/Omega` after every branch is merged into
`develop` on Omega to reduce maintenance burden.
```

If you see unexpected differences between the results for your development
branch and those from the baseline, you should attempt to debug the cause yourself
and then reach out to the Omega team if you need help.

You may want to look at Polaris'
[bisect](https://github.com/E3SM-Project/polaris/blob/main/utils/bisect/README.md)
utility to help debug which commit introduced the unexpected differences.


### What to include in your PR Testing comment

Please add a PR comment titled "Testing" that summarizes what you ran and the
outcomes. Include the following so reviewers can validate apples-to-apples
runs.

```{note}
At a minimum, you need to run both CTests and the `omega_pr` suite on one
machine and compiler.
```

- CTest unit tests
    - Machine(s) and partition/queue (if applicable)
    - Compiler(s), and build type(s) (Debug/Release)
    - Result: either "All tests passed" or a short list of failing tests with
      a one-line note each

- Polaris `omega_pr` regression suite
  - Baseline run (Polaris `e3sm_submodules/Omega`)
    - Build path used for -p (from Polaris `e3sm_submodules/Omega`)
        - Work directory used for -w
    - PR branch run
        - Build directory used for -p
        - Work directory used for -w
    - Machine/partition, compiler, and build type (should match between baseline and PR)
    - Result: "All tests passed" or a concise list of diffs/failures with a brief note
    - If there are issues, add the path to the relevant log(s) (e.g.,
      polaris_omega_pr.o$JOBID) and a short excerpt if helpful

- Tests added/modified/impacted
    - Briefly list any new or updated CTest or Polaris tests and what they cover

- Performance PRs
    - Link to the relevant PACE experiment and summarize before/after metrics

The [Omega CTest Utility](https://github.com/E3SM-Project/polaris/blob/main/utils/omega/ctest/README.md)
and the `omega_pr` suite will produce output in a format you can copy/paste into the PR
description. These outputs follow roughly the template below.  If you want to add the
information manually, you can copy-paste template you can use in your PR comment:

```text
## Testing

CTest unit tests
- Machine: <machine>[, <partition>]
- Compiler: <compiler>
- Build type: <Debug|Release>
- Result: All tests passed | Failures (X of Y): <name> (<one-line note>), ...

Polaris omega_pr regression suite
- Baseline build (-p): <path/to/build of Polaris e3sm_submodules/Omega>
- Baseline workdir (-w): <path/to/baseline_omega_pr>
- PR build (-p): <path/to/pr/build>
- PR workdir (-w): <path/to/my_branch_omega_pr>
- Machine/partition: <machine>[, <partition>]
- Compiler/build type: <compiler>, <Debug|Release>
- Result: All tests passed | Diffs/Failures: <brief summary>
- Logs (if applicable): <path/to/polaris_omega_pr.o$JOBID>

Tests added/modified/impacted
- CTest: <name> — <one-line purpose>
- Polaris: <path or name> — <one-line purpose>

Performance (if applicable)
- PACE: <link>
- Before: <metric(s)>
- After: <metric(s)>
- Delta: <summary>
```

(omega-dev-testing-update-polaris)=

### Updating Polaris

Many Omega changes are likely to also require Polaris changes.  In such cases,
it makes sense to co-develop your Omega branch and a Polaris branch.  The
Omega branch will need to be merged first, then the `e3sm_submodules/Omega`
submodule will need to be updated to include the Omega changes as part of
the Polaris pull request.

Here are some examples of Omega changes that will require corresponding
Polaris changes.

#### Adding new Omega dimensions or variables

If you add new Omega variables that have a corresponding MPAS-Ocean name,
you need to add the mapping between the names to
[mpaso_to_omega.yaml](https://github.com/E3SM-Project/polaris/blob/main/polaris/ocean/model/mpaso_to_omega.yaml).
For more details, see the Polaris
[Ocean Framework documentation](https://docs.e3sm.org/polaris/main/developers_guide/ocean/framework.html).

For example:
```yaml
dimensions:
  Time: time
  nVertLevels: NVertLayers

variables:
  temperature: Temperature
  salinity: Salinity
  ssh: SshCell
```
In each case, the keys are the MPAS-Ocean names and the values are the
corresponding Omega names.

#### Updates to Default.yml

If you add or remove sections, config options, streams, etc. in
[Default.yml](https://github.com/E3SM-Project/Omega/blob/develop/components/omega/configs/Default.yml),
you may need to make corresponding changes to the YAML files for Polaris
test cases for them to continue to work as expected.

Whenever possible, we try to use a mapping between MPAS-Ocean namelist options
and Omega config options within
[mpaso_to_omega.yaml](https://github.com/E3SM-Project/polaris/blob/main/polaris/ocean/model/mpaso_to_omega.yaml).
For example:
```yaml
config:
- section:
    time_management: TimeIntegration
  options:
    config_start_time: StartTime
    config_stop_time: StopTime
    config_run_duration: RunDuration
    config_calendar_type: CalendarType
```

If you have added new Omega config sections or options that have corresponding
MPAS-Ocean namelist sections or options, please add them to the map. In each
case, the keys are the MPAS-Ocean names and the values are the corresponding
Omega names.

Frequently, there is no MPAS-Ocean equivalent to an Omega config option, and
we currently do not try to map streams from MPAS-Ocean to Omega (their
syntax is too different).  In such cases, we have do define Omega YAML config
options and streams in each test case, and you may need to update each one
with your changes.  Here is an example exerpt from the
[manufactured_solution](https://github.com/E3SM-Project/polaris/blob/main/polaris/tasks/ocean/manufactured_solution/forward.yaml)
test:
```yaml
Omega:
  Tendencies:
    VelDiffTendencyEnable: false
    VelHyperDiffTendencyEnable: false
    UseCustomTendency: true
    ManufacturedSolutionTendency: true
  IOStreams:
    InitialVertCoord:
      Filename: init.nc
    InitialState:
      UsePointerFile: false
      Filename: init.nc
      Contents:
      - NormalVelocity
      - PseudoThickness
    RestartRead: {}
    History:
      Filename: output.nc
      Freq: {{ output_freq }}
      FreqUnits: Seconds
      IfExists: append
      # effectively never
      FileFreq: 9999
      FileFreqUnits: years
      Contents:
      - NormalVelocity
      - PseudoThickness
      - SshCell
```
For more details on updating either the map or the YAML files for individual
tests, see the Polaris
[Ocean Framework documentation](https://docs.e3sm.org/polaris/main/developers_guide/ocean/framework.html).

## Memory-leak and coverage analysis (standalone)

Two opt-in CMake options add analysis to the standalone CTest suite. Both
default OFF, are standalone-only, and are independent.

### Memory-leak checking — `-DOMEGA_MEMCHECK=ON`

Each test rank runs under an auto-selected leak checker (valgrind on CPU;
`compute-sanitizer` on CUDA). A leak makes the test fail (`--error-exitcode=1`),
so leaks appear as failed tests in a normal run. Configure, build, then:

> **Point at a compute-node-valid tool path.** The leak-tool path is resolved at
> configure time and baked into every test command. On clusters the login-node
> system tool (e.g. `/usr/bin/valgrind`) does not exist on compute nodes, so
> tests fail with `execve(): .../valgrind: No such file or directory`. The
> robust fix is to pass the module/shared-filesystem path explicitly:
>
> ``` bash
> module load valgrind          # module valgrind lives on the shared FS
> cmake -DOMEGA_MEMCHECK_EXE=$(which valgrind) .   # honored verbatim; re-bakes the path
> ```
>
> `OMEGA_MEMCHECK_EXE` is honored as-is and never overwritten by auto-detection.
> The baked absolute path (e.g. `/gpfs/.../valgrind-3.17.0/bin/valgrind`) runs on
> compute nodes even without the module loaded at run time, since it is on the
> shared filesystem and valgrind self-locates its libraries. Re-running `cmake .`
> in the build dir is enough — no rebuild needed. A configure-time WARNING is
> emitted when a system path (`/usr/bin`, `/bin`, …) is resolved by
> auto-detection, since that path may not exist on compute nodes. (Loading the
> module before configuring *may* also suffice, but Omega's CIME environment
> setup can reset `PATH`, so the explicit `-DOMEGA_MEMCHECK_EXE` override is the
> reliable method.)

``` bash
./omega_memcheck.sh                 # whole suite (compute node)
./omega_memcheck.sh -L memcheck     # only memcheck-labelled tests
./omega_memcheck.sh -R HALO_TEST    # one test
```

valgrind is 10–30× slower, so tests get a 3600 s timeout under memcheck. Benign
third-party init leaks are masked by `test/omega.supp`; regenerate/extend it by
adding `--gen-suppressions=all` to the launcher. A `MEMCHECK_PROBE_TEST` (a
deliberate leak, `WILL_FAIL`) is registered only under memcheck to prove the
checker actually detects leaks.

CDash note: because tests launch through MPI, the checker is embedded per rank
rather than via CTest's `MEMORYCHECK_COMMAND` (which would wrap `srun`). Leaks
therefore appear on CDash as failed tests, not as a Dynamic Analysis widget.

### Coverage — `-DOMEGA_COVERAGE=ON`

Instruments Omega host code (`--coverage -O0 -g`; third-party deps excluded),
selects the gcov front-end per compiler (`gcov` for GNU; `llvm-cov gcov` for
`oneapi-ifx`/Clang), and builds a gcovr HTML+text report. Requires `gcovr`
(in `dev-conda.txt`). Recommended with `-DOMEGA_BUILD_TYPE=Debug`.

``` bash
./omega_coverage.sh                 # run suite, then write coverage/
# -> coverage/omega_coverage.html, coverage/omega_coverage.txt
```

Coverage is host-only: device (CUDA/HIP/SYCL) kernels are not measured.

Multi-rank note: an N-rank test runs N processes that write the same `.gcda`
files. libgcov file-locks and merges per source, which is correct for ranks on
one node. For cross-node robustness, export a per-rank prefix before the run:

``` bash
export GCOV_PREFIX=$PWD/coverage/gcda/rank_${SLURM_PROCID:-0}
export GCOV_PREFIX_STRIP=0
```

then point gcovr at `coverage/gcda` in addition to the build tree.

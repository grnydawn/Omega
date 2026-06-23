# Coverage annotation legend

Each file under this tree mirrors `components/omega/src/`, with every line prefixed
by a 1-character coverage marker in column 1 (a space separates it from the original
line, so line N here is line N of the original source):

| Mark | Meaning |
|------|---------|
| `+`  | line executed during ctest (covered) |
| `-`  | executable line NOT executed (missed) |
| (space) | non-executable line (blank / comment / declaration / brace) |
| `?`  | file had no coverage data at all (never compiled into any instrumented test) |

Source of data: `coverage.info` (llvm-cov export -format=lcov) from the QA ctest run
on Aurora (SERIAL/icpx, Debug, OMEGA_COVERAGE=ON), build commit 44697f9e72. The
`components/omega/src/` tree is byte-identical between that commit and the working
tree, so line numbers align exactly.

Per-line line coverage = covered / (covered + missed). See `_SUMMARY.txt`.
To find uncovered lines quickly: `grep -n '^-' <file>`.

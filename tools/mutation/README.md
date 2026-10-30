# Mutation testing

Mutation testing checks that the properties would fail on a broken kernel. It plants one small change at a time in `arrow-select` with cargo-mutants and records whether the properties catch it and whether `arrow-select`'s own tests do. [Measuring the properties](../../docs/workflow.md#measuring-the-properties) in the workflow says when to run it, how, and when to take a new baseline.

| File                  | Contents                                                                                                                                      |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `mutants-select.json` | The mutants, from `cargo mutants --list --json -p arrow-select` (cargo-mutants 27.1.0) on arrow-rs `main` at 70d8190350                       |
| `results.jsonl`       | One line per mutant: whether it built, whether the properties and the tests caught it, and how long each took                                 |
| `run.py`              | The runner: `run.py setup K ARROW_RS` copies the checkout and this repository for K workers, and `run.py run K MUTANTS.json` runs the mutants |
| `summarize.py`        | Kill counts per file, and the mutants only one side caught: `summarize.py results.jsonl`                                                      |

## Results

On 2026-10-28, every one of the 1,657 mutants of `arrow-select` at arrow-rs `main` 70d8190350 ran against the properties that exercise its file and against `cargo test -p arrow-select`. The properties were those of this repository at 818f008, with seed 1, 1,000 cases and no shrinking. Every property passed on the unmutated commit, seeds 1 to 3 at 1,000 cases each, before the run. It took 1 hour 56 minutes on 12 workers.

Each worker caps its memory at 6 GB across its process group, a property run at 400 seconds, and a test run at 900. A mutant that makes `SlicesIterator::next` return `Some` forever allocates without bound, so without a cap a few such mutants exhaust memory. A cap below the 3.5 GB that `arrow-select`'s unmutated tests use records false kills. The runner counts a mutant that hits either limit as killed: 5 for the properties and 7 for the tests.

Of the 1,348 mutants that compiled:

|                      | Killed | Rate  |
| -------------------- | ------ | ----- |
| The properties       | 930    | 69.0% |
| arrow-select's tests | 1,009  | 74.9% |
| Either               | 1,098  | 81.5% |

| File                            | Killed by the properties / the tests, of viable |
| ------------------------------- | ----------------------------------------------- |
| `take.rs`                       | 283 / 240 of 318                                |
| `filter.rs`                     | 113 / 121 of 156                                |
| `zip.rs`                        | 94 / 107 of 134                                 |
| `coalesce.rs`                   | 66 / 77 of 114                                  |
| `coalesce/byte_view.rs`         | 40 / 83 of 110                                  |
| `interleave.rs`                 | 96 / 93 of 101                                  |
| `union_extract.rs`              | 38 / 60 of 73                                   |
| `concat.rs`                     | 48 / 42 of 67                                   |
| `merge.rs`                      | 49 / 45 of 53                                   |
| `dictionary.rs`                 | 40 / 36 of 52                                   |
| `coalesce/list.rs`              | 7 / 15 of 45                                    |
| `coalesce/byte.rs`              | 5 / 12 of 30                                    |
| `coalesce/primitive.rs`         | 11 / 27 of 28                                   |
| `nullif.rs`                     | 20 / 20 of 20                                   |
| `coalesce/generic.rs`           | 3 / 12 of 15                                    |
| `window.rs`                     | 11 / 11 of 12                                   |
| `coalesce/boolean.rs`           | 4 / 4 of 12                                     |
| `coalesce/fixed_size_binary.rs` | 2 / 4 of 8                                      |

**89 mutants were killed only by the properties.** Most are in `take_bits` and `take_bits_with_validity` (Boolean `take` at bit offsets, 27), the offset arithmetic of `take_list` (13), `masked_primitives_to_bytes` in dictionary merging (8), and the coalescer's documented bypass size (7).

**168 mutants were killed only by the tests.** Each group below was checked against the code at 70d8190350:

| Count | What they change                                                                                                                                                                                                                                  | Why the properties miss them                                                                                                                                                                                                                                                                             |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 88    | A path, heuristic or capacity choice: `zip`'s interleave decision, filter strategies, the coalescer's direct-copy thresholds, buffer and block sizes, `union_extract`'s fast paths                                                                | Equivalent: each path gives the same values, so no property can tell them apart. Two `union_extract` mutants are the exception: they need a shared slot and a gap at exact rows, an input tuned to the implementation, which [rule 7](../../docs/design.md#rules) leaves out                             |
| 26    | `size()` and the buffer bookkeeping only `size()` reads                                                                                                                                                                                           | Equivalent: no property checks reported memory                                                                                                                                                                                                                                                           |
| 11    | `append_values_by_slices` (7), the fixed-size binary filter copy (2), the view builder's `copy_rows_by_filter` (2)                                                                                                                                | Equivalent: unreachable through the public API. The fused filter copy needs at most 1/16 of rows selected and a slices selection more than 0.8, `FixedSizeBinary` never takes the fused copy, and the view builder overrides the only caller                                                             |
| 16    | How the view builder copies long strings into its buffers                                                                                                                                                                                         | Most only change which buffer holds which bytes. Five were killed in an earlier run at seed 1 whose generator drew different cases, so the inputs that kill them are reachable but rare                                                                                                                  |
| 9     | Dictionary merging near the key type's capacity (4), `take` of a nullable run-end or union child whose children cannot hold a null (3), the coalescer's direct copy of offset-based strings (1), `take_list` over a child with its own offset (1) | Reachable but rare inputs                                                                                                                                                                                                                                                                                |
| 1     | `push_batch_interleaved`                                                                                                                                                                                                                          | No property calls it. It is an uncovered entry point                                                                                                                                                                                                                                                     |
| 1     | `filter` with a predicate shorter than the array                                                                                                                                                                                                  | Outside the documented domain: the rustdoc of `filter` does not say what a shorter predicate does                                                                                                                                                                                                        |
| 9     | The list builder's flush limits: `MAX_PENDING_SOURCE_BYTES`, `sparse_limit_reached`, and the flush condition in `copy_rows_by_filter_from`                                                                                                        | Equivalent: they change when pending sparse rows are flushed, not the rows or their order. Only `size()` sees them                                                                                                                                                                                       |
| 4     | The byte builder's `checked_values_len` returning `Ok(0)` or `Ok(1)`, or with `>` replaced by `==` or `>=`                                                                                                                                        | Equivalent on any input that fits in memory: they differ only when one batch holds `i32::MAX` bytes of values                                                                                                                                                                                            |
| 3     | The byte builder's `copy_rows_by_filter_from` and `append_filter_indices` replaced by `Ok(())`, and `>` replaced by `<` in `checked_values_len`                                                                                                   | Reachable but rare. The sparse copy of a byte column needs every column to have a direct filter copy and a mask of at least 1,024 rows that keeps at most 1 in 1,024, which about 2 in 1,000 cases have. `coalesce` at seeds 1 to 20, 1,000 cases each, kills all three on arrow-rs `main` at 9ffd63d528 |

The 250 mutants that neither side kills were not checked one by one.

# arrow-pbt

[![CI](https://github.com/mbutrovich/arrow-pbt/actions/workflows/ci.yml/badge.svg)](https://github.com/mbutrovich/arrow-pbt/actions/workflows/ci.yml)

Property-based tests for [arrow-rs](https://github.com/apache/arrow-rs) and the projects built on it. One generator produces valid Arrow arrays of every type in every physical layout the Arrow format allows. An independent oracle reads any array into plain values. Short properties check the compute kernels against both. A failing case shrinks to a small input and reports the seed that reproduces it.

The properties have found 42 bugs: 38 in arrow-rs `main`, one in atoi, which arrow-cast uses to parse integers, and three in DataFusion `main`. Each was reported with a reproduction or linked to an existing issue. 23 of the 42 are fixed. [docs/known-bugs.md](docs/known-bugs.md) lists them.

arrow-pbt is an independent project. It is not part of Apache Arrow, and the Apache Software Foundation does not endorse it.

## Who it is for

- **arrow-rs contributors** can run the properties against a branch before merging a risky change, such as a new kernel or a rewrite of a fast path, and against `main` every night. [Choosing the commits under test](#choosing-the-commits-under-test) says how.
- **Projects built on arrow-rs**, such as DataFusion, can test their own functions on generated arrays through an adapter crate. [docs/design.md](docs/design.md#downstream-adapters) describes adapters.
- **Contributors to arrow-pbt** add properties and adapters. [CONTRIBUTING.md](CONTRIBUTING.md) says how.

## Getting started

You need Rust 1.88 or later for arrow-rs's properties, and 1.95 or later for the whole workspace, which builds the DataFusion adapter. `Cargo.toml` names the arrow-rs and DataFusion commits to test, so you need no checkout of either:

```sh
git clone https://github.com/mbutrovich/arrow-pbt.git
cd arrow-pbt
cargo test --test properties          # arrow-rs
cargo test -p arrow-pbt-datafusion    # DataFusion
```

The first build fetches and compiles arrow-rs, and DataFusion for the second command.

## Reading a failure

A failing property prints the seed and case count that reproduce it, the failure, the minimal input that shrinking found, and the command that reruns it. This example is a known bug whose filter `ARROW_PBT_NO_KNOWN=1` turned back into a failure, so the message names its issue. A new bug's message is the property's own, such as the first row that differs from the model:

```text
arity failed with seed 1 after 2 cases: known bug, filter disabled: known: binary_mut returns Ok for an empty shared array (apache/arrow-rs#12002)
minimal input: (PrimitiveArray<Int8> [ ], PrimitiveArray<Int8> [ ])
rerun with: ARROW_PBT_SEEDS=1 ARROW_PBT_CASES=256 ARROW_PBT_NO_KNOWN=1 cargo test -p arrow-pbt arity
```

The same seed and case count always give the same case. [When a property fails](docs/workflow.md#when-a-property-fails) says how to decide whose bug it is and how to report it.

## Choosing the commits under test

`Cargo.toml` depends on `arrow-* = "60"` and on DataFusion's crates at the version on DataFusion `main`. Its `[patch.crates-io]` section replaces both with one commit of each project's `main`. The arrow-rs patch covers every crate in the workspace, including the arrow crates DataFusion depends on, so the workspace builds one arrow. The DataFusion commit must build on that arrow-rs commit.

To test a branch, a local checkout, or the latest `main` of either project, override its patch in a `.cargo/config.toml`, which is not committed. Cargo uses the config's patch over the one in `Cargo.toml`:

```sh
cp .cargo/config.toml.example .cargo/config.toml
# edit the paths in .cargo/config.toml to point at the checkouts, and delete
# the entries of a project you don't override
```

To test a branch on GitHub without a checkout, give each crate the git source that the example's comment shows instead of a path.

The override rewrites `Cargo.lock` to the override's source. Don't commit that change, and delete `.cargo/config.toml` to go back to the pinned commits.

Each project's version must match its requirement in `Cargo.toml`. For arrow-rs, Cargo otherwise warns that the patch was not used and builds the crates.io release instead. DataFusion's requirement is exact, because crates.io has DataFusion releases of the same major version on an older arrow, so a mismatch is a resolution error instead of a silent fallback to them. [docs/workflow.md](docs/workflow.md#moving-the-commits) says how to move the commits.

## Running

Every property in the workspace, arrow-rs's and the adapters', goes through one runner, `run::check`. By default it runs 256 cases with seed 1. These variables change what `cargo test` and the binary run and print:

| Variable             | Meaning                                                        | Default                   | Used by      |
| -------------------- | -------------------------------------------------------------- | ------------------------- | ------------ |
| `ARROW_PBT_CASES`    | Cases per property and seed                                    | 256                       | `cargo test` |
| `ARROW_PBT_SEEDS`    | A seed, or a range such as `1-10`                              | `1`                       | `cargo test` |
| `ARROW_PBT_SHRINK`   | Shrink steps after a failure                                   | 1000                      | `cargo test` |
| `ARROW_PBT_REASONS`  | Print discards by reason when set                              | unset                     | both         |
| `ARROW_PBT_NO_KNOWN` | Fail instead of discarding known bugs when set                 | unset                     | both         |
| `ARROW_PBT_FULL`     | Print messages and inputs without truncation when set          | unset                     | both         |
| `ARROW_PBT_THREADS`  | Workers                                                        | the available parallelism | the binary   |
| `ARROW_PBT_PANICS`   | Print panics as they happen when set, such as in the generator | unset                     | the binary   |

The binary takes the seeds, cases and shrink steps as arguments instead, and `cargo test` takes its workers from `RUST_TEST_THREADS`.

A value that does not parse, such as `ARROW_PBT_CASES=1e5`, is an error rather than a fallback to the default. A run that discards too many cases fails too.

The binary is for longer runs and reports. `run` prints one tab-separated row per property and seed, with no header: the property, the seed, whether it failed, the cases to the failure (`-` if none), the cases run, the milliseconds, the failure message, the minimal input, and the discards. `ARROW_PBT_REASONS=1` adds a `discard` row for each reason under it. `list` prints the names that `run` takes.

```sh
# every property, seeds 1 to 3, 1000 cases each
ARROW_PBT_REASONS=1 cargo run -q -- run all 1-3 1000
# the property names
cargo run -q -- list
# a few properties
cargo run -q -- run take,filter 1-10 10000
# what the generator produces
cargo run -q -- coverage 1 2000
cargo run -q -- inventory 1 2000 > inventory.md
```

On a machine with performance and efficiency cores, set `ARROW_PBT_THREADS` to the performance cores to compare one property's time between runs, since the efficiency cores are slower.

## Run sizes

A case costs 11.5 ms of CPU on average in a debug build, from 0.4 ms for `kleene` to 51 ms for `coalesce`. Debug builds keep Rust's overflow checks, which found two of the bugs. The binary runs one property and seed per worker, so wall time is about the total CPU time divided by the workers.

| Use                                    | Cases per property and seed | Seeds |
| -------------------------------------- | --------------------------- | ----- |
| Quick check (the test default)         | 256                         | 1     |
| Checking a refactor                    | 1,000                       | 3     |
| Nightly, or before an arrow-rs release | 100,000                     | 1     |

At the average cost above, every property at the nightly size takes about 14 hours of CPU time. Most bugs appeared within 3,000 cases per property. One needed about 24,000.

## Layout

| Path                                  | Contents                                                                                                                                                                                                   |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/generator.rs`                    | The generator: types, logical rows, and the physical layout chosen by a seed                                                                                                                               |
| `src/val.rs`                          | The oracle: any array as a tree of logical values                                                                                                                                                          |
| `src/spec.rs`                         | The checker of the Arrow columnar format                                                                                                                                                                   |
| `src/cases.rs`                        | Input strategies for each kernel                                                                                                                                                                           |
| `src/props.rs`                        | The shared runner, and `take`, `filter`, `interleave`, `concat`, `shift`, `union_extract`, `slice`, `eq`, `data_ops`, `cmp`, `sort`, `cast`, `arith`, `bools`, the `law_*` properties, and the round trips |
| `src/batches.rs`                      | `coalesce`, `concat_batches`, `interleave_record_batch` and `take_record_batch`                                                                                                                            |
| `src/logic.rs`                        | `is_null`, `kleene`, `merge`, `merge_n`, `nullif` and `zip`                                                                                                                                                |
| `src/numeric.rs`                      | `aggregate`, `arity`, `bitwise`, `decimal_format`, `fixed_point` and `neg`                                                                                                                                 |
| `src/order.rs`                        | `lexsort` and `row_format`                                                                                                                                                                                 |
| `src/strings.rs`                      | `cast_text`, `length` and `string_ops`                                                                                                                                                                     |
| `src/temporal.rs`                     | `interval_scale`, `temporal_arith` and `temporal_cast`                                                                                                                                                     |
| `src/known.rs`                        | The known-bug discard and its switch                                                                                                                                                                       |
| `src/registry.rs`                     | Every property with its strategy, and the runners                                                                                                                                                          |
| `src/coverage.rs`, `src/inventory.rs` | Measurements of what the generator produces                                                                                                                                                                |
| `src/main.rs`                         | The `arrow-pbt` binary for long runs and reports                                                                                                                                                           |
| `tests/properties.rs`                 | One `#[test]` per property                                                                                                                                                                                 |
| `tests/cli.rs`                        | Tests of the binary's `list` and `help` commands                                                                                                                                                           |
| `tests/generator.rs`                  | Tests of the generator's conversion from a `DataType` or `Field`, and of where it places values buffers                                                                                                    |
| `arrow-pbt-datafusion/`               | The DataFusion adapter (`src/lib.rs`) and its laws for DataFusion's scalar functions (`tests/functions.rs`)                                                                                                |
| `docs/`                               | Design, workflow, known bugs, and coverage                                                                                                                                                                 |
| `tools/mutation/`                     | Mutation testing scripts and the arrow-select results                                                                                                                                                      |

The library's public API is `generator`, `val`, `spec`, `types`, `known` and `run`, and `cargo doc --open` shows it with an example for each item. The other modules are internal to this crate's tests and binary.

## More

- [docs/design.md](docs/design.md): the terms used here, the generator, the oracle, the properties, the rules that keep them honest, and how downstream projects use the crate.
- [docs/workflow.md](docs/workflow.md): what to do when a property fails, how to keep finding bugs while one waits for a fix, and how to add a property or an adapter.
- [docs/known-bugs.md](docs/known-bugs.md): every bug the properties found, and the filters of the open ones.
- [docs/coverage.md](docs/coverage.md): every arrow-rs entry point, what covers it, and what was measured.
- [CONTRIBUTING.md](CONTRIBUTING.md): the checks to run before a pull request, and where to report a bug the properties find.

## License

Apache-2.0, as arrow-rs.

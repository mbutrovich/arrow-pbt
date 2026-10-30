# arrow-pbt

[![CI](https://github.com/mbutrovich/arrow-pbt/actions/workflows/ci.yml/badge.svg)](https://github.com/mbutrovich/arrow-pbt/actions/workflows/ci.yml)

Property-based tests for [arrow-rs](https://github.com/apache/arrow-rs) and the projects built on it. One generator produces valid Arrow arrays of every type in every physical layout the Arrow format allows. An independent oracle reads any array into plain values. Short properties check the compute kernels against both. A failing case shrinks to a small input and reports the seed that reproduces it.

The properties have found 39 bugs on arrow-rs `main`. Each was reported with a reproduction or linked to an existing issue. 23 of the 39 are fixed. [docs/known-bugs.md](docs/known-bugs.md) lists them.

arrow-pbt is an independent project. It is not part of Apache Arrow, and the Apache Software Foundation does not endorse it.

## Who it is for

- **arrow-rs contributors** can run the properties against a branch before merging a risky change, such as a new kernel or a rewrite of a fast path, and against `main` every night. [Choosing the arrow-rs under test](#choosing-the-arrow-rs-under-test) says how.
- **Projects built on arrow-rs**, such as DataFusion, can test their own functions on generated arrays through an adapter crate. [docs/design.md](docs/design.md#downstream-adapters) describes adapters.
- **Contributors to arrow-pbt** add properties and adapters. [CONTRIBUTING.md](CONTRIBUTING.md) says how.

## Getting started

You need Rust 1.88 or later for arrow-rs's properties, and 1.95 or later for the whole workspace, which builds the DataFusion adapter. `Cargo.toml` names the arrow-rs commit to test, so you need no checkout of arrow-rs:

```sh
git clone https://github.com/mbutrovich/arrow-pbt.git
cd arrow-pbt
cargo test --test properties          # arrow-rs
cargo test -p arrow-pbt-datafusion    # DataFusion
```

The first build fetches and compiles arrow-rs, and DataFusion for the second command.

## Reading a failure

A failing property prints the seed and case count that reproduce it, the failure, the minimal input that shrinking found, and the command that reruns it:

```text
arity failed with seed 1 after 2 cases: known bug, filter disabled: known: binary_mut returns Ok for an empty shared array (apache/arrow-rs#12002)
minimal input: (PrimitiveArray<Int8> [ ], PrimitiveArray<Int8> [ ])
rerun with: ARROW_PBT_SEEDS=1 ARROW_PBT_CASES=256 cargo test -p arrow-pbt arity
```

The same seed and case count always give the same case. [When a property fails](docs/workflow.md#when-a-property-fails) says how to decide whose bug it is and how to report it.

## Choosing the arrow-rs under test

`Cargo.toml` depends on `arrow-* = "60"`, and its `[patch.crates-io]` section replaces those crates with one commit of arrow-rs `main`. The patch covers every crate in the workspace, including the arrow crates DataFusion depends on, so the workspace builds one arrow.

To test a branch, a local checkout, or the latest `main`, override the patch in a `.cargo/config.toml`, which is not committed. Cargo uses the config's patch over the one in `Cargo.toml`:

```sh
cp .cargo/config.toml.example .cargo/config.toml
# edit the paths in .cargo/config.toml to point at the arrow-rs checkout
```

The override rewrites `Cargo.lock` to the checkout's path. Don't commit that change, and delete `.cargo/config.toml` to go back to the pinned commit.

The arrow-rs version must match the requirement in `Cargo.toml`. Otherwise Cargo warns that the patch was not used and builds the crates.io release instead. [docs/workflow.md](docs/workflow.md#moving-the-arrow-rs-commit) says how to move the commit.

## Running

Every property in the workspace, arrow-rs's and the adapters', goes through one runner, `run::check`. By default it runs 256 cases with seed 1, and these variables change that:

| Variable             | Meaning                                        | Default |
| -------------------- | ---------------------------------------------- | ------- |
| `ARROW_PBT_CASES`    | Cases per property and seed                    | 256     |
| `ARROW_PBT_SEEDS`    | A seed, or a range such as `1-10`              | `1`     |
| `ARROW_PBT_SHRINK`   | Shrink steps after a failure                   | 1000    |
| `ARROW_PBT_REASONS`  | Print discards by reason when set              | unset   |
| `ARROW_PBT_NO_KNOWN` | Fail instead of discarding known bugs when set | unset   |

A value that does not parse, such as `ARROW_PBT_CASES=1e5`, is an error rather than a fallback to the default. A run that discards too many cases fails too.

The binary is for longer runs and reports:

```sh
# every property, seeds 1 to 3, 1000 cases each, as TSV
ARROW_PBT_REASONS=1 cargo run -q -- run all 1-3 1000
# a few properties
cargo run -q -- run take,filter 1-10 10000
# what the generator produces
cargo run -q -- coverage 1 2000
cargo run -q -- inventory 1 2000 > inventory.md
```

`ARROW_PBT_THREADS` sets the workers. The default is the available parallelism, as for `cargo test`. On a machine with performance and efficiency cores, set it to the performance cores to compare one property's time between runs, since the efficiency cores are slower. `ARROW_PBT_FULL=1` prints messages without truncation, and `ARROW_PBT_PANICS=1` keeps panic messages.

## Run sizes

A case costs 11.5 ms of CPU on average in a debug build, from 0.4 ms for `kleene` to 51 ms for `coalesce`. Debug builds keep Rust's overflow checks, which found two of the bugs. The binary runs one property and seed per worker, so wall time is about the total CPU time divided by the workers.

| Use                                    | Cases per property and seed | Seeds |
| -------------------------------------- | --------------------------- | ----- |
| Quick check (the test default)         | 256                         | 1     |
| Checking a refactor                    | 1,000                       | 3     |
| Nightly, or before an arrow-rs release | 100,000                     | 1     |

Most bugs appeared within 3,000 cases per property. One needed about 24,000.

## Layout

| Path                                                                                                                    | Contents                                                                                                      |
| ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `src/generator.rs`                                                                                                      | The generator: types, logical rows, and the physical layout chosen by a seed                                  |
| `src/val.rs`                                                                                                            | The oracle: any array as a tree of logical values                                                             |
| `src/spec.rs`                                                                                                           | The checker of the Arrow columnar format                                                                      |
| `src/cases.rs`                                                                                                          | Input strategies for each kernel                                                                              |
| `src/props.rs`, `src/batches.rs`, `src/logic.rs`, `src/numeric.rs`, `src/order.rs`, `src/strings.rs`, `src/temporal.rs` | The properties                                                                                                |
| `src/known.rs`                                                                                                          | The known-bug discard and its switch                                                                          |
| `src/registry.rs`                                                                                                       | Every property with its strategy, and the runners                                                             |
| `src/coverage.rs`, `src/inventory.rs`                                                                                   | Measurements of what the generator produces                                                                   |
| `src/main.rs`                                                                                                           | The `arrow-pbt` binary for long runs and reports                                                              |
| `tests/properties.rs`                                                                                                   | One `#[test]` per property                                                                                    |
| `tests/generator.rs`                                                                                                    | Tests of the generator's conversion from a `DataType` or `Field`, and of where it places values buffers       |
| `arrow-pbt-datafusion/`                                                                                                 | The DataFusion adapter (`src/lib.rs`) and the broadcast law for a few scalar functions (`tests/functions.rs`) |
| `docs/`                                                                                                                 | Design, workflow, known bugs, and coverage                                                                    |
| `tools/mutation/`                                                                                                       | Mutation testing scripts and the arrow-select results                                                         |

The library's public API is `generator`, `val`, `spec`, `types`, `known` and `run`, and `cargo doc --open` shows it with an example for each item. The other modules are internal to this crate's tests and binary.

## More

- [docs/design.md](docs/design.md): the terms used here, the generator, the oracle, the properties, the rules that keep them honest, and how downstream projects use the crate.
- [docs/workflow.md](docs/workflow.md): what to do when a property fails, how to keep finding bugs while one waits for a fix, and how to add a property or an adapter.
- [docs/known-bugs.md](docs/known-bugs.md): every bug the properties found, and the filters of the open ones.
- [docs/coverage.md](docs/coverage.md): every arrow-rs entry point, what covers it, and what was measured.
- [CONTRIBUTING.md](CONTRIBUTING.md): the checks to run before a pull request, and where to report a bug the properties find.

## License

Apache-2.0, as arrow-rs.

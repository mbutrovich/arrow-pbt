# Contributing

## Reporting a bug the properties find

A bug in arrow-rs, or in a project an adapter tests, goes to that project's issue tracker, not to this repository. Follow [Reporting bugs upstream](README.md#reporting-bugs-upstream) in the README, which asks you to review each report before filing it and to own the issue until a fix lands. [When a property fails](docs/workflow.md#when-a-property-fails) says how to decide whose bug it is, how to reduce the case to a reproduction, and what the issue needs. Once the issue is filed, [Keeping the bug from hiding others](docs/workflow.md#keeping-the-bug-from-hiding-others) says how to filter it here.

Issues and pull requests in this repository are for arrow-pbt itself: a property that is wrong, a gap in the generator, or a new property or adapter.

## Adding a property or an adapter

[Adding a property](docs/workflow.md#adding-a-property) and [Adding an adapter](docs/workflow.md#adding-an-adapter) give the steps. [The rules](docs/design.md#rules) in the design say what keeps a property's results meaningful, such as checking an oracle by breaking it on purpose.

## Layout

| Path                                  | Contents                                                                                                                                                                                                              |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/generator.rs`                    | The generator: types, logical rows, and the physical layout chosen by a seed                                                                                                                                          |
| `src/val.rs`                          | The oracle: any array as a tree of logical values                                                                                                                                                                     |
| `src/spec.rs`                         | The checker of the Arrow columnar format                                                                                                                                                                              |
| `src/cases.rs`                        | Input strategies for each kernel                                                                                                                                                                                      |
| `src/props.rs`                        | The shared runner, and `take`, `filter`, `interleave`, `concat`, `shift`, `union_extract`, `slice`, `eq`, `data_ops`, `cmp`, `sort`, `cast`, `arith`, `bools`, the `law_*` properties, and the round trips            |
| `src/batches.rs`                      | `coalesce`, `concat_batches`, `interleave_record_batch` and `take_record_batch`                                                                                                                                       |
| `src/logic.rs`                        | `is_null`, `kleene`, `merge`, `merge_n`, `nullif` and `zip`                                                                                                                                                           |
| `src/numeric.rs`                      | `aggregate`, `arity`, `bitwise`, `decimal_format`, `decimal_precision`, `divisor`, `fixed_point` and `neg`                                                                                                            |
| `src/order.rs`                        | `lexsort` and `row_format`, and `check_ascending`, which `val` exports                                                                                                                                                |
| `src/strings.rs`                      | `cast_text`, `length` and `string_ops`                                                                                                                                                                                |
| `src/temporal.rs`                     | `interval_scale`, `temporal_arith` and `temporal_cast`                                                                                                                                                                |
| `src/known.rs`                        | The known-bug discard and its switch                                                                                                                                                                                  |
| `src/registry.rs`                     | Every property with its strategy, and the runners                                                                                                                                                                     |
| `src/coverage.rs`, `src/inventory.rs` | Measurements of what the generator produces                                                                                                                                                                           |
| `src/main.rs`                         | The `arrow-pbt` binary for long runs and reports                                                                                                                                                                      |
| `tests/properties.rs`                 | One `#[test]` per property                                                                                                                                                                                            |
| `tests/cli.rs`                        | Tests of the binary's `list` and `help` commands                                                                                                                                                                      |
| `tests/generator.rs`                  | Tests of the generator's conversion from a `DataType` or `Field`, and of where it places values buffers                                                                                                               |
| `arrow-pbt-datafusion/`               | The DataFusion adapter (`src/lib.rs`, `src/expr.rs`, `src/join.rs`), its laws for DataFusion's scalar functions (`tests/functions.rs`), its physical expressions (`tests/exprs.rs`), and its joins (`tests/joins.rs`) |
| `docs/`                               | Running, design, workflow, known bugs, and coverage                                                                                                                                                                   |
| `tools/mutation/`                     | Mutation testing scripts and the arrow-select results                                                                                                                                                                 |

The library's public API is `generator`, `val`, `spec`, `types`, `known` and `run`, and `cargo doc --open` shows it with an example for each item. The other modules are internal to this crate's tests and binary.

## Checks before a pull request

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
npx prettier@3.9.10 --check README.md CONTRIBUTING.md 'docs/**/*.md' 'tools/**/*.md'
typos
```

`rust-toolchain.toml` pins the toolchain arrow-rs uses, so that its lints, which this workspace uses, give the same results, and rustup installs it on first use. `--write` in place of `--check` formats the Markdown. Prettier runs with its defaults, so it needs Node.js but no configuration. `typos` is the spell checker arrow-rs runs, at the version CI pins, `cargo install typos-cli --version 1.37.0`. Without `--all` or `--workspace`, the cargo commands cover only `arrow-pbt`. `cargo test --workspace` runs the property tests at 256 cases each, the generator's tests, the adapters' tests, and the doctests. The lints and formatting settings come from arrow-rs. CI runs the same checks on every push to `main` and every pull request.

A change that should not change behavior, such as a rename or moving code, must also leave every property's results the same, as [Checking a refactor](docs/workflow.md#checking-a-refactor) describes.

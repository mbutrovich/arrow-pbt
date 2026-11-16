# Running

This guide is for running the properties: against a branch before a risky change, against `main` every night, or for a report of what the generator produces. [CONTRIBUTING.md](../CONTRIBUTING.md) covers changing arrow-pbt itself.

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

Each project's version must match its requirement in `Cargo.toml`. For arrow-rs, Cargo otherwise warns that the patch was not used and builds the crates.io release instead. DataFusion's requirement is exact, because crates.io has DataFusion releases of the same major version on an older arrow, so a mismatch is a resolution error instead of a silent fallback to them. [docs/workflow.md](workflow.md#moving-the-commits) says how to move the commits.

## Settings and the binary

Every property in the workspace, arrow-rs's and the adapters', goes through one runner, `run::check`. By default it runs 256 cases with seed 1. These variables change what `cargo test` and the binary run and print:

| Variable             | Meaning                                                        | Default                   | Used by      |
| -------------------- | -------------------------------------------------------------- | ------------------------- | ------------ |
| `ARROW_PBT_CASES`    | Cases per property and seed                                    | 256                       | `cargo test` |
| `ARROW_PBT_SEEDS`    | A seed, or a range such as `1-10`                              | `1`                       | `cargo test` |
| `ARROW_PBT_SHRINK`   | Shrink steps after a failure                                   | 1000                      | `cargo test` |
| `ARROW_PBT_REASONS`  | Print discards by reason when set                              | unset                     | both         |
| `ARROW_PBT_NO_KNOWN` | Fail instead of discarding known bugs when set                 | unset                     | both         |
| `ARROW_PBT_FULL`     | Print messages and inputs without truncation when set          | unset                     | both         |
| `ARROW_PBT_THREADS`  | Seeds that run at once                                         | the available parallelism | both         |
| `ARROW_PBT_PANICS`   | Print panics as they happen when set, such as in the generator | unset                     | the binary   |

The binary takes the seeds, cases and shrink steps as arguments instead. Under `cargo test`, every test hands its seeds to one pool of `ARROW_PBT_THREADS` workers, so a run with many seeds keeps every core busy, and `RUST_TEST_THREADS` only sets how many tests start at once. A seed is the smallest unit of work, so the seeds of one test run in parallel but the cases of one seed do not. libtest captures the output of a test that passes, so under `cargo test` the discards that `ARROW_PBT_REASONS` prints show only for a failing test, or with `cargo test -- --nocapture`.

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

A case costs 6.3 ms of CPU on average in a debug build, from 0.4 ms for `kleene` to 27 ms for `coalesce`. Debug builds keep Rust's overflow checks, which found two of the bugs. The binary and `cargo test` both run one property and seed per worker, so wall time is about the total CPU time divided by the workers, as long as there are more seeds than workers. A long run therefore uses many seeds rather than many cases per seed.

| Use                                    | Cases per property and seed | Seeds |
| -------------------------------------- | --------------------------- | ----- |
| Quick check (the test default)         | 256                         | 1     |
| Checking a refactor                    | 1,000                       | 3     |
| Nightly, or before an arrow-rs release | 1,000                       | 100   |

At the average cost above, every property at the nightly size takes about 8 hours of CPU time. Most bugs appeared within 3,000 cases per property. One needed about 24,000.

## Running regularly

A nightly run against arrow-rs `main` and DataFusion `main` finds bugs within a day of the merge that introduced them. Point `.cargo/config.toml` at up-to-date checkouts of both and run many cases, spread over many seeds so that every core has work. Seeds taken from the date explore new cases each night and are easy to replay:

```sh
git -C /path/to/arrow-rs pull
git -C /path/to/datafusion pull
ARROW_PBT_REASONS=1 cargo run -q -- run all $(date +%Y%m%d)00-$(date +%Y%m%d)99 1000 > nightly-$(date +%F).tsv
ARROW_PBT_SEEDS=$(date +%Y%m%d)00-$(date +%Y%m%d)99 ARROW_PBT_CASES=1000 cargo test -p arrow-pbt-datafusion
```

The binary runs arrow-rs's properties, and `cargo test` runs an adapter's. When DataFusion `main` does not build on arrow-rs `main`, delete DataFusion's entries from `.cargo/config.toml` to keep it on its pinned commit, or skip the second run.

On macOS, prefix a long run with `caffeinate -i`, so that idle sleep does not pause it.

Before merging a risky change to arrow-rs, run the properties for the touched crate against the branch with a large case count. For example, a change to decimal division would run `arith,cast,fixed_point,decimal_format`. For a change to a DataFusion function, point `.cargo/config.toml` at the branch and run that function's test with `ARROW_PBT_CASES` in the thousands, such as `cargo test -p arrow-pbt-datafusion lpad`.

## Validating a release candidate

arrow-rs tags each release candidate as `<version>-rc<n>`, such as `60.1.0-rc1`. To run the properties on one, give each arrow-rs crate in `.cargo/config.toml` the tag as its git source, and delete the DataFusion entries:

```toml
[patch.crates-io]
arrow = { git = "https://github.com/apache/arrow-rs", tag = "60.1.0-rc1" }
arrow-arith = { git = "https://github.com/apache/arrow-rs", tag = "60.1.0-rc1" }
# and the same for every other arrow-rs crate in .cargo/config.toml.example
```

Then run every property at the nightly size from [Run sizes](#run-sizes):

```sh
ARROW_PBT_REASONS=1 cargo run -q -- run all 1-100 1000 > rc.tsv
```

The DataFusion adapter tests DataFusion's code, so an arrow-rs candidate needs only this run. Three things limit which candidates it can check:

- The candidate must contain the arrow-rs commit that `Cargo.toml` names. A known-bug filter is removed when its fix lands on arrow-rs `main`, so on a candidate cut before a fix, the property fails on a bug that [known-bugs.md](known-bugs.md) lists as fixed. `git merge-base --is-ancestor <commit> <tag>` in an arrow-rs checkout checks it.
- A candidate of a new major version needs the `arrow-*` requirement in `Cargo.toml` raised to match, as step 2 of [Moving the commits](workflow.md#moving-the-commits) says. Otherwise Cargo warns that the patch was not used and tests the crates.io release.
- Candidates on an older release line, such as a 59.x maintenance release, cannot be checked. The `coalesce` property uses `BatchCoalescer` and `FilterBuilder` methods first released in 60.1.0.

## When a property fails

[When a property fails](workflow.md#when-a-property-fails) says how to decide whose bug it is, and [Reporting bugs upstream](../README.md#reporting-bugs-upstream) says what a report needs.

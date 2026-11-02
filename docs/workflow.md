# Workflow

## Running regularly

A nightly run against arrow-rs `main` and DataFusion `main` finds bugs within a day of the merge that introduced them. Point `.cargo/config.toml` at up-to-date checkouts of both and run many cases. A seed taken from the date explores new cases each night and is easy to replay:

```sh
git -C /path/to/arrow-rs pull
git -C /path/to/datafusion pull
ARROW_PBT_REASONS=1 cargo run -q -- run all $(date +%Y%m%d) 100000 > nightly-$(date +%F).tsv
ARROW_PBT_SEEDS=$(date +%Y%m%d) ARROW_PBT_CASES=100000 cargo test -p arrow-pbt-datafusion
```

The binary runs arrow-rs's properties, and `cargo test` runs an adapter's. When DataFusion `main` does not build on arrow-rs `main`, delete DataFusion's entries from `.cargo/config.toml` to keep it on its pinned commit, or skip the second run.

On macOS, prefix a long run with `caffeinate -i`, so that idle sleep does not pause it.

Before merging a risky change to arrow-rs, run the properties for the touched crate against the branch with a large case count. For example, a change to decimal division would run `arith,cast,fixed_point,decimal_format`. For a change to a DataFusion function, point `.cargo/config.toml` at the branch and run that function's test with `ARROW_PBT_CASES` in the thousands, such as `cargo test -p arrow-pbt-datafusion lpad`.

## When a property fails

The failure reports the property, the seed, the shrunk input, and a rerun command. Steps:

1. **Rerun it.** `ARROW_PBT_FULL=1` prints the input without truncation, and `ARROW_PBT_PANICS=1` keeps panic messages. The same seed and case count always give the same case.
2. **Decide whose bug it is.** The property can be wrong, or the generator, or the project under test: arrow-rs, or the project an adapter tests. Read the Arrow format specification and the kernel's rustdoc, or the function's documentation, for the behavior in question. If they are silent, look for an issue or pull request in that project where a maintainer settled it. A settled position is not a bug. Encode it in the property as the expected behavior, with a comment that cites the issue. Examples in the code are integer to `Date64` casts keeping out-of-range values ([apache/arrow-rs#5288](https://github.com/apache/arrow-rs/issues/5288)) and `T::Native::MIN` as an unsupported decimal division operand ([apache/arrow-rs#10911](https://github.com/apache/arrow-rs/issues/10911)).
3. **Check upstream `main`.** The commits in `Cargo.toml` or the checkouts under test may be older. Point `.cargo/config.toml` at current `main` and rerun.
4. **Search for an existing issue.** `gh search issues --repo apache/arrow-rs "<keywords>"`, or `--repo apache/datafusion` for a DataFusion function, and `gh search prs` with the function name. A related closed issue often explains the history.
5. **Reduce it by hand.** The shrunk input is usually still large: one shrunk case was a 1,000-row `Decimal128(38, 38)` array. Write a standalone reproduction, with the output of each call in a comment, as a scratch program outside this repository. It goes into the issue, and the fix's pull request turns it into a regression test in the project. Aim for one or two rows and the simplest type that shows the bug. This is most of the work per bug.
6. **Write the issue.** Use the project's bug template: what happens, the reproduction, the expected behavior with its source (the specification, the rustdoc, or a maintainer's comment), and the code at fault, linked with a permalink that includes a commit SHA. Ask a question where the intended behavior is unclear. The reproduction becomes the regression test in the fix's pull request.

## Keeping the bug from hiding others

A failing property stops at its first failure, so a known bug would mask every other bug that the same property could find. Filter the known bug so the run discards its cases and keeps going:

1. **Find the failure's signature.** This is a substring of the failure message, plus the condition that makes it this bug and no other: an operation, a type that appears in the message, or a value at a boundary. Never filter by input type alone, because that would also hide every other bug for that type.
2. **Return the discard through `known`.** In the property, or in the shared runner in `src/props.rs` when the failure shows up in many properties:

   ```rust
   Err(TestCaseError::Fail(m)) if m.to_string().contains("Cannot take null indices") => {
       Err(crate::known::known("known: take rejects a null index a parent could hold (apache/arrow-rs#11608)"))
   }
   ```

   Start the reason with `known: ` and end it with the issue, qualified by its repository (`apache/arrow-rs#11608`), so the discard counts say which bug ate the cases. The repository matters once downstream projects such as DataFusion and iceberg-rust use the crate and file their own issues.

3. **Add a comment above the filter.** It says what the bug is, where it was found (`found by this harness on main at <sha>`), and the issue.
4. **Add a row to the open table in [known-bugs.md](known-bugs.md).**
5. **Check what the filter costs.** Run the property with `ARROW_PBT_REASONS=1`. A filter that discards a large share of a property's cases also stops it from finding other bugs in those cases. The filter for [apache/arrow-rs#11607](https://github.com/apache/arrow-rs/issues/11607) discards about 10% of `row_format`'s cases.

When the bug aborts the process, as undefined behavior that a standard library check catches in a debug build does, its signature cannot be read after the call. The filter then checks the bug's precondition before the call, by calling the project on its own or by computing the condition from the input's values, never by input type. The filter for `take_record_batch` calls `take` on the first column and discards the case when it returns for an index out of bounds.

When the bug is in a path the generator itself uses to build arrays (FFI import, for example), the generator falls back to another layout and counts each fallback instead. `arrow-pbt coverage` prints those counts, so the gap stays visible.

## When a fix lands upstream

A fixed bug loses its filter and keeps its row in [known-bugs.md](known-bugs.md), which records every bug the properties found.

1. **Find the fixed bugs.** List the open filters' issues with `gh issue view <n> --repo apache/arrow-rs --json state,stateReason`. An issue closed as completed names the pull request that closed it. Its merge commit must be in the checkout under test, which `git -C /path/to/arrow-rs merge-base --is-ancestor <sha> HEAD` checks. A fix in a dependency, such as an atoi release, also needs this workspace's `Cargo.lock` to resolve the fixed version.
2. **Remove the filter.** Remove the discard, the comment above it, any generator fallback and its counter, and any check a property skipped because of the bug. Remove the helpers only the filter used. When the issue number appears nowhere in `src/` or `tests/`, the filter is gone.
3. **Run without it.** Run the properties the filter was in against a checkout with the fix, with many cases and the other filters still on:

   ```sh
   ARROW_PBT_REASONS=1 cargo run -q -- run take,coalesce 1-10 10000
   ```

   Run `all` when the filter was in a place every property goes through: `check_in`, the shared runner in `src/props.rs`, or the generator.

4. **Record it.** If nothing fails, move the row to the fixed table in [known-bugs.md](known-bugs.md), with the pull request and the commit the run checked. If a property fails, the fix is incomplete or the filter was hiding another bug. Follow [When a property fails](#when-a-property-fails).

`ARROW_PBT_NO_KNOWN=1` turns every filter into a failure. A property stops at its first failure, so it cannot show one bug fixed while another in the same property is open.

## Moving the commits

The `[patch.crates-io]` section of `Cargo.toml` names the arrow-rs and DataFusion commits a fresh clone tests. Move one when a fix lands, so that its filter can go, or when the properties need a new entry point:

1. Set the new commit as `rev` in every entry of that project in the section.
2. If the arrow-rs commit changes arrow-rs's major version, update the `arrow-*` requirement in `[workspace.dependencies]` to match. Otherwise Cargo warns that the patch was not used and builds the crates.io release. If the DataFusion commit changes DataFusion's version, update the exact `datafusion-*` requirement to that version. Otherwise Cargo fails to resolve.
3. If Cargo then cannot select a version of a dependency that the new commit needs, such as `libc`, update that dependency alone with `cargo update -p <name>`.
4. The DataFusion commit must build on the arrow-rs commit. If `arrow-pbt-datafusion` no longer builds after moving arrow-rs, move DataFusion to a commit that builds on it, such as one on its arrow upgrade branch.
5. Run the checks in [CONTRIBUTING.md](../CONTRIBUTING.md#checks-before-a-pull-request).
6. For arrow-rs, list the new entry points and update [coverage.md](coverage.md), as step 1 of [When arrow-rs changes](#when-arrow-rs-changes) says.

## Checking a refactor

A change that should not change behavior, such as a rename or moving code, must leave every property's results the same. Run every property for a few seeds before and after, and compare the output:

```sh
ARROW_PBT_REASONS=1 cargo run -q -- run all 1-3 1000 > before.tsv
# make the change
ARROW_PBT_REASONS=1 cargo run -q -- run all 1-3 1000 > after.tsv
```

The rows must match on whether each property failed, the cases it ran, and its discards by reason. Timings differ.

## Adding a property

For arrow-rs:

1. Write the property in the module that holds the properties most like it, which [Layout](../README.md#layout) in the README lists. Compute the expected output from the oracle (`val::vals`), call the kernel, and check the output with `val::check_rows`, which validates it and compares its values.
2. Give it a strategy in `src/cases.rs` that covers the kernel's documented domain, using the shared generator.
3. Register it in `src/registry.rs` and add a `#[test]` in `tests/properties.rs`.
4. Break the oracle on purpose a few ways and confirm the property fails each time. Build each broken copy into its own binary and compare the binaries' hashes, because Cargo reuses a build when the copied sources keep their old modification times.
5. Run it with a few thousand cases and read its discards by reason.

For a DataFusion scalar function, add a line to `law_tests!` in `arrow-pbt-datafusion/tests/functions.rs` with the function and its signatures, the argument types it accepts after coercion. The test first checks each signature with DataFusion's `fields_with_udf`, which leaves a signature unchanged only if the planner would pass it as it is, then runs `check_broadcast` and `check_row_by_row` through `run::check`. A function with a known bug calls `laws_with` instead, whose hooks hold the filters: one before the call, for a bug that aborts the process, and one that maps a failure to its discard. `repeat` uses the first, `ceil` the second, and `lpad` both. Run it with `ARROW_PBT_CASES` in the thousands.

## Adding an adapter

An adapter tests a project built on arrow-rs, through the four parts that [Downstream adapters](design.md#downstream-adapters) describes:

1. Create a member crate named `arrow-pbt-<project>`, and add it to `members` in the root `Cargo.toml`. Add the project's crates to `[workspace.dependencies]` at the version on its `main`, and put a commit of its `main` in `[patch.crates-io]`, as for DataFusion. Make the requirement exact when crates.io has releases of that version line on an older arrow. Once a release builds on the arrow-rs version the workspace pins, depend on the release and drop the patch. Never depend on a local checkout. The workspace's `[patch.crates-io]` already puts the project's arrow crates on the pinned arrow-rs, and `.cargo/config.toml.example` should gain the project's entries.
2. Use only the project's public API, so the adapter needs no change to the project.
3. Write the types the project accepts, the conversions to its own types, the case strategies, and a harness function for each law, in the crate's `src/lib.rs`. Each public item gets a doctest.
4. Write each property as a `#[test]` in the crate's `tests/` that runs a harness function through `run::check`. Run a harness function on an entry point broken on purpose, and check that its failure says why, as `arrow-pbt-datafusion/tests/functions.rs` does.
5. File the bugs it finds in the project's own repository, and qualify each known-bug reason with that repository, such as `(apache/datafusion#<n>)`.
6. Add the project to "Downstream projects" in [coverage.md](coverage.md).

## Measuring the properties

Passing properties say nothing until we know they would fail on a broken kernel. Mutation testing checks that. It plants one small bug at a time in an arrow-rs crate and records whether the properties catch it and whether the crate's own tests do. A mutant that only the tests catch points at a gap in the generator or in a property. [tools/mutation/README.md](../tools/mutation/README.md) has the results for `arrow-select`.

A run is a measurement, not a test. Each mutant rebuilds arrow-pbt and runs both sides, so a full run on `arrow-select` takes hours, and it does not run nightly or in CI. Run it in these cases:

- After a change to the generator or a property that is meant to reach code the properties missed, to check that it does.
- When properties first cover a crate, to find what they miss there.
- Before citing kill rates, so that the numbers describe the current properties.

### The baseline

The numbers of a run hold for one arrow-rs commit and one list of mutants from that commit, together the baseline. Keep the baseline fixed while measuring a change to the properties, so that only the properties differ between runs. Take a new baseline when the old one stops working:

- The properties fail on the baseline commit without any mutant. Every mutant would then count as killed. This happens when a bug is fixed after the baseline commit and its known-bug filter is removed.
- arrow-pbt no longer builds against the baseline commit.
- The kill rates should describe current arrow-rs code.

The stored baseline is arrow-rs `main` at 70d8190350.

### Running it

The runner, `tools/mutation/run.py`, covers `arrow-select` only. Another crate needs its own table of properties and its own test command in the script.

1. Check out the baseline commit with `git worktree add`, so that the run does not depend on a working clone.
2. Point `.cargo/config.toml` at that checkout, and run the properties that `PROPS` in `run.py` lists, the way the runner does: `cargo run -q -- run <properties> 1 1000 0`. They must pass. When a property starts to cover another file of the crate, add it to `PROPS`. The runner stops before running anything if a mutant is in a file `PROPS` does not list.
3. For a new baseline, list the mutants in the checkout with `cargo mutants --list --json -p arrow-select > mutants.json`.
4. Run `tools/mutation/run.py setup <workers> <checkout>`. It gives each worker a copy of the checkout and of this repository's working tree under `target/mutation`, so uncommitted changes to the properties are measured too. Each worker builds and tests with one job, and can use up to 6 GB of memory, so choose the workers from the cores and the memory. Rerun `setup` after changing the properties.
5. Run `tools/mutation/run.py run <workers> mutants.json`. It writes one line per mutant to `target/mutation/results.jsonl` and skips the mutants already there, so a stopped run resumes. A third argument runs a random sample of that many mutants.
6. Run `tools/mutation/summarize.py target/mutation/results.jsonl` for the kills per file and the mutants that only one side caught.
7. Read each mutant that only the tests caught. Either the generator or a property cannot reach it, which gets a fix, or no property that sees only outputs can notice it, such as a change to `size()`. The second kind is recorded in the results as such.
8. To measure a fix, write the mutants it targets to a JSON file, move `target/mutation/results.jsonl` aside, rerun `setup`, and run that file against the same baseline.
9. To publish a new baseline, replace `mutants-select.json` and `results.jsonl` in `tools/mutation/` together, and update the commit and the numbers in its README, in [coverage.md](coverage.md), and in this section.

## When arrow-rs changes

Nightly runs find bugs in code the properties reach. They cannot show code that arrow-rs adds and the properties do not reach: a new entry point, such as a new `BatchCoalescer` method, or a new path behind an entry point a property already calls, such as a new builder inside `BatchCoalescer`. List the new entry points each time the commit in `Cargo.toml` moves, since it takes seconds per crate. Measure the changed code for each arrow-rs release candidate, since a release makes new API permanent.

1. **List the new entry points.** For each crate in [coverage.md](coverage.md), diff the public API of the release the catalog names against the new commit. Run it in Cargo's checkout of the new commit, which a build of this repository creates, so that no git checkout is needed. The checkout's directory is named after the first 7 characters of the commit:

   ```sh
   cd "${CARGO_HOME:-$HOME/.cargo}"/git/checkouts/arrow-rs-*/<first 7 characters of the commit>
   CARGO_TARGET_DIR=/tmp/public-api cargo public-api -p arrow-select --all-features diff <release>
   ```

   `--all-features` lists items behind a feature, such as `ArrayStreamReader` behind `ffi`. The separate target directory keeps the build out of Cargo's cache. It needs a nightly toolchain installed, and takes seconds per crate. The diff also lists items added between the release and the catalog's commit, which the catalog may already have. Add each new item to the catalog as "Not covered", or as "Indirect" when a covered kernel calls it, and update the commit the catalog names. A "Not covered" entry point needs a property, as in [Adding a property](#adding-a-property).

2. **Measure the changed code.** List the mutants in code changed since the mutation baseline:

   ```sh
   git diff <baseline> <new> > change.diff
   cargo mutants --list --json --in-diff change.diff -p arrow-select > mutants.json
   ```

   This lists only the mutants in changed code, usually a small share of the crate's. Run them as in [Running it](#running-it), with a `git worktree` of the new commit as the checkout. Each mutant that only the tests kill is a gap in the generator or a property, or an equivalent mutant. These counts cover only the changed code, so they do not replace the baseline.

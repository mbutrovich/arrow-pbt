# arrow-pbt

[![CI](https://github.com/mbutrovich/arrow-pbt/actions/workflows/ci.yml/badge.svg)](https://github.com/mbutrovich/arrow-pbt/actions/workflows/ci.yml)

Property-based tests for [arrow-rs](https://github.com/apache/arrow-rs) and the projects built on it. One generator produces valid Arrow arrays of every type in every physical layout the Arrow format allows. An independent oracle reads any array into plain values. Short properties check the compute kernels against both. A failing case shrinks to a small input and reports the seed that reproduces it.

The properties and the generator's tests have found 48 bugs: 41 in arrow-rs `main`, one in atoi, which arrow-cast uses to parse integers, and six in DataFusion `main`. Each was reported with a reproduction or linked to an existing issue. 31 of the 48 are fixed. [docs/known-bugs.md](docs/known-bugs.md) lists them.

arrow-pbt is an independent project. It is not part of Apache Arrow, and the Apache Software Foundation does not endorse it.

## How it differs from arrow-rs's tests

arrow-rs's unit tests check inputs that a contributor chose against results written by hand. Its randomized tests, such as `fuzz_filter`, draw random values for a few types. arrow-pbt draws every type in every layout the format allows, computes each expected result with an oracle that does not use arrow-rs's kernels, and shrinks a failure to a small input with a seed that reproduces it. It adds to arrow-rs's tests and does not replace them. Of the 1,348 mutants of `arrow-select` that compiled, each a copy with one small bug planted, the properties caught 69.0%, the crate's tests caught 74.9%, and the two together caught 81.5%. [Compared with other tests](docs/design.md#compared-with-other-tests) has the details.

## Who it is for

- **arrow-rs contributors** can run the properties against a branch before merging a risky change, such as a new kernel or a rewrite of a fast path, against `main` every night, and against a release candidate. [docs/running.md](docs/running.md) says how.
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

The first build fetches and compiles arrow-rs, and DataFusion for the second command. Each property runs 256 cases with seed 1 by default, and `ARROW_PBT_SEEDS` and `ARROW_PBT_CASES` run more, such as `ARROW_PBT_SEEDS=1-10 ARROW_PBT_CASES=1000 cargo test --test properties`. [docs/running.md](docs/running.md) describes every setting, the binary for long runs, and how to test a branch or a local checkout.

## Reading a failure

A failing property prints the seed and case count that reproduce it, the failure, the minimal input that shrinking found, and the command that reruns it. This example is a known bug whose filter `ARROW_PBT_NO_KNOWN=1` turned back into a failure, so the message names its issue. A new bug's message is the property's own, such as the first row that differs from the model:

```text
arity failed with seed 1 after 2 cases: known bug, filter disabled: known: binary_mut returns Ok for an empty shared array (apache/arrow-rs#12002)
minimal input: (PrimitiveArray<Int8> [ ], PrimitiveArray<Int8> [ ])
rerun with: ARROW_PBT_SEEDS=1 ARROW_PBT_CASES=256 ARROW_PBT_NO_KNOWN=1 cargo test -p arrow-pbt arity
```

The same seed and case count always give the same case. [When a property fails](docs/workflow.md#when-a-property-fails) says how to decide whose bug it is and how to report it.

## Reporting bugs upstream

A failing property is the start of a bug report, not the report itself. I follow these rules for every bug filed from this project, and I ask anyone who uses it to do the same:

- Reduce the failure by hand to a minimal reproduction, and run it against upstream `main`.
- Search for an existing issue before filing a new one.
- Review every report before filing it, whoever or whatever drafted it.
- Take ownership of the issue once it is filed. Fix it yourself if you can. If someone else fixes it, steward the fix: answer the maintainers' questions, review the pull request, and rerun the property without its filter once the fix lands.
- Write a fix's tests the way the project writes its own. A minimal reproduction shows one failing input, but a regression test should cover the class of inputs the bug affects, in the project's own test format, such as a unit test next to the kernel in arrow-rs or a sqllogictest case in DataFusion.

[When a property fails](docs/workflow.md#when-a-property-fails) and [When a fix lands upstream](docs/workflow.md#when-a-fix-lands-upstream) describe the steps.

## More

- [docs/running.md](docs/running.md): how to choose the commits under test, the settings, the binary for long runs, run sizes, nightly runs, and how to test an arrow-rs release candidate or pull request.
- [docs/design.md](docs/design.md): the terms used here, the generator, the oracle, the properties, the rules that keep them honest, how they compare with arrow-rs's own tests, and how downstream projects use the crate.
- [docs/workflow.md](docs/workflow.md): what to do when a property fails, how to keep finding bugs while one waits for a fix, how to move the commits under test, and how to add a property, including one for API in an unmerged arrow-rs pull request, or an adapter.
- [docs/known-bugs.md](docs/known-bugs.md): every bug the properties found, and the filters of the open ones.
- [docs/coverage.md](docs/coverage.md): every arrow-rs entry point, what covers it, and what was measured.
- [CONTRIBUTING.md](CONTRIBUTING.md): the layout of the code, the checks to run before a pull request, and where to report a bug the properties find.

## AI Disclosure

I wrote this project with help from Claude. I understand and support its design, the generator, the oracle, the properties, and the adapters.

## License

Apache-2.0, as arrow-rs.

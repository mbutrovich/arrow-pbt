# Design

## Terms

| Term              | Meaning                                                                                                                                                                                                                                                             |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Property          | A Rust function that checks one claim about arrow-rs on one case. It passes, fails, or discards the case                                                                                                                                                            |
| Case              | One generated input to a property, as a tuple of arrays and parameters. proptest calls it a test case                                                                                                                                                               |
| Strategy          | A proptest value that generates values of one type and knows how to shrink them                                                                                                                                                                                     |
| Case strategy     | The strategy that generates the cases of a property. arrow-rs's are in `src/cases.rs`, named after their property, such as `cases::take`. Properties that take the same inputs share one. An adapter names its own, such as `calls`                                 |
| Generator         | The strategies in `src/generator.rs` for types, rows, layout seeds, masks and indices, and `build`, which turns them into an array                                                                                                                                  |
| `GeneratorType`   | The generator's description of a type: a `DataType`, plus which child fields may not be null, the union type ids, and which value pool to draw from                                                                                                                 |
| Rows              | The logical values of an array, as `Val`s, before a layout is chosen                                                                                                                                                                                                |
| Layout seed       | A `u64` that decides every physical choice `build` makes: slicing, offsets, data under nulls, buffers. Seed 0 is the plain layout                                                                                                                                   |
| Run seed          | The seed of a proptest run (`ARROW_PBT_SEEDS`). The same run seed and case count always give the same cases                                                                                                                                                         |
| Value regime      | How all the values of one numeric array are drawn, such as every value narrow or one outlier, because optimized kernels branch on the whole batch                                                                                                                   |
| Feature           | One thing the generator varies, such as a nonzero offset or a non-nullable child field. Layout features are about the physical layout and numeric features about the values. `arrow-pbt coverage` measures each feature on the built arrays                         |
| Oracle            | `val::vals`, which reads any array into plain values without arrow-rs's kernels                                                                                                                                                                                     |
| Model             | The output a property expects, computed from the oracle's values of the inputs                                                                                                                                                                                      |
| Spec checker      | `spec::check`, which checks an array against the Arrow columnar format, independently of `validate_full`                                                                                                                                                            |
| Law               | A property that relates two computations instead of checking against a model, such as two `take` calls against one. The literature calls it a metamorphic relation                                                                                                  |
| Round trip        | A property that writes an array and reads it back, such as through IPC or FFI                                                                                                                                                                                       |
| Discard           | A case a property cannot use, counted by reason. proptest calls it a rejection                                                                                                                                                                                      |
| Known bug         | A filed arrow-rs bug that some properties would otherwise fail on                                                                                                                                                                                                   |
| Signature         | What identifies a known bug's failure: a substring of the failure message, plus the condition that makes it that bug                                                                                                                                                |
| Known-bug filter  | Code that matches a signature and discards the case through `known::known`, naming the issue                                                                                                                                                                        |
| Shrinking         | proptest reducing a failing case to a smaller one that still fails                                                                                                                                                                                                  |
| Entry             | A property in `src/registry.rs`, with the name of its case strategy and closures to run it and to sample its cases                                                                                                                                                  |
| Adapter           | A crate in this repository that connects this crate to a downstream project: the types the project accepts, conversions to its own types, case strategies for its entry points, and harness functions for its laws. See [Downstream adapters](#downstream-adapters) |
| Harness function  | A function in an adapter that checks one law for whatever entry point it is given, such as `check_broadcast`, so that a test of one entry point is one call                                                                                                         |
| Mutant            | A copy of arrow-rs with one small change, such as `>` replaced by `>=`, made by cargo-mutants to check whether the properties detect it                                                                                                                             |
| Killed            | A mutant on which some property or test fails. cargo-mutants calls it caught                                                                                                                                                                                        |
| Survived          | A mutant on which every property and test passes. cargo-mutants calls it missed                                                                                                                                                                                     |
| Equivalent mutant | A mutant that behaves like the original on every valid input, so no property can kill it, such as a change that only picks a different path to the same output, or one in code the public API cannot reach                                                          |

## The pieces

A test case is a type, a list of logical rows, and a layout seed. The generator (`src/generator.rs`) draws all three. `build` turns them into an array, and the seed decides every physical choice, so shrinking the seed moves toward the plain layout. The oracle (`src/val.rs`) reads any array back into logical values without using arrow-rs's comparison or cast code. The spec checker (`src/spec.rs`) checks an array against the Arrow columnar format: buffer sizes, offsets, union type ids, run ends, and the rest.

A property is a Rust function from a case to `Result<(), TestCaseError>`. It computes what the kernel should return from the oracle's values, calls the kernel, and checks that the output is valid and holds those values. `src/cases.rs` gives each kernel its input strategy, and `src/registry.rs` pairs every property with its strategy.

The properties run on proptest because arrow-rs already depends on it. Which property testing library runs them matters far less than the generator.

## Rules

These rules keep the results meaningful.

1. The generator never drops valid inputs. Every array it produces is valid under the Arrow format, and every valid layout it knows is reachable.
2. A case strategy restricts inputs only to the domain a kernel documents. When a kernel documents an error, the property asserts the error instead of avoiding the input.
3. A known bug is filtered by its failure signature, never by input type. Each filter names its issue. See [workflow.md](workflow.md).
4. Every discard is counted by reason. `ARROW_PBT_REASONS=1` prints them.
5. Generators are measured, not assumed. `arrow-pbt coverage` and `arrow-pbt inventory` report what the inputs contain, measured on the built arrays.
6. An oracle is checked by breaking it. Changing the oracle on purpose (truncating instead of rounding, dropping a leading zero) must make its property fail. If it does not, the property is not checking what it claims.
7. A generator dimension comes from the Arrow format or a documented API, never from a constant inside an implementation, such as a chunk or block size. Mutation testing and the inventory show which inputs are missing, and the fix is the general class of input the format or the API defines. A mutant that only an input tuned to one implementation could kill is recorded as such.

## What the generator varies

**Types.** Null, Boolean, every integer and float width except Float16, decimals of four widths with several precisions and scales (including a negative scale), dates, times, timestamps with and without time zones, durations, intervals, Utf8, LargeUtf8, Utf8View, Binary, LargeBinary, BinaryView, FixedSizeBinary (including width 0), List, LargeList, ListView, LargeListView, FixedSizeList (including size 0), Struct (including no fields), Map (sorted and unsorted), Dictionary with every key type, run-end encoded arrays with every run-end type, and sparse and dense unions (with default and non-default type ids). Types nest to depth 2.

**Nullability.** Child fields are not nullable one time in four, and the generator then never puts a null under them. A union or run-end encoded array cannot hold a null when none of its children can.

**Lengths.** Empty and tiny arrays, lengths at byte, word and vector-lane boundaries (7 to 9, 15 to 17, 31 to 33, 63 to 65, 127 to 129, 255 to 257), a uniform range, and occasional batch sizes around 1024 and 2048.

**Nulls.** A null rate chosen per array (none, few, some, half, or all), random data under null slots, set padding bits in validity bitmaps, and a validity bitmap with no nulls.

**Slicing.** An array may be built longer and sliced in four ways: `Array::slice`, `ArrayData::slice`, `ArrayData::slice` followed by an FFI round trip, and an offset on the parent with whole children, as pyarrow exports.

**Physical layouts.** Offsets that do not start at 0, string and list null slots over non-empty ranges, view arrays with several data buffers, shared bytes and unused buffers, list views with out-of-order offsets, shared ranges and gaps, list and list view children that start at a nonzero offset of their own, which the C data interface allows for every array, dictionaries with unused, duplicate and null values, non-maximal runs, dense unions with unused child slots and with children longer than their offsets use (including children no row refers to), dense union rows that share a child slot, which the format allows because each child's offsets need only be in order and Arrow C++ accepts equal ones, dense union offsets that are sequential within each child, which kernels slice instead of taking, and nearly sequential offsets, with one shared slot and one gap, which pass a check of the first and last offsets without being sequential, and struct children that are valid under a null parent row. A primitive values buffer sometimes sits at exactly the alignment arrow-rs requires, `align_of` of the native type, and not at the larger alignment of arrow-rs's own allocations. That is below the width for `Interval(DayTime)`, `Interval(MonthDayNano)` and `Decimal256`. These placements come from a random stream of their own, so they do not change the other layout choices a layout seed makes.

**Values.** Each integer, decimal or float array draws its values in one regime for the whole batch, because optimized numeric kernels branch on properties of the whole batch:

| Share | Regime                                                                                                           |
| ----- | ---------------------------------------------------------------------------------------------------------------- |
| 40%   | Each value on its own, from the type's edges and random bit patterns                                             |
| 20%   | Every value fits in 8, 16, 32, 64 or 128 bits                                                                    |
| 10%   | The same, except one row is just past that width                                                                 |
| 20%   | Values within one of a power of two or of ten, and the type's limits                                             |
| 10%   | Small values, except one row is the type's minimum, maximum, zero or -1 (for floats: a NaN, an infinity or -0.0) |

The odd row lands first, last, at row 63 or 64, or anywhere. Two in seven unions take every row from one child, because kernels branch on whether every row has one type id. One in seven takes all rows but one from one child, with the odd row placed as above. Strings mix multi-byte characters, LIKE syntax, and text that looks like numbers. A view array mixes inline values (12 bytes or less) and long ones, or holds only inline values and has no data buffer, or only long ones, which puts all its bytes in data buffers. Temporal values include calendar edges and the limits where unit conversions overflow.

**Decimals past their precision.** By default, a decimal value of up to 128 bits that is drawn on its own, as in the first regime above and inside nested types, lies past its declared precision one time in ten, and random garbage that lands in a valid slot often does. The Arrow format describes a decimal's precision as the "Total number of decimal digits" ([Schema.fbs](https://github.com/apache/arrow/blob/e2524aa088cd6a391b40635245d757542e2b2810/format/Schema.fbs#L229-L231)). Arrow C++'s full validation rejects a value past it in a valid slot ([validate.cc](https://github.com/apache/arrow/blob/fcf8f3014227cafe24deeb044c2d51f5fc9902e5/cpp/src/arrow/array/validate.cc#L957-L972)), and so does DataFusion ([apache/datafusion#26774](https://github.com/apache/datafusion/issues/26774)). arrow-rs accepts it and checks precision only on request, so the arrow-rs properties draw such values, and they found [apache/arrow-rs#10947](https://github.com/apache/arrow-rs/issues/10947). `ValueDomain::Logical` keeps every valid decimal slot within its precision, for adapters whose project rejects these values. Garbage under a primitive's own null slot stays unconstrained in both domains.

**Masks and indices.** Filter masks include all and none selected, exactly one row kept or dropped, runs, exact 64-bit words, fixed selectivities, and sparse masks that keep between 1 row and 1/16 of the rows. Index arrays include runs, repeats, the identity, and nulls that hide out-of-bounds values. The `take` property also sends one valid out-of-bounds index in five cases and expects the documented error.

## Kinds of properties

The names in parentheses are the standard ones, from John Hughes' ["How to Specify It!"](https://doi.org/10.1007/978-3-030-47147-7_4) and Scott Wlaschin's ["Choosing properties for property-based testing"](https://fsharpforfunandprofit.com/posts/property-based-testing-2/).

- **Agreement with a model** (model-based, or "the test oracle"). The kernel's output equals values computed from the oracle: `take`, `filter`, `interleave`, `concat`, `zip`, `nullif`, aggregates, arithmetic, the arity helpers and the bitwise kernels against exact integers, casts, fixed-point multiplication, decimal formatting, string kernels, and temporal arithmetic.
- **Laws** (metamorphic relations, or "different paths, same destination"). Two `take` calls equal one with composed indices, a filter equals a take of the kept rows, concatenated slices equal the whole, the row format orders rows as `lexsort` does, a DataFusion function gives the same result with scalar arguments as with those scalars broadcast, and each row of a DataFusion call equals the call on that row's values as scalars.
- **Round trips** ("there and back again"). IPC, FFI and `ArrayData` preserve values.
- **Valid output** (invariants, or "some things never change"). Every output passes `validate_full` and the spec checker, whatever its values.

## Downstream adapters

A project built on arrow-rs, such as DataFusion or iceberg-rust, is tested through an adapter: a crate in this repository that connects this crate to the project. An adapter has four parts:

1. **The types the project accepts.** For a DataFusion function, these are its signatures, the lists of argument types it accepts after coercion. For iceberg-rust, they are the arrow types that map to Iceberg types, which a `BoxedStrategy<GeneratorType>` can draw from.
2. **Conversions.** Plain functions from this crate's output to the project's own types, such as an array row to a DataFusion `ScalarValue`, or a `Val` to an Iceberg `Datum`. They generate nothing and do not shrink.
3. **Case strategies.** One for each entry point under test, built from the type strategy, the conversions, and this crate's rows, seeds and arrays. Shrinking works through them.
4. **Harness functions.** One for each law, so that a test of one function is one call. portable-simd's `test_unary_elementwise`, which checks a vector function against a scalar one lane by lane, is the model.

The project's properties use the case strategies, the oracle, and the known-bug discard. This crate stays arrow-only: it has no dependency on any downstream project and knows nothing of `ScalarValue`, `Datum` or Iceberg field ids.

### Layout

The repository is a Cargo workspace. `arrow-pbt` is the root package, and each adapter is a member crate, which tests its project from outside with no change to the project.

| Crate                             | Depends on                                                                                                                             | Contains                                                                                |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `arrow-pbt`                       | arrow-rs                                                                                                                               | The generator, the oracle, the spec checker and `known`, and the properties of arrow-rs |
| `arrow-pbt-datafusion`            | `arrow-pbt`, `datafusion-common` and `datafusion-expr`, and `datafusion-functions` for its tests, all on a commit of DataFusion `main` | The DataFusion adapter as a library, and DataFusion's properties as its tests           |
| `arrow-pbt-iceberg` (not created) | `arrow-pbt`, `iceberg`                                                                                                                 | The same for iceberg-rust                                                               |

The workspace has one lockfile, and the `[patch.crates-io]` section of `Cargo.toml` puts all of it on one commit of arrow-rs `main`, so an adapter tests its project on arrow-rs `main`. That is the combination the project ships next, and the one its own pre-release testing checks, as [apache/datafusion#26384](https://github.com/apache/datafusion/issues/26384) does for DataFusion. A project's releases are on an older arrow, so `[patch.crates-io]` also puts each project on a commit of its own `main` until a release catches up, as [Choosing the commits under test](../README.md#choosing-the-commits-under-test) describes. When arrow-rs `main` breaks the project's build, the adapter pins the project's arrow upgrade branch instead. `arrow-pbt-iceberg` waits until iceberg-rust's `main` builds on the arrow version that arrow-rs `main` is on. An adapter that needs another arrow version cannot share the workspace. It would also need the arrow-rs properties moved out of the library first, since the `coalesce` property in `src/batches.rs` uses `BatchCoalescer` and `FilterBuilder` methods first released in arrow 60.1.0.

An adapter uses only its project's public API. That keeps it independent of the project's maintainers, and it lets the adapter move into the project's own test crate unchanged if that community wants it later. Other projects can use an adapter too: Comet's Spark functions are DataFusion `ScalarUDF`s, so `arrow-pbt-datafusion` can test them.

### DataFusion

Scalar functions take `ScalarFunctionArgs`: one `ColumnarValue` per argument, which is either an array or a `ScalarValue`, plus each argument's field, the number of rows, the return field, and the session config. `calls` generates them from a function's signatures, the lists of argument types it accepts after coercion, since calling `invoke_with_args` directly skips coercion. Each argument's field is nullable three times in four, and each argument is an array of the call's rows or a scalar, because functions have separate paths for each mix. A scalar comes from a one-row array through `ScalarValue::try_from_array`, so it covers every type and keeps the array's layout.

`check_broadcast` checks the broadcast law, which needs no model. Calling the function as drawn and calling it with every scalar argument broadcast to an array must both fail, or both return valid output with the same values, of the type and nullability the return field promises. The return field comes from `return_field_from_args` for each call. A result of one row from only scalar arguments counts as a scalar, as `ScalarFunctionExpr::evaluate` reads it, and any other row count that differs from the call's is a failure. Two kinds of case are outside the law and discarded: the two calls promise different return fields, as `round` does when its `decimal_places` is a scalar, or there are no rows and only the call as drawn fails, since broadcasting to no rows drops the scalars' values. The return fields are compared before the results, since a call can fail only because of the type it promised. The session config is the default.

`check_row_by_row` checks the row-by-row law, which also needs no model. Each row of the call as drawn must equal the call with every argument as that row's scalar, and the call as drawn may fail only when some row fails on its own. A one-row call cannot take a fast path that branches on the whole batch, so the law checks those fast paths and the handling of layouts, such as a validity bitmap with an offset, which the broadcast law cannot see: broadcasting passes an array argument through unchanged, so both of its calls read the array the same way. It checks every row, which cost 10% more per case than the broadcast law for `abs` and 5% more for `upper`, at 1,000 cases each. A call with no array argument or no rows passes, a row that promises a different return field discards the case, and a volatile function is a failure. Neither law sees a function that is wrong the same way on every path. Many DataFusion functions build their scalar path by running their array code on one row, through `make_scalar_function`, and for those the row-by-row law compares the array code at the call's rows with the same code at one row.

A function's test runs both laws through `run::check` over `calls`, after checking that each signature is one that DataFusion's coercion leaves unchanged, with `fields_with_udf`. `tests/functions.rs` also runs both laws on functions broken on purpose: one whose scalar path differs from its array path, one that returns too many rows, one that panics, and one that ignores the offset of its input's validity bitmap. It checks that each failure says why, and that the broadcast law misses the last one.

### iceberg-rust

A partition transform has an array path, `TransformFunction::transform`, and a value path, `transform_literal`, on Iceberg's `Datum`. The Iceberg spec defines each transform exactly, so the model for a transform property comes from the spec. The adapter builds each `Datum` from the oracle's `Val` instead of through iceberg-rust's arrow conversion, so that a bug in the conversion cannot hide a bug in the transform. The conversion gets its own property.

Iceberg types map to a subset of arrow types, and one Iceberg type often has several arrow encodings, such as `Utf8`, `LargeUtf8`, `Utf8View` and dictionaries for `string`. iceberg-rust reads each field's id from its arrow metadata, nested fields included. The entry points that take record batches, such as the partition splitter and the data file writer, give properties with no model. Every output batch of the splitter has one partition key, and together the outputs hold exactly the input rows. The writer's null counts, NaN counts and bounds equal the oracle's.

### What the library gives adapters

- `GeneratorType::try_from` for a `DataType` or a `Field`. Arrays built from the result have that type exactly, with every child field's name, nullability and metadata, and hold no null that a field forbids. Dictionary and run-end encoded values can be of any type. A type the generator cannot build, such as `Float16`, is an error.
- `array(ty, n)` for an array of a given type and length, such as a function's argument or a batch's column, and `array_in(ty, n, domain)` for one whose values come from a `ValueDomain`. [Decimals past their precision](#what-the-generator-varies) says which domain an adapter needs.
- `run::check`, the runner every property goes through, with its settings in `ARROW_PBT_*` variables, fixed seeds, discards by reason, and a rerun command on failure.
- `GeneratedArray`, `Mask` and `Indices`, the results of `array_of`, `mask` and `indices`, each with a `build` method.
- `run::caught`, which turns a panic into a failure inside a property, so that a known-bug filter can match its message.
- `val::check_rows`, the output check of every property. It accepts the generator's rows as they are.

Two things wait until a property needs them. The [Parquet properties](coverage.md#parquet) and the Iceberg adapter need both:

1. **Record batches.** A strategy for a `RecordBatch` of a schema, a few lines over `array`. A batch with no columns needs its row count set.
2. **Typed access to `Val`.** A fixed-width value is a `Val::Raw` of little-endian bytes. Turning it into a downstream value, such as an Iceberg `Datum`, needs a decoder for each `DataType`, which belongs in `val` rather than in each adapter.

## Measuring

`arrow-pbt coverage` reports, for generated arrays, numeric arrays, masks and indices, the share of cases that show each feature. A feature at 0% is one the generator lost. `arrow-pbt inventory` reports which type families and layout features each property's inputs reach, and which internal paths of `zip`, `lexsort` and `BatchCoalescer` they take.

Mutation testing measures whether the properties would catch a bug in the kernels. Every cargo-mutants mutant of `arrow-select` has run against both these properties and the crate's own tests. [tools/mutation/README.md](../tools/mutation/README.md) has the results and the runner, and [workflow.md](workflow.md#measuring-the-properties) says when to run it.

## Not covered yet

See [coverage.md](coverage.md) for every entry point and what covers it.

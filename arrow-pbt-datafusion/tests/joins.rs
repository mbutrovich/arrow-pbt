//! Laws for DataFusion's joins. The agreement law: a hash join or a
//! sort-merge join gives the same rows as a nested loop join of the same
//! inputs. The config law: a hash join gives the same rows under two session
//! configs. The order law: a sort-merge join returns its rows in the output
//! ordering its plan states. The split law: a join gives the same rows when
//! its inputs' rows are cut into other batches. The fetch laws: a hash join
//! with a fetch returns that many of its rows, and stops reading its probe
//! side once it has them.

use std::fmt::Debug;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type,
    UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrayRef};
use arrow_pbt::generator::GeneratedBatch;
use arrow_pbt::known::{known, ree_values_nullable};
use arrow_pbt::run::{caught, check, run};
use arrow_pbt::types::{any_type, is_view_dict};
use arrow_pbt_datafusion::{
    JoinCase, JoinPlan, check_join_agreement, check_join_config, check_join_fetch,
    check_join_order, check_join_split, hash_join, joins, nested_loop_join, sort_merge_join,
    sort_merge_joins, splits,
};
use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef, SortOptions, TimeUnit};
use datafusion_common::{JoinType, NullEquality};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::{LexOrdering, PhysicalExpr, PhysicalSortExpr};
use datafusion_physical_plan::joins::{HashJoinExec, PartitionMode};
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::streaming::{PartitionStream, StreamingTableExec};
use datafusion_physical_plan::{ExecutionPlan, Partitioning};
use proptest::prelude::{Just, any};
use proptest::sample::Index;
use proptest::strategy::{BoxedStrategy, Strategy};
use proptest::test_runner::TestCaseError;

/// Key types for which DataFusion's hash join and nested loop join both
/// define equality: integers, floats, decimals, temporal types, strings and
/// binaries in each layout, dictionaries, and nested types
fn key_types() -> Vec<DataType> {
    vec![
        DataType::Boolean,
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
        DataType::UInt8,
        DataType::UInt16,
        DataType::UInt32,
        DataType::UInt64,
        DataType::Float32,
        DataType::Float64,
        DataType::Decimal128(10, 2),
        DataType::Decimal256(45, 5),
        DataType::Date32,
        DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        DataType::Utf8,
        DataType::LargeUtf8,
        DataType::Utf8View,
        DataType::Binary,
        DataType::BinaryView,
        DataType::FixedSizeBinary(3),
        DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
        DataType::Struct(Fields::from(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Utf8, true),
        ])),
        DataType::List(Arc::new(Field::new_list_field(DataType::Int32, true))),
    ]
}

/// Panics as failures, and the failures of known bugs as discards
fn filtered(
    case: &JoinCase,
    law: impl FnOnce() -> Result<(), TestCaseError>,
) -> Result<(), TestCaseError> {
    coalescer_panic(view_dict_overflow(
        case,
        null_padding(ree_values_nullable(caught(law))),
    ))
}

/// Known: the sort-merge join calls `expect` on its coalescers' `push_batch`
/// in five places, so an error there panics rather than failing the query.
/// Arrow returns one when concatenating the joined batches overflows a
/// dictionary key or run-end type, as `ListView` children of run-end encoded
/// values do, since `concat` keeps each view's whole child. Found by the
/// agreement and order laws on DataFusion `main` at dbb98dedc3
/// (apache/datafusion#27728)
fn coalescer_panic(r: Result<(), TestCaseError>) -> Result<(), TestCaseError> {
    match &r {
        Err(TestCaseError::Fail(m))
            if m.message().contains("Failed to push output batch")
                || m.message()
                    .contains("Failed to push batch to BatchCoalescer") =>
        {
            Err(known(
                "known: the sort-merge join panics when its output coalescer returns an error (apache/datafusion#27728)",
            ))
        }
        _ => r,
    }
}

/// Known: `concat` and `interleave` do not merge dictionaries of views, so
/// combining batches overflows the key type even when the values they use
/// fit (apache/arrow-rs#10925). A hash join concatenates its build side, and
/// the nested loop join combines batches differently, so one join fails.
/// Found by the agreement law on DataFusion `main` at e2dc8b048b, on a
/// `FixedSizeList` of `Dictionary(Int8, Utf8View)`
fn view_dict_overflow(case: &JoinCase, r: Result<(), TestCaseError>) -> Result<(), TestCaseError> {
    let views = [&case.left_schema, &case.right_schema]
        .iter()
        .flat_map(|s| s.fields().iter())
        .any(|f| any_type(f.data_type(), &is_view_dict));
    match &r {
        Err(TestCaseError::Fail(m))
            if views
                && m.message()
                    .contains("Dictionary key bigger than the key type") =>
        {
            Err(known(
                "known: dictionaries of views are not merged, so keys overflow (apache/arrow-rs#10925)",
            ))
        }
        _ => r,
    }
}

/// Known: an outer join pads the unmatched side's columns with nulls. For a
/// union with no field that can hold a null, or a run-end encoded array with
/// a non-nullable values field, `take` with a null index returns an error, as
/// apache/arrow-rs#10992 settled, but `new_null_array` writes the nulls into
/// a field that cannot hold one (apache/arrow-rs#11835). Join paths use both,
/// so one join fails and the other succeeds. Found by the agreement law on
/// DataFusion `main` at e2dc8b048b
fn null_padding(r: Result<(), TestCaseError>) -> Result<(), TestCaseError> {
    let refused = |m: &str| {
        m.contains("Cannot take null indices from a union with no field that can represent nulls")
            || m.contains(
                "Cannot take null indices from a RunEndEncoded array with a non-nullable values field",
            )
    };
    match &r {
        Err(TestCaseError::Fail(m))
            if m.message().contains(" succeeds: ") && refused(m.message()) =>
        {
            Err(known(
                "known: new_null_array puts nulls in union and run-end encoded fields that cannot hold one (apache/arrow-rs#11835)",
            ))
        }
        _ => r,
    }
}

/// Known: DataFusion's sort puts `-0.0` before `0.0`, but the sort-merge
/// join compares them as equal. When another key follows a float key, the
/// sorted inputs are then out of order for the join, which misses matches:
/// `(-0.0, 5), (0.0, 1)` joined with `(0.0, 1)` on both keys gives no rows
/// (apache/datafusion#26115 reports it for nested keys, and
/// apache/datafusion#26138 for multi-column ones). Found by the agreement law
/// on DataFusion `main` at dbb98dedc3, with arrow-rs `main` at 6a3f719f1d
fn signed_zero_order(case: &JoinCase, r: Result<(), TestCaseError>) -> Result<(), TestCaseError> {
    match &r {
        Err(TestCaseError::Fail(m))
            if m.message().contains("the rows differ") && zeros_before_a_key(case) =>
        {
            Err(known(
                "known: sorting puts -0.0 before 0.0, which a sort-merge join compares equal (apache/datafusion#26115)",
            ))
        }
        _ => r,
    }
}

/// Whether a float key other than the last holds both `-0.0` and `0.0` on
/// one side
fn zeros_before_a_key(case: &JoinCase) -> bool {
    let zeros = |values: &mut dyn Iterator<Item = f64>, signs: &mut [bool; 2]| {
        for v in values.filter(|v| *v == 0.0) {
            signs[usize::from(v.is_sign_negative())] = true;
        }
    };
    (0..case.keys.saturating_sub(1)).any(|key| {
        [&case.left, &case.right].iter().any(|side| {
            let mut signs = [false; 2];
            for batch in *side {
                let a = batch.columns[key].build();
                match a.data_type() {
                    DataType::Float32 => zeros(
                        &mut a
                            .as_primitive::<Float32Type>()
                            .iter()
                            .flatten()
                            .map(f64::from),
                        &mut signs,
                    ),
                    DataType::Float64 => zeros(
                        &mut a.as_primitive::<Float64Type>().iter().flatten(),
                        &mut signs,
                    ),
                    _ => {}
                }
            }
            signs == [true, true]
        })
    })
}

fn agree(case: &JoinCase, a: &JoinPlan, b: &JoinPlan) -> Result<(), TestCaseError> {
    filtered(case, || check_join_agreement(case, a, b))
}

#[test]
fn hash_join_agrees_with_nested_loop_join() {
    check(
        "hash join against nested loop join",
        || joins(&key_types()).unwrap(),
        |case: JoinCase| agree(&case, &hash_join, &nested_loop_join),
    );
}

/// With a key range threshold of 0 and a density no build side reaches,
/// the hash join never uses its perfect hash map (`ArrayMap`), so this
/// compares it with the general hash map wherever the default config uses it
#[test]
fn perfect_hash_join_agrees_with_hash_map() {
    let options = [
        (
            "datafusion.execution.perfect_hash_join_small_build_threshold",
            "0",
        ),
        (
            "datafusion.execution.perfect_hash_join_min_key_density",
            "1e300",
        ),
    ];
    check(
        "perfect hash join against hash map",
        || joins(&key_types()).unwrap(),
        |case: JoinCase| filtered(&case, || check_join_config(&case, &hash_join, &options)),
    );
}

/// The sort-merge join reads inputs sorted on their keys with drawn sort
/// options, and the nested loop join reads the same sorted inputs
#[test]
fn sort_merge_join_agrees_with_nested_loop_join() {
    check(
        "sort-merge join against nested loop join",
        || sort_merge_joins(&key_types()).unwrap(),
        |case: JoinCase| {
            signed_zero_order(&case, agree(&case, &sort_merge_join, &nested_loop_join))
        },
    );
}

/// The sort-merge join's inputs declare the order they are sorted in, and
/// the join's plan derives its output ordering from theirs
#[test]
fn sort_merge_join_keeps_its_output_order() {
    check(
        "sort-merge join output order",
        || sort_merge_joins(&key_types()).unwrap(),
        |case: JoinCase| filtered(&case, || check_join_order(&case, &sort_merge_join)),
    );
}

/// The split law on the hash join, in one partition or several
#[test]
fn hash_join_keeps_its_rows_when_resplit() {
    check(
        "hash join split",
        || {
            joins(&key_types())
                .unwrap()
                .prop_flat_map(|c| (Just(c.clone()), splits(&c)))
        },
        |(case, split)| filtered(&case, || check_join_split(&case, &hash_join, &split)),
    );
}

/// The split law on the sort-merge join, whose cuts fall in the sorted rows
#[test]
fn sort_merge_join_keeps_its_rows_when_resplit() {
    check(
        "sort-merge join split",
        || {
            sort_merge_joins(&key_types())
                .unwrap()
                .prop_flat_map(|c| (Just(c.clone()), splits(&c)))
        },
        |(case, split)| {
            signed_zero_order(
                &case,
                filtered(&case, || check_join_split(&case, &sort_merge_join, &split)),
            )
        },
    );
}

/// Known: when the hash join's build-side hash map is empty, because the
/// build side has no rows or every build row has a null key under
/// `NullEqualsNothing`, its stream takes a path that ignores the output
/// coalescer's `LimitReached`. A right anti or right mark join, which
/// returns every probe row there, then reads its whole probe side after it
/// has a fetch's rows. Inner, left, right, and full joins take another path
/// by default.
/// Found by the fetch law on DataFusion `main` at dbb98dedc3, with arrow-rs
/// `main` at 6a3f719f1d (apache/datafusion#27726)
fn empty_build_reads_on(
    case: &JoinCase,
    r: Result<(), TestCaseError>,
) -> Result<(), TestCaseError> {
    match &r {
        Err(TestCaseError::Fail(m))
            if m.message().contains("the join reads on")
                && matches!(case.join_type, JoinType::RightAnti | JoinType::RightMark)
                && !has_matchable_build_rows(case) =>
        {
            Err(known(
                "known: the hash join reads its probe side past a fetch when its build-side hash map is empty (apache/datafusion#27726)",
            ))
        }
        _ => r,
    }
}

/// Whether the hash join's build-side hash map has an entry: under
/// `NullEqualsNothing` it leaves out a row with a null key
fn has_matchable_build_rows(case: &JoinCase) -> bool {
    case.left.iter().any(|batch| {
        let keys: Vec<ArrayRef> = batch.columns[..case.keys]
            .iter()
            .map(|c| c.build())
            .collect();
        let nulls: Vec<_> = keys.iter().map(|k| k.logical_nulls()).collect();
        (0..batch.num_rows).any(|i| {
            case.null_equality == NullEquality::NullEqualsNull
                || nulls
                    .iter()
                    .all(|n| n.as_ref().is_none_or(|n| n.is_valid(i)))
        })
    })
}

/// The fetch laws on the hash join
#[test]
fn hash_join_fetch_returns_its_rows() {
    check(
        "hash join fetch",
        || (joins(&key_types()).unwrap(), 0.0..=1.0f64, any::<usize>()),
        |(case, fraction, probe_batches)| {
            empty_build_reads_on(
                &case,
                filtered(&case, || {
                    check_join_fetch(&case, &hash_join, fraction, probe_batches)
                }),
            )
        },
    );
}

/// The integer values of `a`, an array of an integer type, with nulls
/// skipped
fn integers(a: &dyn Array) -> Vec<i128> {
    macro_rules! of {
        ($t:ty) => {
            a.as_primitive::<$t>()
                .iter()
                .flatten()
                .map(i128::from)
                .collect()
        };
    }
    match a.data_type() {
        DataType::Int8 => of!(Int8Type),
        DataType::Int16 => of!(Int16Type),
        DataType::Int32 => of!(Int32Type),
        DataType::Int64 => of!(Int64Type),
        DataType::UInt8 => of!(UInt8Type),
        DataType::UInt16 => of!(UInt16Type),
        DataType::UInt32 => of!(UInt32Type),
        DataType::UInt64 => of!(UInt64Type),
        _ => vec![],
    }
}

/// The perfect hash join (`ArrayMap`) serves a single integer key when the
/// build side's key range is below `perfect_hash_join_small_build_threshold`
/// or its density, rows over range plus one, is above
/// `perfect_hash_join_min_key_density`, and builds none under
/// `NullEqualsNull` when a build key is null. Each threshold is drawn at the
/// case's own value, one past it, or far from it, so that the join decides
/// at each boundary, and the result must match the default config
#[test]
fn perfect_hash_join_thresholds_keep_the_rows() {
    let integer_keys = [
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
        DataType::UInt8,
        DataType::UInt16,
        DataType::UInt32,
        DataType::UInt64,
    ];
    check(
        "perfect hash join thresholds",
        || {
            (
                joins(&integer_keys).unwrap(),
                any::<Index>(),
                any::<Index>(),
            )
        },
        |(case, range_pick, density_pick)| {
            // The map serves a single key, so a second key joins as an
            // ordinary column
            let case = JoinCase { keys: 1, ..case };
            let built: Vec<ArrayRef> = case.left.iter().map(|b| b.columns[0].build()).collect();
            let keys: Vec<i128> = built.iter().flat_map(|a| integers(a.as_ref())).collect();
            let (Some(min), Some(max)) = (keys.iter().min(), keys.iter().max()) else {
                return Err(TestCaseError::reject("no build keys"));
            };
            let rows: usize = case.left.iter().map(|b| b.num_rows).sum();
            let range = u64::try_from(max - min).unwrap_or(u64::MAX);
            // `array_map_key_range` builds no map over the whole build side
            // for these
            let no_map = range == u64::MAX
                || (case.null_equality == NullEquality::NullEqualsNull
                    && built.iter().any(|a| a.null_count() > 0));
            // As `array_map_key_range` in DataFusion's hash join computes it
            let dense = rows as f64 / range.saturating_add(1) as f64;
            let threshold = *range_pick.get(&[range, range.saturating_add(1), 0, u64::MAX]);
            let density = *density_pick.get(&[dense, dense.next_down(), 0.0, 1e300]);
            // The map holds a `u32` for each key in its range. A partitioned
            // join decides for each partition, whose range is at most the
            // whole build side's, so the largest range any partition could
            // use a map for bounds what this case allocates
            let largest =
                if case.partition_mode == PartitionMode::Partitioned && case.partitions > 1 {
                    let by_threshold = range.min(threshold.saturating_sub(1));
                    let by_density = if density > 0.0 {
                        range.min((rows as f64 / density) as u64)
                    } else {
                        range
                    };
                    by_threshold.max(by_density)
                } else if !no_map && (range < threshold || dense > density) {
                    range
                } else {
                    0
                };
            // Known: the map's `vec![0; range + 1]` panics with a capacity
            // overflow past `isize::MAX` bytes rather than returning an
            // error. Found by this property on DataFusion `main` at
            // dbb98dedc3 (apache/datafusion#27727). Checked before the call, since a
            // range just below that aborts on a failed allocation
            if largest >= isize::MAX as u64 / 4 {
                return Err(known(
                    "known: the perfect hash join allocates its map without checking the range fits (apache/datafusion#27727)",
                ));
            }
            if largest >= 1 << 28 {
                return Err(TestCaseError::reject(
                    "a perfect hash map of more than 1 GiB",
                ));
            }
            let (threshold, density) = (threshold.to_string(), density.to_string());
            let options = [
                (
                    "datafusion.execution.perfect_hash_join_small_build_threshold",
                    threshold.as_str(),
                ),
                (
                    "datafusion.execution.perfect_hash_join_min_key_density",
                    density.as_str(),
                ),
            ];
            filtered(&case, || check_join_config(&case, &hash_join, &options))
        },
    );
}

/// Inner, left, right, and full hash joins take the probe path from
/// apache/datafusion#27399 by default. `enable_migration_hash_join` set to
/// false falls back to the previous path until the migration in
/// apache/datafusion#26361 is complete. Both must give the same rows
#[test]
fn migration_hash_join_agrees_with_legacy() {
    let options = [("datafusion.execution.enable_migration_hash_join", "false")];
    check(
        "migration hash join against legacy",
        || joins(&key_types()).unwrap(),
        |case: JoinCase| filtered(&case, || check_join_config(&case, &hash_join, &options)),
    );
}

/// The message of the first failure of `law` on 256 cases of `cases` at
/// seed 1, or `None`. It does not shrink, since a check that a law finds a
/// mistake needs only the failure, and shrinking a join case can take
/// minutes
fn finds<T: Debug + Clone>(
    cases: &impl Strategy<Value = T>,
    law: impl Fn(T) -> Result<(), TestCaseError>,
) -> Option<String> {
    let outcome = run(cases, &law, 1, 256, 0);
    outcome.failed.then_some(outcome.message)
}

/// A way to build a join, as `hash_join` and `sort_merge_join` are
type Build = fn(
    &JoinCase,
    Arc<dyn ExecutionPlan>,
    Arc<dyn ExecutionPlan>,
) -> datafusion_common::Result<Arc<dyn ExecutionPlan>>;

/// The join `build` makes of the case's inputs, with a mistake in how it is
/// built. The inputs come from the case without the mistake
fn broken(build: Build, mistake: fn(&JoinCase) -> JoinCase) -> Box<JoinPlan> {
    Box::new(move |case, left, right| build(&mistake(case), left, right))
}

/// The agreement law finds joins broken on purpose, and the failure says
/// why
#[test]
fn finds_broken_joins() {
    let flip_nulls = |c: &JoinCase| JoinCase {
        null_equality: match c.null_equality {
            NullEquality::NullEqualsNothing => NullEquality::NullEqualsNull,
            NullEquality::NullEqualsNull => NullEquality::NullEqualsNothing,
        },
        ..c.clone()
    };
    let always_inner = |c: &JoinCase| JoinCase {
        join_type: JoinType::Inner,
        ..c.clone()
    };
    let no_filter = |c: &JoinCase| JoinCase {
        filter: false,
        ..c.clone()
    };
    // The join reads its inputs in the reverse of the order they are sorted in
    let flip_order = |c: &JoinCase| JoinCase {
        sort_options: c.sort_options.as_ref().map(|options| {
            options
                .iter()
                .map(|o| SortOptions::new(!o.descending, o.nulls_first))
                .collect()
        }),
        ..c.clone()
    };
    type Mistake = fn(&JoinCase) -> JoinCase;
    type Cases = fn() -> BoxedStrategy<JoinCase>;
    let hash: (&str, Build, Cases) = ("hash join", hash_join, || joins(&key_types()).unwrap());
    let merge: (&str, Build, Cases) = ("sort-merge join", sort_merge_join, || {
        sort_merge_joins(&key_types()).unwrap()
    });
    for ((join, build, cases), (name, mistake)) in [
        (hash, ("null equality", flip_nulls as Mistake)),
        (hash, ("inner join", always_inner)),
        (hash, ("no filter", no_filter)),
        (merge, ("null equality", flip_nulls)),
        (merge, ("inner join", always_inner)),
        (merge, ("no filter", no_filter)),
        (merge, ("sort order", flip_order)),
    ] {
        let name = format!("{join}, {name}");
        let plan = broken(build, mistake);
        let message = finds(&cases(), |case: JoinCase| {
            agree(&case, &*plan, &nested_loop_join)
        })
        .unwrap_or_else(|| panic!("{name}: not found"));
        assert!(message.contains("differ"), "{name}: {message}");
    }
}

/// A partition that yields a side's batches as drawn
#[derive(Debug)]
struct Drawn(SchemaRef, Vec<RecordBatch>);

impl PartitionStream for Drawn {
    fn schema(&self) -> &SchemaRef {
        &self.0
    }

    fn execute(&self, _: Arc<TaskContext>) -> SendableRecordBatchStream {
        let batches = self.1.clone().into_iter().map(Ok);
        Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.0),
            futures::stream::iter(batches),
        ))
    }
}

/// The order law finds a sort-merge join whose left input declares the
/// case's key order but yields its rows as drawn. The join keeps the order
/// of its left input for the join types that stream it, so its output is
/// out of the order it states
#[test]
fn finds_misordered_output() {
    let plan: Box<JoinPlan> = Box::new(|case, _, right| {
        let schema = &case.left_schema;
        let ordering = LexOrdering::new(case.sort_options.iter().flatten().enumerate().map(
            |(i, &o)| PhysicalSortExpr::new(Arc::new(Column::new(schema.field(i).name(), i)), o),
        ));
        let batches = case.left.iter().map(GeneratedBatch::build).collect();
        let left = StreamingTableExec::try_new(
            Arc::clone(schema),
            vec![Arc::new(Drawn(Arc::clone(schema), batches))],
            None,
            ordering,
            false,
            None,
        )?;
        sort_merge_join(case, Arc::new(left), right)
    });
    let message = finds(
        &sort_merge_joins(&key_types()).unwrap(),
        |case: JoinCase| filtered(&case, || check_join_order(&case, &*plan)),
    )
    .expect("not found");
    assert!(message.contains("sorts after"), "{message}");
}

/// The split law finds a partitioned hash join whose right input is split
/// by batch, round robin, rather than hash-partitioned on its keys, so that
/// which rows meet depends on where the batches split
#[test]
fn finds_a_join_that_depends_on_its_batches() {
    let plan: Box<JoinPlan> = Box::new(|case, left, right| {
        let keys = |schema: &Schema| {
            (0..case.keys)
                .map(|i| Arc::new(Column::new(schema.field(i).name(), i)) as Arc<dyn PhysicalExpr>)
                .collect::<Vec<_>>()
        };
        let left = RepartitionExec::try_new(left, Partitioning::Hash(keys(&case.left_schema), 2))?;
        let right = RepartitionExec::try_new(right, Partitioning::RoundRobinBatch(2))?;
        Ok(Arc::new(HashJoinExec::try_new(
            Arc::new(left),
            Arc::new(right),
            keys(&case.left_schema)
                .into_iter()
                .zip(keys(&case.right_schema))
                .collect(),
            None,
            &case.join_type,
            None,
            PartitionMode::Partitioned,
            case.null_equality,
            false,
        )?))
    });
    let cases = joins(&key_types())
        .unwrap()
        .prop_flat_map(|c| (Just(c.clone()), splits(&c)));
    let message = finds(&cases, |(case, split)| {
        filtered(&case, || check_join_split(&case, &*plan, &split))
    })
    .expect("not found");
    assert!(message.contains("differ"), "{message}");
}

/// The fetch law finds a hash join whose probe side reads all of its input
/// before the join sees a row, as a sort below the join does
#[test]
fn finds_a_probe_side_that_reads_ahead() {
    let plan: Box<JoinPlan> = Box::new(|case, left, right| {
        let key = Arc::new(Column::new(case.right_schema.field(0).name(), 0));
        // One sort expression, so the ordering is never empty
        let ordering = LexOrdering::new([PhysicalSortExpr::new(key, SortOptions::default())])
            .expect("one sort expression");
        hash_join(case, left, Arc::new(SortExec::new(ordering, right)))
    });
    let cases = (joins(&key_types()).unwrap(), 0.0..=1.0f64, any::<usize>());
    let message = finds(&cases, |(case, fraction, probe_batches)| {
        empty_build_reads_on(
            &case,
            filtered(&case, || {
                check_join_fetch(&case, &*plan, fraction, probe_batches)
            }),
        )
    })
    .expect("not found");
    assert!(message.contains("reads on"), "{message}");
}

/// A key type the generator cannot build, or no key types, is an error
#[test]
fn joins_of_an_unsupported_type() {
    assert!(joins(&[DataType::Float16]).is_err());
    assert!(joins(&[]).is_err());
}

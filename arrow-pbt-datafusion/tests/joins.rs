//! Laws for DataFusion's joins. The agreement law: a hash join or a
//! sort-merge join gives the same rows as a nested loop join of the same
//! inputs. The config law: a hash join gives the same rows under two session
//! configs. The order law: a sort-merge join returns its rows in the output
//! ordering its plan states.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type};
use arrow_pbt::generator::GeneratedBatch;
use arrow_pbt::known::{known, ree_values_nullable};
use arrow_pbt::run::{caught, check};
use arrow_pbt::types::{any_type, is_view_dict};
use arrow_pbt_datafusion::{
    JoinCase, JoinPlan, check_join_agreement, check_join_config, check_join_order, hash_join,
    joins, nested_loop_join, sort_merge_join, sort_merge_joins,
};
use arrow_schema::SchemaRef;
use arrow_schema::{DataType, Field, Fields, SortOptions, TimeUnit};
use datafusion_common::{JoinType, NullEquality};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::{LexOrdering, PhysicalSortExpr};
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::streaming::{PartitionStream, StreamingTableExec};
use proptest::strategy::BoxedStrategy;
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
    view_dict_overflow(case, null_padding(ree_values_nullable(caught(law))))
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
        let found = catch_unwind(AssertUnwindSafe(|| {
            check("broken", cases, |case: JoinCase| {
                agree(&case, &*plan, &nested_loop_join)
            });
        }));
        let Err(p) = found else {
            panic!("{name}: not found");
        };
        let message = p.downcast_ref::<String>().cloned().unwrap_or_default();
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
    let found = catch_unwind(AssertUnwindSafe(|| {
        check(
            "misordered",
            || sort_merge_joins(&key_types()).unwrap(),
            |case: JoinCase| filtered(&case, || check_join_order(&case, &*plan)),
        );
    }));
    let Err(p) = found else {
        panic!("not found");
    };
    let message = p.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(message.contains("sorts after"), "{message}");
}

/// A key type the generator cannot build, or no key types, is an error
#[test]
fn joins_of_an_unsupported_type() {
    assert!(joins(&[DataType::Float16]).is_err());
    assert!(joins(&[]).is_err());
}

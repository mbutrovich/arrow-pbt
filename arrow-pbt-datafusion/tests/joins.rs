//! Laws for DataFusion's joins. The agreement law: a hash join and a nested
//! loop join of the same inputs give the same rows. The config law: a hash
//! join gives the same rows under two session configs.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use arrow_pbt::known::{known, ree_values_nullable};
use arrow_pbt::run::{caught, check};
use arrow_pbt::types::{any_type, is_view_dict};
use arrow_pbt_datafusion::{
    JoinCase, JoinPlan, check_join_agreement, check_join_config, hash_join, joins, nested_loop_join,
};
use arrow_schema::{DataType, Field, Fields, TimeUnit};
use datafusion_common::{JoinType, NullEquality};
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

/// apache/datafusion#27399 moves inner, left, right, and full hash joins
/// to a new path behind `enable_migration_hash_join`, on by default, and
/// keeps the old path for one release. Both must give the same rows
#[test]
fn migration_hash_join_agrees_with_legacy() {
    let options = [("datafusion.execution.enable_migration_hash_join", "false")];
    check(
        "migration hash join against legacy",
        || joins(&key_types()).unwrap(),
        |case: JoinCase| filtered(&case, || check_join_config(&case, &hash_join, &options)),
    );
}

/// A hash join of the case's inputs with a mistake in how it is built
fn broken(mistake: fn(&JoinCase) -> JoinCase) -> Box<JoinPlan> {
    Box::new(move |case, left, right| hash_join(&mistake(case), left, right))
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
    for (name, mistake) in [
        ("null equality", flip_nulls as fn(&JoinCase) -> JoinCase),
        ("inner join", always_inner),
        ("no filter", no_filter),
    ] {
        let plan = broken(mistake);
        let found = catch_unwind(AssertUnwindSafe(|| {
            check(
                "broken",
                || joins(&key_types()).unwrap(),
                |case: JoinCase| agree(&case, &*plan, &nested_loop_join),
            );
        }));
        let Err(p) = found else {
            panic!("{name}: not found");
        };
        let message = p.downcast_ref::<String>().cloned().unwrap_or_default();
        assert!(message.contains("differ"), "{name}: {message}");
    }
}

/// A key type the generator cannot build, or no key types, is an error
#[test]
fn joins_of_an_unsupported_type() {
    assert!(joins(&[DataType::Float16]).is_err());
    assert!(joins(&[]).is_err());
}

//! `GeneratorType::try_from(&DataType)`: arrays built from the converted type
//! have exactly the given type, hold the generated rows, and are valid.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Decimal32Type, Decimal64Type, Decimal128Type, Decimal256Type};
use arrow_array::{Array, make_array};
use arrow_data::{ArrayData, BufferSpec, layout};
use arrow_pbt::generator::{
    GeneratedArray, GeneratorType, ValueDomain, all_types, array, batch, batches, build, len, rows,
    schemas, seed,
};
use arrow_pbt::run::check;
use arrow_pbt::val::{Val, check_rows};
use arrow_schema::{
    ArrowError, DataType, Field, FieldRef, Fields, IntervalUnit, Schema, SchemaRef, TimeUnit,
    UnionFields, UnionMode,
};
use proptest::prelude::*;
use proptest::sample::select;
use proptest::strategy::ValueTree;
use proptest::test_runner::{TestCaseError, TestRunner};

/// Whether a field of type `dt` can hold no nulls. Every `Null` row is null,
/// and a union or run-end encoded row is null when its child's value is
fn can_be_non_nullable(dt: &DataType) -> bool {
    match dt {
        DataType::Null => false,
        DataType::Union(fs, _) => fs.iter().all(|(_, f)| can_be_non_nullable(f.data_type())),
        DataType::RunEndEncoded(_, v) => can_be_non_nullable(v.data_type()),
        _ => true,
    }
}

/// `f` with a new name, an Iceberg-style field id in its metadata, and
/// nullability cleared every third field where that is possible
fn relabel_field(f: &FieldRef, n: &mut usize) -> FieldRef {
    *n += 1;
    let id = *n;
    let dt = relabel(f.data_type(), n);
    let nullable = f.is_nullable() && (!id.is_multiple_of(3) || !can_be_non_nullable(&dt));
    let metadata = HashMap::from([("PARQUET:field_id".to_string(), id.to_string())]);
    Arc::new(Field::new(format!("c{id}"), dt, nullable).with_metadata(metadata))
}

/// `dt` with every child field passed through `relabel_field`
fn relabel(dt: &DataType, n: &mut usize) -> DataType {
    match dt {
        DataType::List(f) => DataType::List(relabel_field(f, n)),
        DataType::LargeList(f) => DataType::LargeList(relabel_field(f, n)),
        DataType::ListView(f) => DataType::ListView(relabel_field(f, n)),
        DataType::LargeListView(f) => DataType::LargeListView(relabel_field(f, n)),
        DataType::FixedSizeList(f, k) => DataType::FixedSizeList(relabel_field(f, n), *k),
        DataType::Struct(fs) => DataType::Struct(fs.iter().map(|f| relabel_field(f, n)).collect()),
        // The entries field and its key may not be null, so only names and
        // metadata change there
        DataType::Map(entries, sorted) => {
            let DataType::Struct(kv) = entries.data_type() else {
                unreachable!("map entries are a struct")
            };
            let key = relabel_field(&kv[0], n);
            let key = Arc::new(key.as_ref().clone().with_nullable(false));
            let kv = Fields::from(vec![key, relabel_field(&kv[1], n)]);
            let entries = Field::new("key_value", DataType::Struct(kv), false);
            DataType::Map(Arc::new(entries), *sorted)
        }
        DataType::Union(fs, mode) => DataType::Union(
            UnionFields::try_new(
                fs.iter().map(|(id, _)| id),
                fs.iter().map(|(_, f)| relabel_field(f, n)),
            )
            .unwrap(),
            *mode,
        ),
        DataType::Dictionary(k, v) => DataType::Dictionary(k.clone(), Box::new(relabel(v, n))),
        DataType::RunEndEncoded(r, v) => DataType::RunEndEncoded(r.clone(), relabel_field(v, n)),
        dt => dt.clone(),
    }
}

/// Builds `dt` through the conversion with a few row sets and seeds, and
/// checks each array's type, validity and values
fn check_builds(dt: &DataType) {
    let t = GeneratorType::try_from(dt).unwrap_or_else(|e| panic!("{dt}: {e}"));
    assert_eq!(&t.data_type(), dt);
    let mut runner = TestRunner::deterministic();
    for n in [0, 1, 7, 64, 300] {
        let r = rows(&t, n).new_tree(&mut runner).unwrap().current();
        for s in [0, 1, 2, 3] {
            if let Err(TestCaseError::Fail(e)) = check_rows(build(&t, &r, s).as_ref(), dt, &r) {
                panic!("{dt}, {n} rows, seed {s}: {e}");
            }
        }
    }
}

fn field(name: &str, dt: DataType, nullable: bool, id: u32) -> Field {
    Field::new(name, dt, nullable).with_metadata(HashMap::from([(
        "PARQUET:field_id".to_string(),
        id.to_string(),
    )]))
}

proptest! {
    // No regression files, as in the crate's own runner
    #![proptest_config(ProptestConfig {
        cases: 256,
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    /// Every type the generator draws, with every child field renamed, given
    /// metadata, and sometimes made non-nullable
    #[test]
    fn keeps_any_data_type(
        (dt, r, s) in all_types()
            .prop_map(|t| relabel(&t.data_type(), &mut 0))
            .prop_flat_map(|dt| {
                let t = GeneratorType::try_from(&dt).unwrap_or_else(|e| panic!("{dt}: {e}"));
                (Just(dt), len().prop_flat_map(move |n| rows(&t, n)), seed())
            })
    ) {
        let t = GeneratorType::try_from(&dt).unwrap();
        check_rows(build(&t, &r, s).as_ref(), &dt, &r)?;
    }

    /// The same, as a non-nullable field wherever the type allows one: no
    /// row is null, at the top level or through a union or run-end encoded
    /// array's children
    #[test]
    fn keeps_any_non_nullable_field(
        (f, r, s) in all_types()
            .prop_map(|t| relabel(&t.data_type(), &mut 0))
            .prop_filter("a type that can be non-nullable", can_be_non_nullable)
            .prop_flat_map(|dt| {
                let f = Field::new("c", dt, false);
                let t = GeneratorType::try_from(&f).unwrap_or_else(|e| panic!("{f}: {e}"));
                (Just(f), len().prop_flat_map(move |n| rows(&t, n)), seed())
            })
    ) {
        let t = GeneratorType::try_from(&f).unwrap();
        let a = build(&t, &r, s);
        prop_assert!(no_logical_nulls(a.as_ref()), "a null row");
        check_rows(a.as_ref(), f.data_type(), &r)?;
    }
}

#[test]
fn leaves() {
    for dt in [
        DataType::Null,
        DataType::Boolean,
        DataType::Int8,
        DataType::UInt64,
        DataType::Float32,
        DataType::Float64,
        DataType::Decimal32(9, 9),
        DataType::Decimal64(1, 0),
        DataType::Decimal128(20, 3),
        DataType::Decimal128(38, -5),
        DataType::Decimal256(76, 18),
        DataType::Date32,
        DataType::Date64,
        DataType::Time32(TimeUnit::Second),
        DataType::Time64(TimeUnit::Nanosecond),
        DataType::Timestamp(TimeUnit::Microsecond, Some("+00:00".into())),
        DataType::Timestamp(TimeUnit::Nanosecond, Some("America/New_York".into())),
        DataType::Duration(TimeUnit::Millisecond),
        DataType::Interval(arrow_schema::IntervalUnit::MonthDayNano),
        DataType::Utf8,
        DataType::LargeUtf8,
        DataType::Utf8View,
        DataType::Binary,
        DataType::LargeBinary,
        DataType::BinaryView,
        DataType::FixedSizeBinary(0),
        DataType::FixedSizeBinary(16),
    ] {
        check_builds(&dt);
    }
}

/// A table schema as iceberg-rust converts it: field ids on every field,
/// `element` for list items, and `key_value` for map entries
#[test]
fn iceberg_schema() {
    let element = field("element", DataType::Utf8, true, 3);
    let kv = Fields::from(vec![
        field("key", DataType::Utf8, false, 5),
        field("value", DataType::Int32, true, 6),
    ]);
    let dt = DataType::Struct(Fields::from(vec![
        field("id", DataType::Int64, false, 1),
        field("tags", DataType::List(Arc::new(element)), true, 2),
        field(
            "props",
            DataType::Map(
                Arc::new(Field::new("key_value", DataType::Struct(kv), false)),
                false,
            ),
            true,
            4,
        ),
    ]));
    check_builds(&dt);
}

/// Dictionary and run-end encoded values of types outside the generator's
/// own pools, with keys too narrow for distinct values on every row
#[test]
fn encoded_values_of_any_type() {
    let item = Arc::new(Field::new("item", DataType::Int32, true));
    for dt in [
        DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::LargeUtf8)),
        DataType::Dictionary(Box::new(DataType::UInt8), Box::new(DataType::Float64)),
        DataType::Dictionary(
            Box::new(DataType::Int16),
            Box::new(DataType::Decimal128(20, 3)),
        ),
        DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::List(item))),
        DataType::RunEndEncoded(
            Arc::new(Field::new("run_ends", DataType::Int16, false)),
            Arc::new(Field::new("values", DataType::LargeBinary, true)),
        ),
        DataType::RunEndEncoded(
            Arc::new(Field::new("ends", DataType::Int64, false)),
            Arc::new(Field::new("v", DataType::Boolean, false)),
        ),
    ] {
        check_builds(&dt);
    }
}

/// Degenerate shapes and a union with non-default type ids
#[test]
fn degenerate_and_unions() {
    let item = Arc::new(Field::new("x", DataType::Int32, false));
    let union_fields = UnionFields::try_new(
        [5, 2],
        [
            Field::new("a", DataType::Utf8, true),
            Field::new("b", DataType::Int64, false),
        ],
    )
    .unwrap();
    for dt in [
        DataType::Struct(Fields::empty()),
        DataType::FixedSizeList(item.clone(), 0),
        DataType::FixedSizeList(item.clone(), 3),
        DataType::LargeListView(item),
        DataType::Union(union_fields.clone(), UnionMode::Sparse),
        DataType::Union(union_fields, UnionMode::Dense),
    ] {
        check_builds(&dt);
    }
}

/// A converted type as the child of a type the generator draws itself
#[test]
fn nested_in_a_generator_type() {
    let inner = DataType::Struct(Fields::from(vec![field("id", DataType::Int64, false, 1)]));
    let t = GeneratorType::List(Box::new(GeneratorType::try_from(&inner).unwrap()));
    let DataType::List(item) = t.data_type() else {
        unreachable!()
    };
    assert_eq!(item.data_type(), &inner);
    let mut runner = TestRunner::deterministic();
    let r = rows(&t, 50).new_tree(&mut runner).unwrap().current();
    for s in 0..4 {
        if let Err(TestCaseError::Fail(e)) =
            check_rows(build(&t, &r, s).as_ref(), &t.data_type(), &r)
        {
            panic!("seed {s}: {e}");
        }
    }
}

#[test]
fn unsupported_types_are_errors() {
    let null_item = Arc::new(Field::new("item", DataType::Null, false));
    let list_key = Fields::from(vec![
        Field::new(
            "key",
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
            false,
        ),
        Field::new("value", DataType::Int32, true),
    ]);
    for dt in [
        DataType::Float16,
        DataType::Struct(Fields::from(vec![Field::new("h", DataType::Float16, true)])),
        DataType::Time32(TimeUnit::Microsecond),
        DataType::Time64(TimeUnit::Second),
        DataType::Decimal128(0, 0),
        DataType::Decimal128(39, 0),
        DataType::Decimal32(5, 6),
        DataType::FixedSizeBinary(-1),
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Int32, true)), -1),
        DataType::Dictionary(Box::new(DataType::Utf8), Box::new(DataType::Utf8)),
        DataType::RunEndEncoded(
            Arc::new(Field::new("run_ends", DataType::Int8, false)),
            Arc::new(Field::new("values", DataType::Int32, true)),
        ),
        DataType::List(null_item),
        DataType::Map(
            Arc::new(Field::new("entries", DataType::Struct(list_key), false)),
            true,
        ),
        DataType::Union(UnionFields::empty(), UnionMode::Sparse),
    ] {
        assert!(GeneratorType::try_from(&dt).is_err(), "{dt} converted");
    }
}

/// Whether no row of `a` is null, counting the nulls a union or run-end
/// encoded array takes from its children
fn no_logical_nulls(a: &dyn Array) -> bool {
    a.logical_null_count() == 0
}

/// A non-nullable field's type, at the top level, makes no null rows
#[test]
fn non_nullable_fields() {
    let union_fields = UnionFields::try_new(
        [0, 1],
        [
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Utf8, true),
        ],
    )
    .unwrap();
    for dt in [
        DataType::Int32,
        DataType::Utf8,
        DataType::Float64,
        DataType::Decimal128(20, 3),
        DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
        DataType::Struct(Fields::from(vec![Field::new("a", DataType::Int64, true)])),
        DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::LargeUtf8)),
        DataType::Union(union_fields.clone(), UnionMode::Sparse),
        DataType::Union(union_fields, UnionMode::Dense),
        DataType::RunEndEncoded(
            Arc::new(Field::new("run_ends", DataType::Int32, false)),
            Arc::new(Field::new("values", DataType::Int32, true)),
        ),
    ] {
        let f = Field::new("c", dt.clone(), false);
        let t = GeneratorType::try_from(&f).unwrap_or_else(|e| panic!("{f}: {e}"));
        assert_eq!(t.data_type(), dt);
        let mut runner = TestRunner::deterministic();
        for n in [1, 7, 64, 300] {
            let r = rows(&t, n).new_tree(&mut runner).unwrap().current();
            assert!(!r.contains(&Val::Null), "{f}: a null row");
            for s in [0, 1, 2, 3] {
                let a = build(&t, &r, s);
                assert!(no_logical_nulls(a.as_ref()), "{f}, seed {s}: a null row");
                if let Err(TestCaseError::Fail(e)) = check_rows(a.as_ref(), &dt, &r) {
                    panic!("{f}, {n} rows, seed {s}: {e}");
                }
            }
        }
    }
    // A nullable field converts as its type does
    let f = Field::new("c", DataType::Utf8, true);
    assert_eq!(
        GeneratorType::try_from(&f).unwrap().data_type(),
        DataType::Utf8
    );
    // Every row of a `Null` field is null
    assert!(GeneratorType::try_from(&Field::new("c", DataType::Null, false)).is_err());
}

/// Arrays of a given type and length, including no rows, a non-nullable
/// type and a converted one
#[test]
fn arrays_of_a_type_and_length() {
    let element = Arc::new(field("element", DataType::Utf8, false, 2));
    let types = [
        GeneratorType::try_from(&Field::new("a", DataType::Int64, false)).unwrap(),
        GeneratorType::try_from(&DataType::List(element)).unwrap(),
        GeneratorType::Utf8View,
    ];
    let mut runner = TestRunner::deterministic();
    for n in [0, 1, 300] {
        for t in &types {
            let g = array(t, n, ValueDomain::Arrow)
                .new_tree(&mut runner)
                .unwrap()
                .current();
            assert_eq!(g.ty.data_type(), t.data_type());
            assert_eq!(g.rows.len(), n);
            if let Err(TestCaseError::Fail(e)) =
                check_rows(g.build().as_ref(), &t.data_type(), &g.rows)
            {
                panic!("{n} rows: {e}");
            }
        }
        let g = array(&types[0], n, ValueDomain::Arrow)
            .new_tree(&mut runner)
            .unwrap()
            .current();
        assert!(
            !g.rows.contains(&Val::Null),
            "a null in a non-nullable array"
        );
    }
}

/// Some arrays put their values at exactly arrow-rs's minimum alignment, which
/// for these types is below their width, and none goes below it
#[test]
fn values_at_the_minimum_alignment() {
    let mut runner = TestRunner::deterministic();
    for dt in [
        DataType::Interval(IntervalUnit::DayTime),
        DataType::Interval(IntervalUnit::MonthDayNano),
        DataType::Decimal256(76, 2),
    ] {
        let t = GeneratorType::try_from(&dt).unwrap();
        let width = dt.primitive_width().unwrap();
        let BufferSpec::FixedWidth { alignment, .. } = layout(&dt).buffers[0] else {
            unreachable!("{dt} is fixed width")
        };
        let mut below_width = 0;
        for n in 1..=200 {
            let a = array(&t, n, ValueDomain::Arrow)
                .new_tree(&mut runner)
                .unwrap()
                .current()
                .build();
            let d = a.to_data();
            let address = d.buffers()[0].as_ptr() as usize + d.offset() * width;
            assert_eq!(address % alignment, 0, "{dt}");
            below_width += usize::from(!address.is_multiple_of(width));
        }
        assert!(
            below_width > 0,
            "{dt}: no values below {width}-byte alignment"
        );
    }
}

/// Each value in a valid decimal slot of `d`, or of any array inside it,
/// fits the declared precision, as Arrow C++'s full validation requires
fn within_precision(d: &ArrayData) -> Result<(), ArrowError> {
    let a = make_array(d.clone());
    match d.data_type() {
        DataType::Decimal32(p, _) => a
            .as_primitive::<Decimal32Type>()
            .validate_decimal_precision(*p)?,
        DataType::Decimal64(p, _) => a
            .as_primitive::<Decimal64Type>()
            .validate_decimal_precision(*p)?,
        DataType::Decimal128(p, _) => a
            .as_primitive::<Decimal128Type>()
            .validate_decimal_precision(*p)?,
        DataType::Decimal256(p, _) => a
            .as_primitive::<Decimal256Type>()
            .validate_decimal_precision(*p)?,
        _ => {}
    }
    d.child_data().iter().try_for_each(within_precision)
}

/// Decimal types on their own, and inside the types whose layouts fill
/// valid slots that hold no row: list children, dictionary values, run-end
/// encoded values, struct children, and union children
fn decimal_types() -> BoxedStrategy<GeneratorType> {
    let d = DataType::Decimal128(10, 2);
    let field = |name: &str, dt: DataType| Field::new(name, dt, true);
    let union = |mode| {
        DataType::Union(
            UnionFields::try_new(
                [0, 1],
                [
                    field("a", d.clone()),
                    field("b", DataType::Decimal256(20, 0)),
                ],
            )
            .unwrap(),
            mode,
        )
    };
    let types = [
        DataType::Decimal32(9, 2),
        DataType::Decimal64(18, 4),
        DataType::Decimal128(38, 10),
        DataType::Decimal256(20, 0),
        DataType::List(Arc::new(field("item", d.clone()))),
        DataType::FixedSizeList(Arc::new(field("item", DataType::Decimal32(5, 0))), 2),
        DataType::Dictionary(
            Box::new(DataType::Int8),
            Box::new(DataType::Decimal64(10, 3)),
        ),
        DataType::RunEndEncoded(
            Arc::new(Field::new("run_ends", DataType::Int32, false)),
            Arc::new(field("values", d.clone())),
        ),
        DataType::Struct(Fields::from(vec![field("a", d.clone())])),
        union(UnionMode::Sparse),
        union(UnionMode::Dense),
    ];
    select(
        types
            .iter()
            .map(|dt| GeneratorType::try_from(dt).unwrap_or_else(|e| panic!("{dt}: {e}")))
            .collect::<Vec<_>>(),
    )
    .boxed()
}

fn arrays_in(domain: ValueDomain) -> BoxedStrategy<GeneratedArray> {
    (prop_oneof![decimal_types(), all_types()], len())
        .prop_flat_map(move |(t, n)| array(&t, n, domain))
        .boxed()
}

/// With `ValueDomain::Logical`, every valid decimal slot fits its precision,
/// in the rows and in the slots the layout fills
#[test]
fn logical_decimals_fit_their_precision() {
    check(
        "logical decimals",
        || arrays_in(ValueDomain::Logical),
        |g| within_precision(&g.build().to_data()).map_err(|e| TestCaseError::fail(e.to_string())),
    );
}

/// The default domain draws decimals past their precision, so the check
/// above can fail
#[test]
fn arrow_decimals_can_exceed_their_precision() {
    let mut runner = TestRunner::deterministic();
    let strategy = arrays_in(ValueDomain::Arrow);
    let past = (0..256).any(|_| {
        let g = strategy.new_tree(&mut runner).unwrap().current();
        within_precision(&g.build().to_data()).is_err()
    });
    assert!(past, "no decimal past its precision in 256 arrays");
}

/// Batches of a given schema have that schema exactly, metadata included,
/// the requested rows, and columns that hold their generated rows, for a
/// schema with no fields too
#[test]
fn batches_of_a_schema() {
    let element = Arc::new(field("element", DataType::Utf8, false, 3));
    let schema: SchemaRef = Arc::new(Schema::new_with_metadata(
        vec![
            field("id", DataType::Int64, false, 1),
            field("names", DataType::List(element), true, 2),
            Field::new("price", DataType::Decimal128(5, 2), true),
        ],
        HashMap::from([("k".to_string(), "v".to_string())]),
    ));
    let mut runner = TestRunner::deterministic();
    for s in [schema, Arc::new(Schema::empty())] {
        for n in [0, 1, 300] {
            for domain in [ValueDomain::Arrow, ValueDomain::Logical] {
                let g = batch(&s, n, domain)
                    .unwrap()
                    .new_tree(&mut runner)
                    .unwrap()
                    .current();
                let b = g.build();
                assert_eq!(b.schema(), s);
                assert_eq!(b.num_rows(), n);
                assert_eq!(g.columns.len(), s.fields().len());
                for (c, col) in b.columns().iter().zip(&g.columns) {
                    assert_eq!(col.domain, domain);
                    if let Err(TestCaseError::Fail(e)) = check_rows(c, c.data_type(), &col.rows) {
                        panic!("{n} rows, {domain:?}: {e}");
                    }
                    if domain == ValueDomain::Logical {
                        within_precision(&c.to_data()).unwrap();
                    }
                }
                if n > 0 && !s.fields().is_empty() {
                    assert_eq!(
                        b.column(0).null_count(),
                        0,
                        "a null in a non-nullable column"
                    );
                }
            }
        }
    }
}

/// A schema with a type the generator cannot build is an error that names
/// the field
#[test]
fn batches_of_an_unsupported_schema() {
    let s: SchemaRef = Arc::new(Schema::new(vec![Field::new(
        "half",
        DataType::Float16,
        true,
    )]));
    let e = batch(&s, 1, ValueDomain::Arrow).err().unwrap();
    assert!(e.contains("field half"), "{e}");
    assert!(batches(&s, ValueDomain::Arrow).is_err());
}

/// Streams of batches of generated schemas share their schema, and each
/// batch holds its rows
#[test]
fn streams_of_generated_schemas() {
    let strategy = || {
        schemas().prop_flat_map(|s| {
            batches(&s, ValueDomain::Arrow)
                .unwrap()
                .prop_map(move |b| (s.clone(), b))
        })
    };
    check("streams of generated schemas", strategy, |(s, gs)| {
        for g in &gs {
            let b = g.build();
            prop_assert_eq!(b.schema(), s.clone());
            prop_assert_eq!(b.num_rows(), g.num_rows);
            for (c, col) in b.columns().iter().zip(&g.columns) {
                check_rows(c, c.data_type(), &col.rows)?;
            }
        }
        Ok(())
    });
}

/// A stream is sometimes no batches, and sometimes holds an empty batch or
/// batches of different lengths
#[test]
fn streams_vary_in_shape() {
    let s: SchemaRef = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
    let strategy = batches(&s, ValueDomain::Arrow).unwrap();
    let mut runner = TestRunner::deterministic();
    let (mut none, mut empty, mut uneven) = (false, false, false);
    for _ in 0..256 {
        let gs = strategy.new_tree(&mut runner).unwrap().current();
        none |= gs.is_empty();
        empty |= gs.iter().any(|g| g.num_rows == 0);
        uneven |= gs.windows(2).any(|w| w[0].num_rows != w[1].num_rows);
    }
    assert!(
        none && empty && uneven,
        "none {none}, empty {empty}, uneven {uneven}"
    );
}

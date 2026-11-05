//! The generator. A case is a type, logical rows, and a layout seed.
//! The seed drives every physical choice in `build`, and seed 0 is the plain
//! layout, so shrinking the seed moves towards it.
//!
//! Designed from bug classes in arrow-rs issues, not from individual fixes:
//! degenerate shapes, slicing at every level, data under nulls, rare types,
//! dictionary encodings, view buffers, and sizes at byte and word boundaries.

use std::sync::Arc;

use arrow_array::builder::{BinaryViewBuilder, StringViewBuilder};
use arrow_array::types::{
    Decimal32Type, Decimal64Type, Decimal128Type, Decimal256Type, DecimalType, Int32Type,
    validate_decimal_precision_and_scale,
};
use arrow_array::{
    Array, ArrayRef, BinaryArray, BooleanArray, DictionaryArray, FixedSizeBinaryArray,
    FixedSizeListArray, Int8Array, Int32Array, Int64Array, LargeListArray, LargeStringArray,
    ListArray, ListViewArray, NullArray, RecordBatch, RecordBatchOptions, RunArray, StringArray,
    StructArray, UnionArray, make_array,
};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_data::{ArrayData, BufferSpec, layout};
use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef, UnionFields, UnionMode};
use num_bigint::{BigInt, Sign};
use num_traits::{Signed, ToPrimitive, Zero};
use proptest::collection::vec;
use proptest::prelude::*;
use proptest::sample::select;
use proptest::strategy::ValueTree;
use proptest::test_runner::TestRunner;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::props::leaf_cmp;
use crate::types::any_type;
use crate::val::Val;
use arrow_array::Int16Array;
use arrow_array::LargeBinaryArray;
use arrow_array::LargeListViewArray;
use arrow_array::StringViewArray;
use arrow_array::UInt8Array;
use arrow_array::UInt16Array;
use arrow_array::UInt32Array;
use arrow_array::UInt64Array;
use arrow_array::ffi::from_ffi;
use arrow_array::ffi::to_ffi;
use arrow_buffer::MutableBuffer;
use arrow_schema::IntervalUnit;
use num_traits::pow;
use proptest::bool::weighted;
use std::iter::repeat_n;
use std::iter::successors;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

/// A type the generator can produce. It maps to a `DataType` through
/// `data_type`, and adds what the generator needs and `DataType` does not
/// say: whether a child field may be null (`NotNull`), the union type ids to
/// use, and separate value pools for `Int32` and `Int64`
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum GeneratorType {
    Null,
    Bool,
    I32,
    I64,
    /// Any other fixed-width type: integers of other widths, floats,
    /// decimals, dates, times, timestamps and durations, built from and read
    /// as little-endian bytes
    Prim(DataType),
    Utf8,
    LargeUtf8,
    Utf8View,
    Binary,
    LargeBinary,
    BinaryView,
    Fsb(i32),
    List(Box<GeneratorType>),
    LargeList(Box<GeneratorType>),
    ListView(Box<GeneratorType>),
    LargeListView(Box<GeneratorType>),
    Fsl(Box<GeneratorType>, i32),
    Struct(Vec<GeneratorType>),
    /// Key type and value type
    Dict(DataType, Box<GeneratorType>),
    /// Run end type and value type
    Ree(DataType, Box<GeneratorType>),
    Union(Vec<GeneratorType>, UnionMode, Vec<i8>),
    /// Key type, value type, and whether keys are sorted
    Map(Box<GeneratorType>, Box<GeneratorType>, bool),
    /// A child whose field is not nullable, so its array holds no nulls
    NotNull(Box<GeneratorType>),
    /// A type converted from a `DataType` by `GeneratorType::try_from`, whose
    /// arrays have that `DataType` exactly
    Exact(Exact),
}

/// A `DataType` and the type it is built as. The two differ only in child
/// field names, nullability and metadata, which `build` then sets to the
/// `DataType`'s. Only `GeneratorType::try_from` constructs one, which keeps
/// that true
#[derive(Clone, Debug)]
pub struct Exact {
    dt: DataType,
    built_as: Box<GeneratorType>,
}

fn field(name: &str, t: &GeneratorType) -> Field {
    Field::new(name, t.data_type(), !matches!(t, GeneratorType::NotNull(_)))
}

/// `t` as a child that may not be null. Null arrays are all null, and
/// unions and run-end encoded arrays take their nulls from their children,
/// whose own fields say whether they may be null
fn not_null(t: GeneratorType) -> GeneratorType {
    match t {
        GeneratorType::Null | GeneratorType::Union(..) | GeneratorType::Ree(..) => t,
        t => GeneratorType::NotNull(Box::new(t)),
    }
}

/// A child type, not nullable one time in four
fn child(t: BoxedStrategy<GeneratorType>) -> BoxedStrategy<GeneratorType> {
    prop_oneof![3 => t.clone(), 1 => t.prop_map(not_null)].boxed()
}

impl GeneratorType {
    /// The Arrow type of arrays built from this type
    ///
    /// # Example
    ///
    /// ```
    /// use arrow_pbt::generator::GeneratorType;
    /// use arrow_schema::DataType;
    /// let t = GeneratorType::try_from(&DataType::Float64).unwrap();
    /// assert_eq!(t.data_type(), DataType::Float64);
    /// ```
    pub fn data_type(&self) -> DataType {
        match self {
            GeneratorType::Null => DataType::Null,
            GeneratorType::Bool => DataType::Boolean,
            GeneratorType::I32 => DataType::Int32,
            GeneratorType::I64 => DataType::Int64,
            GeneratorType::Prim(dt) => dt.clone(),
            GeneratorType::Utf8 => DataType::Utf8,
            GeneratorType::LargeUtf8 => DataType::LargeUtf8,
            GeneratorType::Utf8View => DataType::Utf8View,
            GeneratorType::Binary => DataType::Binary,
            GeneratorType::LargeBinary => DataType::LargeBinary,
            GeneratorType::BinaryView => DataType::BinaryView,
            GeneratorType::Fsb(w) => DataType::FixedSizeBinary(*w),
            GeneratorType::List(c) => DataType::List(Arc::new(field("item", c))),
            GeneratorType::LargeList(c) => DataType::LargeList(Arc::new(field("item", c))),
            GeneratorType::ListView(c) => DataType::ListView(Arc::new(field("item", c))),
            GeneratorType::LargeListView(c) => DataType::LargeListView(Arc::new(field("item", c))),
            GeneratorType::Fsl(c, n) => DataType::FixedSizeList(Arc::new(field("item", c)), *n),
            GeneratorType::Struct(fs) => DataType::Struct(struct_fields(fs)),
            GeneratorType::Dict(k, v) => {
                DataType::Dictionary(Box::new(k.clone()), Box::new(v.data_type()))
            }
            GeneratorType::Ree(r, v) => DataType::RunEndEncoded(
                Arc::new(Field::new("run_ends", r.clone(), false)),
                Arc::new(field("values", v)),
            ),
            GeneratorType::Union(cs, mode, ids) => DataType::Union(union_fields(cs, ids), *mode),
            GeneratorType::Map(k, v, sorted) => DataType::Map(
                Arc::new(Field::new("entries", map_entries(k, v), false)),
                *sorted,
            ),
            GeneratorType::NotNull(t) => t.data_type(),
            GeneratorType::Exact(e) => e.dt.clone(),
        }
    }
}

/// The generator type for a `DataType`. Arrays built from it have that type
/// exactly, with every child field's name, nullability and metadata, and hold
/// no null where a field is not nullable
///
/// # Errors
///
/// A type the generator cannot build, such as `Float16`, a type arrow-rs
/// rejects, such as `Time32(Microsecond)`, or a non-nullable field of type
/// `Null`, which has no valid rows
///
/// # Example
///
/// ```
/// use std::collections::HashMap;
/// use std::sync::Arc;
/// use arrow_array::Array;
/// use arrow_pbt::generator::{GeneratorType, build, rows};
/// use arrow_schema::{DataType, Field};
/// use proptest::strategy::{Strategy, ValueTree};
/// use proptest::test_runner::TestRunner;
/// // A list as iceberg-rust writes it: `element` items with a field id
/// let id = HashMap::from([("PARQUET:field_id".to_string(), "1".to_string())]);
/// let dt = DataType::List(Arc::new(
///     Field::new("element", DataType::Utf8, false).with_metadata(id),
/// ));
/// let t = GeneratorType::try_from(&dt).unwrap();
/// let mut runner = TestRunner::deterministic();
/// let r = rows(&t, 10).new_tree(&mut runner).unwrap().current();
/// assert_eq!(build(&t, &r, 1).data_type(), &dt);
/// assert!(GeneratorType::try_from(&DataType::Float16).is_err());
/// ```
impl TryFrom<&DataType> for GeneratorType {
    type Error = String;

    fn try_from(dt: &DataType) -> Result<Self, String> {
        Ok(exact(convert(dt)?, dt))
    }
}

/// `t`, built as `dt` when its own type names or annotates fields
/// differently
fn exact(t: GeneratorType, dt: &DataType) -> GeneratorType {
    if t.data_type() == *dt {
        t
    } else {
        GeneratorType::Exact(Exact {
            dt: dt.clone(),
            built_as: Box::new(t),
        })
    }
}

/// The generator type for a column or child field: its `DataType`, as the
/// conversion from a `DataType` gives it, holding no nulls when the field is
/// not nullable
///
/// # Errors
///
/// Those of the conversion from the field's `DataType`
///
/// # Example
///
/// ```
/// use arrow_pbt::generator::{GeneratorType, rows};
/// use arrow_pbt::val::Val;
/// use arrow_schema::{DataType, Field};
/// use proptest::strategy::{Strategy, ValueTree};
/// use proptest::test_runner::TestRunner;
/// let t = GeneratorType::try_from(&Field::new("id", DataType::Int64, false)).unwrap();
/// let mut runner = TestRunner::deterministic();
/// let r = rows(&t, 100).new_tree(&mut runner).unwrap().current();
/// assert!(!r.contains(&Val::Null));
/// ```
impl TryFrom<&Field> for GeneratorType {
    type Error = String;

    fn try_from(f: &Field) -> Result<Self, String> {
        Ok(exact(from_field(f)?, f.data_type()))
    }
}

/// `dt` as the generator builds it, with the child field names `data_type`
/// gives
fn convert(dt: &DataType) -> Result<GeneratorType, String> {
    use arrow_schema::TimeUnit::{Microsecond, Millisecond, Nanosecond, Second};
    Ok(match dt {
        DataType::Null => GeneratorType::Null,
        DataType::Boolean => GeneratorType::Bool,
        DataType::Int32 => GeneratorType::I32,
        DataType::Int64 => GeneratorType::I64,
        DataType::Utf8 => GeneratorType::Utf8,
        DataType::LargeUtf8 => GeneratorType::LargeUtf8,
        DataType::Utf8View => GeneratorType::Utf8View,
        DataType::Binary => GeneratorType::Binary,
        DataType::LargeBinary => GeneratorType::LargeBinary,
        DataType::BinaryView => GeneratorType::BinaryView,
        DataType::Int8
        | DataType::Int16
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float32
        | DataType::Float64
        | DataType::Date32
        | DataType::Date64
        | DataType::Time32(Second | Millisecond)
        | DataType::Time64(Microsecond | Nanosecond)
        | DataType::Timestamp(..)
        | DataType::Duration(_)
        | DataType::Interval(_) => GeneratorType::Prim(dt.clone()),
        DataType::Decimal32(p, s) => decimal::<Decimal32Type>(dt, *p, *s)?,
        DataType::Decimal64(p, s) => decimal::<Decimal64Type>(dt, *p, *s)?,
        DataType::Decimal128(p, s) => decimal::<Decimal128Type>(dt, *p, *s)?,
        DataType::Decimal256(p, s) => decimal::<Decimal256Type>(dt, *p, *s)?,
        DataType::FixedSizeBinary(w) if *w >= 0 => GeneratorType::Fsb(*w),
        DataType::List(f) => GeneratorType::List(Box::new(from_field(f)?)),
        DataType::LargeList(f) => GeneratorType::LargeList(Box::new(from_field(f)?)),
        DataType::ListView(f) => GeneratorType::ListView(Box::new(from_field(f)?)),
        DataType::LargeListView(f) => GeneratorType::LargeListView(Box::new(from_field(f)?)),
        DataType::FixedSizeList(f, n) if *n >= 0 => {
            GeneratorType::Fsl(Box::new(from_field(f)?), *n)
        }
        DataType::Struct(fs) => {
            GeneratorType::Struct(fs.iter().map(|f| from_field(f)).collect::<Result<_, _>>()?)
        }
        DataType::Union(fs, mode) if !fs.is_empty() => GeneratorType::Union(
            fs.iter()
                .map(|(_, f)| from_field(f))
                .collect::<Result<_, _>>()?,
            *mode,
            fs.iter().map(|(id, _)| id).collect(),
        ),
        DataType::Dictionary(k, v) if k.is_dictionary_key_type() => {
            GeneratorType::Dict((**k).clone(), Box::new(convert(v)?))
        }
        DataType::RunEndEncoded(r, v) if r.data_type().is_run_ends_type() => {
            GeneratorType::Ree(r.data_type().clone(), Box::new(from_field(v)?))
        }
        // The generator never puts a null in a key, whatever the key field
        // says, as the format requires
        DataType::Map(entries, sorted) => {
            let DataType::Struct(kv) = entries.data_type() else {
                return Err(format!("{dt}: map entries are not a struct"));
            };
            let [k, v] = &kv[..] else {
                return Err(format!("{dt}: map entries do not have two fields"));
            };
            let kt = k.data_type();
            // Sorting compares keys with `leaf_cmp`, which orders leaves only
            let orderable = kt.is_primitive()
                || matches!(
                    kt,
                    DataType::Boolean
                        | DataType::Utf8
                        | DataType::LargeUtf8
                        | DataType::Utf8View
                        | DataType::Binary
                        | DataType::LargeBinary
                        | DataType::BinaryView
                        | DataType::FixedSizeBinary(_)
                );
            if *kt == DataType::Null || (*sorted && !orderable) {
                return Err(format!("{dt}: the generator cannot build these keys"));
            }
            GeneratorType::Map(Box::new(convert(kt)?), Box::new(from_field(v)?), *sorted)
        }
        dt => return Err(format!("the generator cannot build {dt}")),
    })
}

/// `dt`, a decimal type, if arrow-rs accepts its precision and scale
fn decimal<T: DecimalType>(dt: &DataType, p: u8, s: i8) -> Result<GeneratorType, String> {
    validate_decimal_precision_and_scale::<T>(p, s).map_err(|e| format!("{dt}: {e}"))?;
    Ok(GeneratorType::Prim(dt.clone()))
}

/// The type of a child field, holding no nulls when the field is not
/// nullable
fn from_field(f: &Field) -> Result<GeneratorType, String> {
    let t = convert(f.data_type())?;
    if f.is_nullable() {
        Ok(t)
    } else {
        without_nulls(t)
    }
}

/// `t` with no null rows. A union or run-end encoded row is null when its
/// child's value is, so their children hold no nulls either
fn without_nulls(t: GeneratorType) -> Result<GeneratorType, String> {
    let inner = match t {
        GeneratorType::Null => return Err("a non-nullable field of type Null".to_string()),
        GeneratorType::NotNull(_) => return Ok(t),
        GeneratorType::Union(cs, mode, ids) => GeneratorType::Union(
            cs.into_iter()
                .map(without_nulls)
                .collect::<Result<_, _>>()?,
            mode,
            ids,
        ),
        GeneratorType::Ree(r, v) => GeneratorType::Ree(r, Box::new(without_nulls(*v)?)),
        t => t,
    };
    Ok(GeneratorType::NotNull(Box::new(inner)))
}

/// `a` with type `dt`, which `Exact` guarantees differs from `a`'s type only
/// in child field names, nullability and metadata
fn retype(a: &ArrayRef, dt: &DataType) -> ArrayRef {
    let d = retype_data(a.to_data(), dt);
    debug_assert!(d.validate_full().is_ok(), "{:?}", d.validate_full());
    make_array(d)
}

fn retype_data(d: ArrayData, dt: &DataType) -> ArrayData {
    let child_types: Vec<&DataType> = match dt {
        DataType::List(f)
        | DataType::LargeList(f)
        | DataType::ListView(f)
        | DataType::LargeListView(f)
        | DataType::FixedSizeList(f, _)
        | DataType::Map(f, _) => vec![f.data_type()],
        DataType::Struct(fs) => fs.iter().map(|f| f.data_type()).collect(),
        DataType::Union(fs, _) => fs.iter().map(|(_, f)| f.data_type()).collect(),
        DataType::Dictionary(_, v) => vec![v.as_ref()],
        DataType::RunEndEncoded(r, v) => vec![r.data_type(), v.data_type()],
        _ => vec![],
    };
    debug_assert_eq!(child_types.len(), d.child_data().len());
    let children: Vec<ArrayData> = d
        .child_data()
        .iter()
        .zip(child_types)
        .map(|(c, t)| retype_data(c.clone(), t))
        .collect();
    let builder = d.into_builder().data_type(dt.clone()).child_data(children);
    // SAFETY: the buffers, offsets, lengths and nulls are unchanged, and `dt`
    // differs from the old type only in child field names, nullability and
    // metadata, of which only nullability affects validity. arrow-rs's safe
    // constructors checked the nulls when the generator built `d`, against
    // fields that forbid nulls wherever `dt`'s do, because `from_field` makes
    // each non-nullable field's type `NotNull`. `retype` validates the result
    // in debug builds
    unsafe { builder.build_unchecked() }
}

/// A map's entries: a key that may not be null and a nullable value
/// (Schema.fbs lines 118-138)
fn map_entries(k: &GeneratorType, v: &GeneratorType) -> DataType {
    DataType::Struct(Fields::from(vec![
        Field::new("key", k.data_type(), false),
        field("value", v),
    ]))
}

/// Key types for maps: leaves the row format and comparisons can order
fn map_key() -> BoxedStrategy<GeneratorType> {
    prop_oneof![
        Just(GeneratorType::I32),
        Just(GeneratorType::Utf8),
        Just(GeneratorType::Prim(DataType::Int8)),
        Just(GeneratorType::Binary)
    ]
    .boxed()
}

fn struct_fields(fs: &[GeneratorType]) -> Fields {
    fs.iter()
        .enumerate()
        .map(|(i, f)| field(&format!("f{i}"), f))
        .collect()
}

fn union_fields(cs: &[GeneratorType], ids: &[i8]) -> UnionFields {
    UnionFields::try_new(
        ids.iter().copied(),
        cs.iter()
            .enumerate()
            .map(|(i, c)| field(&format!("u{i}"), c)),
    )
    .unwrap()
}

// ---------------------------------------------------------------- types

pub(crate) fn leaf() -> BoxedStrategy<GeneratorType> {
    prop_oneof![
        1 => Just(GeneratorType::Null),
        3 => Just(GeneratorType::Bool),
        3 => Just(GeneratorType::I32),
        2 => Just(GeneratorType::I64),
        3 => Just(GeneratorType::Utf8),
        2 => Just(GeneratorType::LargeUtf8),
        3 => Just(GeneratorType::Utf8View),
        2 => Just(GeneratorType::Binary),
        1 => Just(GeneratorType::LargeBinary),
        2 => Just(GeneratorType::BinaryView),
        // Width 0 is a degenerate shape with its own bugs
        2 => prop_oneof![2 => Just(0), 3 => 1..=4i32].prop_map(GeneratorType::Fsb),
        6 => select(value_types()).prop_map(GeneratorType::Prim),
    ]
    .boxed()
}

/// The fixed-width value types, with the precisions, scales, units and time
/// zones that have their own edge cases
pub(crate) fn value_types() -> Vec<DataType> {
    use arrow_schema::TimeUnit::*;
    let tz = |s: &str| Some(Arc::from(s));
    vec![
        DataType::Int8,
        DataType::Int16,
        DataType::UInt8,
        DataType::UInt16,
        DataType::UInt32,
        DataType::UInt64,
        DataType::Float32,
        DataType::Float64,
        DataType::Decimal32(9, 2),
        DataType::Decimal64(18, 0),
        DataType::Decimal128(1, 0),
        DataType::Decimal128(10, 2),
        DataType::Decimal128(38, 10),
        DataType::Decimal128(38, 38),
        DataType::Decimal128(5, -2),
        DataType::Decimal256(39, 0),
        DataType::Decimal256(76, 18),
        DataType::Date32,
        DataType::Date64,
        DataType::Time32(Second),
        DataType::Time32(Millisecond),
        DataType::Time64(Microsecond),
        DataType::Time64(Nanosecond),
        DataType::Timestamp(Second, None),
        DataType::Timestamp(Millisecond, tz("+05:30")),
        DataType::Timestamp(Microsecond, tz("UTC")),
        DataType::Timestamp(Nanosecond, None),
        DataType::Duration(Second),
        DataType::Duration(Nanosecond),
        DataType::Interval(IntervalUnit::YearMonth),
        DataType::Interval(IntervalUnit::DayTime),
        DataType::Interval(IntervalUnit::MonthDayNano),
    ]
}

/// `v` as `w` little-endian bytes, sign-extended
fn le(v: i128, w: usize) -> Vec<u8> {
    let mut b = v.to_le_bytes().to_vec();
    b.resize(w, if v < 0 { 0xFF } else { 0 });
    b.truncate(w);
    b
}

/// Signed values at the limits of every narrower width, which casts and
/// narrowing arithmetic get wrong
fn signed_edges(bits: u32) -> Vec<i128> {
    let mut out = vec![0, 1, -1, 2, -2];
    for b in [8u32, 16, 32, 64].into_iter().filter(|&b| b <= bits) {
        let max = (1i128 << (b - 1)) - 1;
        out.extend([max, -max - 1, max - 1, -max]);
    }
    out
}

/// The value of 10^k as an i128, for decimal precisions up to 38
fn pow10(k: u32) -> i128 {
    10i128.pow(k)
}

/// Bytes of one valid, non-null value of type `dt`
pub(crate) fn prim_value(dt: &DataType) -> BoxedStrategy<Vec<u8>> {
    use arrow_schema::TimeUnit::*;
    let w = dt.primitive_width().unwrap();
    let pick = |vals: Vec<i128>| -> BoxedStrategy<Vec<u8>> {
        prop_oneof![3 => select(vals).prop_map(move |v| le(v, w)), 2 => vec(any::<u8>(), w)].boxed()
    };
    let ranged = move |vals: Vec<i128>, lo: i128, hi: i128, step: i128| -> BoxedStrategy<Vec<u8>> {
        let vals: Vec<i128> = vals.into_iter().filter(|v| (lo..hi).contains(v)).collect();
        let span = (hi - lo) / step;
        prop_oneof![
            3 => select(vals).prop_map(move |v| le(v, w)),
            2 => (0..span).prop_map(move |k| le(lo + k * step, w)),
        ]
        .boxed()
    };
    match dt {
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
            pick(signed_edges(w as u32 * 8))
        }
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
            let bits = w as u32 * 8;
            let mut v = vec![0i128, 1, 2];
            for b in [8u32, 16, 32, 64].into_iter().filter(|&b| b <= bits) {
                v.extend([(1i128 << b) - 1, (1i128 << (b - 1)), (1i128 << (b - 1)) - 1]);
            }
            pick(v)
        }
        DataType::Float32 => {
            let specials: Vec<u32> = [
                0.0f32,
                -0.0,
                1.0,
                -1.0,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::MIN,
                f32::MAX,
                f32::MIN_POSITIVE,
                f32::EPSILON,
            ]
            .iter()
            .map(|f| f.to_bits())
            .chain([f32::NAN.to_bits(), 0x7FC0_0001, 0xFFC0_0000, 1, 0x8000_0001])
            .collect();
            prop_oneof![3 => select(specials), 2 => any::<u32>()]
                .prop_map(|b| b.to_le_bytes().to_vec())
                .boxed()
        }
        DataType::Float64 => {
            let specials: Vec<u64> = [
                0.0f64,
                -0.0,
                1.0,
                -1.0,
                0.1,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::MIN,
                f64::MAX,
                f64::MIN_POSITIVE,
                f64::EPSILON,
                9_007_199_254_740_992.0,
            ]
            .iter()
            .map(|f| f.to_bits())
            .chain([
                f64::NAN.to_bits(),
                0x7FF8_0000_0000_0001,
                0xFFF8_0000_0000_0000,
                1,
                0x8000_0000_0000_0001,
            ])
            .collect();
            prop_oneof![3 => select(specials), 2 => any::<u64>()]
                .prop_map(|b| b.to_le_bytes().to_vec())
                .boxed()
        }
        DataType::Decimal32(p, s) | DataType::Decimal64(p, s) | DataType::Decimal128(p, s) => {
            let (p, s) = (*p as u32, *s);
            let max = pow10(p) - 1;
            let mut v = vec![0, 1, -1, max, -max, pow10(p - 1), -pow10(p - 1)];
            if s > 0 && (s as u32) <= p {
                let one = pow10(s as u32);
                // One unit, and values at the rounding midpoint of the last digit
                v.extend([
                    one,
                    -one,
                    one - 1,
                    one / 2,
                    -one / 2,
                    one / 2 + 1,
                    5,
                    -5,
                    15,
                    -15,
                ]);
            }
            let in_range = move |x: &i128| -max <= *x && *x <= max;
            let ok: Vec<i128> = v.into_iter().filter(in_range).collect();
            // Values past the declared precision, which the spec does not
            // rule out and arrow-rs does not reject (apache/arrow-rs#10947)
            let past: Vec<i128> = vec![max + 1, -max - 1];
            prop_oneof![
                6 => select(ok).prop_map(move |x| le(x, w)),
                3 => (-max..=max).prop_map(move |x| le(x, w)),
                1 => select(past).prop_map(move |x| le(x, w)),
            ]
            .boxed()
        }
        DataType::Decimal256(p, _) => {
            use arrow_buffer::i256;
            let ten = i256::from_i128(10);
            let mut m = i256::ONE;
            for _ in 0..*p {
                m = m.wrapping_mul(ten);
            }
            let max = m.wrapping_sub(i256::ONE);
            let edges = vec![
                i256::ZERO,
                i256::ONE,
                i256::MINUS_ONE,
                max,
                max.wrapping_neg(),
                i256::from_i128(i128::MAX),
                i256::from_i128(i128::MIN),
            ];
            let t = GeneratorType::Prim(dt.clone());
            prop_oneof![
                3 => select(edges).prop_map(|x| x.to_le_bytes().to_vec()),
                2 => any::<i128>().prop_map(|x| i256::from_i128(x).to_le_bytes().to_vec()),
                // Magnitudes past i128, which i256 division handles apart
                2 => spread(domain(&t).unwrap()).prop_map(move |v| match int_val(&t, &v) {
                    Val::Raw(b) => b,
                    _ => unreachable!(),
                }),
            ]
            .boxed()
        }
        DataType::Date32 => {
            // 106751 days is where nanosecond timestamps overflow
            prop_oneof![
                3 => pick(vec![0, 1, -1, 106751, -106752, i32::MAX as i128, i32::MIN as i128, 2932896]),
                2 => calendar_day().prop_map(|d| le(d, 4)),
            ]
            .boxed()
        }
        // Whole days only (Schema.fbs lines 249-250)
        DataType::Date64 => {
            let day = 86_400_000i128;
            let days = vec![0i128, 1, -1, 106751, -106752, 2932896, 100_000_000];
            prop_oneof![
                3 => select(days).prop_map(move |d| le(d * day, 8)),
                2 => (-10_000_000i128..10_000_000).prop_map(move |d| le(d * day, 8)),
                2 => calendar_day().prop_map(move |d| le(d * day, 8)),
            ]
            .boxed()
        }
        // Within one day (Schema.fbs lines 266-268)
        DataType::Time32(u) | DataType::Time64(u) => {
            let per_day: i128 = match u {
                Second => 86_400,
                Millisecond => 86_400_000,
                Microsecond => 86_400_000_000,
                Nanosecond => 86_400_000_000_000,
            };
            ranged(
                vec![0, 1, per_day - 1, per_day / 2],
                0,
                per_day,
                1.max(per_day / 1_000_000),
            )
        }
        DataType::Timestamp(u, _) | DataType::Duration(u) => {
            let mut v = signed_edges(64);
            // Values where converting to a finer unit overflows i64
            let per_ns: i128 = match u {
                Second => 1_000_000_000,
                Millisecond => 1_000_000,
                Microsecond => 1_000,
                Nanosecond => 1,
            };
            let limit = i64::MAX as i128 / per_ns;
            v.extend([limit, limit + 1, -limit, -limit - 1, 86_399, 86_400]);
            let per_day = 86_400 * 1_000_000_000 / per_ns;
            let time_of_day = prop_oneof![select(vec![0, 1, per_day - 1, per_day / 2]), 0..per_day];
            prop_oneof![
                3 => pick(v.into_iter().filter(|x| i64::try_from(*x).is_ok()).collect()),
                2 => (calendar_day(), time_of_day).prop_map(move |(d, t)| le(d * per_day + t, 8)),
            ]
            .boxed()
        }
        // Every bit pattern is valid, and the fields are independent
        // (Schema.fbs lines 402-419)
        DataType::Interval(_) => {
            let i32s = signed_edges(32);
            // Small fields as well as extremes, so most rows do not overflow
            let field32 = move || prop_oneof![3 => select(i32s.clone()), 3 => -100i128..100, 1 => any::<i32>().prop_map(|x| x as i128)];
            let field64 = prop_oneof![
                3 => select(signed_edges(64)),
                3 => -1_000_000_000_000i128..1_000_000_000_000,
                1 => any::<i64>().prop_map(|x| x as i128),
            ];
            match w {
                4 => field32().prop_map(|m| le(m, 4)).boxed(),
                8 => (field32(), field32())
                    .prop_map(|(d, ms)| [le(d, 4), le(ms, 4)].concat())
                    .boxed(),
                _ => (field32(), field32(), field64)
                    .prop_map(|(m, d, ns)| [le(m, 4), le(d, 4), le(ns, 8)].concat())
                    .boxed(),
            }
        }
        dt => panic!("no values for {dt}"),
    }
}

/// Days since 1970 where calendar arithmetic differs: month ends, leap
/// days and the days on either side, in leap, common and century years, and
/// ordinary days within a few centuries of 1970
fn calendar_day() -> BoxedStrategy<i128> {
    use crate::temporal::days_from_civil;
    let mut special = vec![];
    for y in [
        1, 1582, 1900, 1969, 1970, 1999, 2000, 2004, 2023, 2024, 2100, 9999,
    ] {
        for (m, d) in [
            (1, 1),
            (1, 31),
            (2, 28),
            (2, 29),
            (3, 1),
            (3, 31),
            (4, 30),
            (12, 31),
        ] {
            if m != 2 || d != 29 || (y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)) {
                let day = days_from_civil(y, m, d);
                // The neighbors fall on a month end in a time zone a day
                // ahead or behind
                special.extend([day - 1, day, day + 1]);
            }
        }
    }
    prop_oneof![3 => select(special), 1 => -150_000i128..150_000].boxed()
}

fn dict_value() -> BoxedStrategy<GeneratorType> {
    prop_oneof![
        Just(GeneratorType::Utf8),
        Just(GeneratorType::I32),
        Just(GeneratorType::Utf8View),
        Just(GeneratorType::Binary)
    ]
    .boxed()
}

/// Every integer type a dictionary key may have
const KEY_TYPES: [DataType; 8] = [
    DataType::Int8,
    DataType::Int16,
    DataType::Int32,
    DataType::Int64,
    DataType::UInt8,
    DataType::UInt16,
    DataType::UInt32,
    DataType::UInt64,
];

pub(crate) fn ty(depth: u32) -> BoxedStrategy<GeneratorType> {
    if depth >= 2 {
        return leaf();
    }
    let sub = move || child(ty(depth + 1));
    prop_oneof![
        10 => leaf(),
        2 => sub().prop_map(|c| GeneratorType::List(Box::new(c))),
        1 => sub().prop_map(|c| GeneratorType::LargeList(Box::new(c))),
        2 => sub().prop_map(|c| GeneratorType::ListView(Box::new(c))),
        1 => sub().prop_map(|c| GeneratorType::LargeListView(Box::new(c))),
        2 => (sub(), prop_oneof![1 => Just(0), 2 => 1..=3i32]).prop_map(|(c, n)| GeneratorType::Fsl(Box::new(c), n)),
        // Zero fields is a degenerate shape with its own bugs
        3 => vec(sub(), 0..=3).prop_map(GeneratorType::Struct),
        2 => (map_key(), sub(), any::<bool>()).prop_map(|(k, v, sorted)| GeneratorType::Map(Box::new(k), Box::new(v), sorted)),
        2 => (select(KEY_TYPES.to_vec()), dict_value()).prop_map(|(k, v)| GeneratorType::Dict(k, Box::new(v))),
        2 => (select(vec![DataType::Int16, DataType::Int32, DataType::Int64]), child(prop_oneof![Just(GeneratorType::I32), Just(GeneratorType::Utf8)].boxed()))
            .prop_map(|(r, v)| GeneratorType::Ree(r, Box::new(v))),
        2 => (vec(child(leaf()), 1..=3), prop_oneof![Just(UnionMode::Sparse), Just(UnionMode::Dense)], any::<bool>())
            .prop_map(|(cs, mode, odd_ids)| {
                // Non-default type ids, not 0..n, in about half of the unions
                let ids = (0..cs.len() as i8).map(|i| if odd_ids { 5 + 3 * i } else { i }).collect();
                GeneratorType::Union(cs, mode, ids)
            }),
    ]
    .boxed()
}

// ---------------------------------------------------------------- numeric batches

/// How the values of one numeric array are drawn. Optimized numeric kernels
/// branch on properties of a whole batch, such as every value fitting a
/// narrower type or one row overflowing, which drawing each value on its own
/// rarely produces
#[derive(Clone, Copy, Debug)]
enum Regime {
    /// Each value on its own, from the type's edges and random bit patterns
    Mixed,
    /// Every value fits in an integer of this many bits
    Narrow(u32),
    /// As `Narrow`, except one row holds a value just past it
    PastNarrow(u32),
    /// Values within one of a power of two or of ten, and the type's limits,
    /// which divisor and rescaling fast paths special-case
    NearPowers,
    /// Small values, except one row holds a limit of the type, zero or -1
    /// (for floats, a NaN, an infinity or -0.0)
    Outlier,
}

/// The integers a type holds, from `lo` to `hi`. Decimals are bounded by
/// their precision
#[derive(Clone, Debug)]
pub(crate) struct Domain {
    pub lo: BigInt,
    pub hi: BigInt,
}

pub(crate) fn domain_of(dt: &DataType) -> Option<Domain> {
    let one = BigInt::from(1);
    match dt {
        DataType::Decimal32(p, _)
        | DataType::Decimal64(p, _)
        | DataType::Decimal128(p, _)
        | DataType::Decimal256(p, _) => {
            let max = pow(BigInt::from(10), *p as usize) - 1;
            Some(Domain { lo: -&max, hi: max })
        }
        d if d.is_signed_integer() => {
            let b = 8 * d.primitive_width()? - 1;
            Some(Domain {
                lo: -(&one << b),
                hi: (&one << b) - 1,
            })
        }
        d if d.is_unsigned_integer() => Some(Domain {
            lo: BigInt::zero(),
            hi: (&one << (8 * d.primitive_width()?)) - 1,
        }),
        _ => None,
    }
}

fn domain(t: &GeneratorType) -> Option<Domain> {
    match t {
        GeneratorType::I32 => domain_of(&DataType::Int32),
        GeneratorType::I64 => domain_of(&DataType::Int64),
        GeneratorType::Prim(d) => domain_of(d),
        _ => None,
    }
}

fn is_float(t: &GeneratorType) -> bool {
    matches!(
        t,
        GeneratorType::Prim(DataType::Float32 | DataType::Float64)
    )
}

/// The values of `d` that fit in an integer of `b` bits, unsigned when `d`
/// has no negative values
pub(crate) fn narrow(d: &Domain, b: u32) -> Domain {
    let one = BigInt::from(1);
    if d.lo.is_zero() {
        Domain {
            lo: d.lo.clone(),
            hi: ((&one << b) - &one).min(d.hi.clone()),
        }
    } else {
        Domain {
            lo: (-(&one << (b - 1))).max(d.lo.clone()),
            hi: ((&one << (b - 1)) - &one).min(d.hi.clone()),
        }
    }
}

/// The integer widths narrower than `d`
pub(crate) fn narrower(d: &Domain) -> Vec<u32> {
    [8, 16, 32, 64, 128]
        .into_iter()
        .filter(|&b| narrow(d, b).hi < d.hi)
        .collect()
}

/// A value of `d` whose bit length is uniform, so small and large
/// magnitudes are equally likely
fn spread(d: Domain) -> BoxedStrategy<BigInt> {
    let bits = d.hi.bits().max((-&d.lo).bits());
    (
        0..=bits,
        vec(any::<u8>(), bits as usize / 8 + 1),
        any::<bool>(),
    )
        .prop_map(move |(k, bytes, neg)| {
            let one = BigInt::from(1);
            let m = BigInt::from_bytes_le(Sign::Plus, &bytes) % (&one << k);
            if neg && d.lo.is_negative() {
                -(m % (-&d.lo + &one))
            } else {
                m % (&d.hi + &one)
            }
        })
        .boxed()
}

/// Values of `d`, with its limits more often than `spread` gives them
fn within(d: Domain) -> BoxedStrategy<BigInt> {
    prop_oneof![1 => select(vec![d.lo.clone(), d.hi.clone()]), 5 => spread(d)].boxed()
}

/// Zero, one, minus one, the limits of `d`, and the values within one of a
/// power of two or of ten
fn near_powers(d: &Domain) -> Vec<BigInt> {
    let one = BigInt::from(1);
    let mut v = vec![
        BigInt::zero(),
        one.clone(),
        -&one,
        d.lo.clone(),
        d.hi.clone(),
        &d.lo + 1,
        &d.hi - 1,
    ];
    let twos = (1..=d.hi.bits()).map(|k| &one << k);
    let tens = successors(Some(BigInt::from(10)), |p| (p <= &d.hi).then(|| p * 10));
    for p in twos.chain(tens).collect::<Vec<_>>() {
        for x in [&p - 1, p.clone(), &p + 1] {
            v.push(-&x);
            v.push(x);
        }
    }
    v.retain(|x| &d.lo <= x && x <= &d.hi);
    v.sort();
    v.dedup();
    v
}

/// The same for floats, with their special values
fn float_near_powers() -> Vec<f64> {
    let mut v = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MAX,
        f64::MIN,
        f64::MIN_POSITIVE,
    ];
    for k in 1..=64 {
        let p = 2f64.powi(k);
        v.extend([p - 1.0, p, p + 1.0, 1.0 - p, -p, -p - 1.0]);
    }
    for k in 1..=18 {
        let p = 10f64.powi(k);
        v.extend([p - 1.0, p, p + 1.0, 1.0 - p, -p, -p - 1.0]);
    }
    v
}

fn int_val(t: &GeneratorType, v: &BigInt) -> Val {
    match t {
        GeneratorType::I32 => Val::I32(v.to_i32().unwrap()),
        GeneratorType::I64 => Val::I64(v.to_i64().unwrap()),
        GeneratorType::Prim(dt) => {
            // Unsigned values past the signed range get a sign byte that the
            // resize drops
            let mut b = v.to_signed_bytes_le();
            b.resize(
                dt.primitive_width().unwrap(),
                if v.is_negative() { 0xFF } else { 0 },
            );
            Val::Raw(b)
        }
        t => panic!("not an integer type {t:?}"),
    }
}

fn float_val(t: &GeneratorType, x: f64) -> Val {
    match t {
        GeneratorType::Prim(DataType::Float32) => Val::Raw((x as f32).to_le_bytes().to_vec()),
        _ => Val::Raw(x.to_le_bytes().to_vec()),
    }
}

fn regime(t: &GeneratorType) -> BoxedStrategy<Regime> {
    let widths = if is_float(t) {
        vec![8, 16, 32]
    } else {
        domain(t).map(|d| narrower(&d)).unwrap_or_default()
    };
    if widths.is_empty() {
        return prop_oneof![4 => Just(Regime::Mixed), 2 => Just(Regime::NearPowers), 1 => Just(Regime::Outlier)].boxed();
    }
    prop_oneof![
        4 => Just(Regime::Mixed),
        2 => select(widths.clone()).prop_map(Regime::Narrow),
        1 => select(widths).prop_map(Regime::PastNarrow),
        2 => Just(Regime::NearPowers),
        1 => Just(Regime::Outlier),
    ]
    .boxed()
}

/// A non-null value of a batch in regime `r`
fn regime_value(t: &GeneratorType, r: Regime) -> BoxedStrategy<Val> {
    let t = t.clone();
    if is_float(&t) {
        let integral = |b: u32| -(1i64 << (b - 1))..(1i64 << (b - 1));
        return match r {
            Regime::Mixed => value(&t),
            Regime::Narrow(b) | Regime::PastNarrow(b) => integral(b)
                .prop_map(move |x| float_val(&t, x as f64))
                .boxed(),
            Regime::NearPowers => select(float_near_powers())
                .prop_map(move |x| float_val(&t, x))
                .boxed(),
            Regime::Outlier => integral(8)
                .prop_map(move |x| float_val(&t, x as f64))
                .boxed(),
        };
    }
    let d = domain(&t).unwrap();
    let s = match r {
        Regime::Mixed => return value(&t),
        Regime::Narrow(b) | Regime::PastNarrow(b) => within(narrow(&d, b)),
        Regime::NearPowers => select(near_powers(&d)).boxed(),
        Regime::Outlier => within(narrow(&d, 8)),
    };
    s.prop_map(move |v| int_val(&t, &v)).boxed()
}

/// The value of the one row that `PastNarrow` and `Outlier` batches set
/// apart
fn odd_value(t: &GeneratorType, r: Regime) -> Option<BoxedStrategy<Val>> {
    let t = t.clone();
    if is_float(&t) {
        let xs = match r {
            Regime::PastNarrow(b) => {
                vec![2f64.powi(b as i32 - 1), -(2f64.powi(b as i32 - 1)) - 1.0]
            }
            Regime::Outlier => vec![
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                -0.0,
                f64::MAX,
                f64::MIN,
            ],
            _ => return None,
        };
        return Some(select(xs).prop_map(move |x| float_val(&t, x)).boxed());
    }
    let d = domain(&t).unwrap();
    let mut xs = match r {
        Regime::PastNarrow(b) => {
            let n = narrow(&d, b);
            vec![&n.hi + 1, &n.lo - 1]
        }
        Regime::Outlier => vec![d.lo.clone(), d.hi.clone(), BigInt::zero(), BigInt::from(-1)],
        _ => return None,
    };
    xs.retain(|x| &d.lo <= x && x <= &d.hi);
    Some(select(xs).prop_map(move |v| int_val(&t, &v)).boxed())
}

/// A row where kernels change course: the first, the last, either side of
/// a 64-row chunk, or any
fn row_at(n: usize) -> BoxedStrategy<usize> {
    prop_oneof![
        Just(0),
        Just(n - 1),
        Just(63.min(n - 1)),
        Just(64.min(n - 1)),
        0..n
    ]
    .boxed()
}

/// `n` rows of an integer, decimal or float type, in a regime chosen for
/// the whole array
fn numeric_rows(t: &GeneratorType, n: usize, rate: BoxedStrategy<f64>) -> BoxedStrategy<Vec<Val>> {
    let t = t.clone();
    (rate, regime(&t))
        .prop_flat_map(move |(rate, r)| {
            let rows = vec((weighted(rate), regime_value(&t, r)), n).prop_map(|v| {
                v.into_iter()
                    .map(|(null, x)| if null { Val::Null } else { x })
                    .collect::<Vec<_>>()
            });
            match odd_value(&t, r) {
                Some(odd) if n > 0 => (rows, odd, row_at(n))
                    .prop_map(|(mut rows, v, i)| {
                        rows[i] = v;
                        rows
                    })
                    .boxed(),
                _ => rows.boxed(),
            }
        })
        .boxed()
}

// ---------------------------------------------------------------- sizes

/// Lengths: tiny ones, byte, word and vector-lane boundaries, a uniform
/// range, and occasional batch sizes
///
/// # Example
///
/// ```
/// use arrow_pbt::generator::len;
/// use proptest::strategy::{Strategy, ValueTree};
/// use proptest::test_runner::TestRunner;
/// let mut runner = TestRunner::deterministic();
/// let n = len().new_tree(&mut runner).unwrap().current();
/// assert!(n <= 2048);
/// ```
pub fn len() -> BoxedStrategy<usize> {
    prop_oneof![
        4 => 0..=8usize,
        3 => select(vec![7usize, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257]),
        4 => 0..=300usize,
        1 => select(vec![1023usize, 1024, 1025, 2048]),
    ]
    .boxed()
}

// ---------------------------------------------------------------- rows

pub(crate) fn text() -> BoxedStrategy<String> {
    // Lengths around 12 cross the view inline limit; multi-byte characters
    // make byte and character lengths differ; `%`, `_` and `\\` are LIKE
    // syntax, and a newline is what `.` in a regex skips by default
    let n = prop_oneof![0..=3usize, 11..=13usize, 0..=30usize];
    let free = n
        .prop_flat_map(|n| {
            vec(
                select(vec![
                    'a', 'b', 'z', ' ', 'é', '€', '𝄞', '%', '_', '\\', '\n',
                ]),
                n,
            )
        })
        .prop_map(|cs| cs.into_iter().collect::<String>());
    prop_oneof![3 => free, 1 => numeric_text()].boxed()
}

/// Text that looks like a number, or almost does: integers past every width,
/// decimals, exponents, signs, padding, and malformed forms (after Comet's
/// fuzz data generator)
pub(crate) fn numeric_text() -> BoxedStrategy<String> {
    let int = prop_oneof![
        select(vec![
            "0",
            "-0",
            "+7",
            "127",
            "-128",
            "255",
            "256",
            "32767",
            "-32769",
            "2147483647",
            "2147483648",
            "-2147483649",
            "9223372036854775807",
            "9223372036854775808",
            "-9223372036854775809",
            "18446744073709551615",
            "18446744073709551616",
            "000123",
            "99999999999999999999999999999999999999",
            "100000000000000000000000000000000000000"
        ])
        .prop_map(String::from),
        any::<i64>().prop_map(|v| v.to_string()),
        any::<u64>().prop_map(|v| v.to_string()),
    ];
    let dec = prop_oneof![
        select(vec![
            "1.005",
            "-1.005",
            "0.5",
            "-0.5",
            "2.5",
            ".5",
            "5.",
            "-.5",
            "1.5e2",
            "1E-3",
            "1e+3",
            "0e999999",
            "1e999999",
            "1e-999999",
            "123.456789",
            "-0.0",
            "1.2.3",
            "1e",
            "e5",
            ".",
            "-",
            "+",
            "",
            " ",
            "1 2",
            "0x10",
            "1_000",
            "inf",
            "-inf",
            "NaN",
            "nan",
            "Infinity"
        ])
        .prop_map(String::from),
        (any::<i32>(), 0..20u32).prop_map(|(m, k)| {
            let s = m.unsigned_abs().to_string();
            let k = (k as usize).min(s.len());
            format!(
                "{}{}.{}",
                if m < 0 { "-" } else { "" },
                &s[..s.len() - k],
                &s[s.len() - k..]
            )
        }),
        any::<f64>().prop_map(|f| f.to_string()),
        any::<f32>().prop_map(|f| format!("{f:e}")),
    ];
    let pad = select(vec!["", "", "", " ", "\t", "  "]);
    (prop_oneof![int, dec], pad.clone(), pad)
        .prop_map(|(v, a, b)| format!("{a}{v}{b}"))
        .boxed()
}

fn bytes(n: BoxedStrategy<usize>) -> BoxedStrategy<Vec<u8>> {
    n.prop_flat_map(|n| vec(prop_oneof![Just(0u8), Just(255u8), any::<u8>()], n))
        .boxed()
}

fn int32() -> BoxedStrategy<i32> {
    prop_oneof![
        select(vec![
            0,
            1,
            -1,
            i32::MIN,
            i32::MAX,
            i8::MAX as i32,
            i8::MIN as i32
        ]),
        -3..=3i32,
        any::<i32>(),
    ]
    .boxed()
}

fn nullable(s: BoxedStrategy<Val>) -> BoxedStrategy<Val> {
    prop_oneof![1 => Just(Val::Null), 4 => s].boxed()
}

/// Values for dictionaries and run-end encoding come from small pools so that
/// rows repeat, and Int8 keys never run out. Integer pools hold the type's
/// extremes so sums can overflow
fn pool(t: &GeneratorType) -> Vec<Val> {
    match t {
        GeneratorType::I32 => [-2, -1, 0, 1, 2, i32::MIN, i32::MAX].map(Val::I32).to_vec(),
        GeneratorType::I64 => [-2, -1, 0, 1, 2, i64::MIN, i64::MAX].map(Val::I64).to_vec(),
        GeneratorType::Utf8 | GeneratorType::Utf8View => ["", "a", "bb", "a long string value"]
            .map(|s| Val::Str(s.into()))
            .to_vec(),
        GeneratorType::Binary => vec![
            Val::Bytes(vec![]),
            Val::Bytes(vec![0]),
            Val::Bytes(vec![1, 2]),
        ],
        GeneratorType::NotNull(t) => pool(t),
        GeneratorType::Exact(e) => pool(&e.built_as),
        // Any other type: a sample of its values, the same every time so
        // that every array of the type draws from one pool
        t => {
            let mut runner = TestRunner::deterministic();
            let values = value(t);
            (0..8)
                .map(|_| {
                    // `value` strategies never reject, so a tree always exists
                    values.new_tree(&mut runner).unwrap().current()
                })
                .collect()
        }
    }
}

fn pooled(t: &GeneratorType) -> BoxedStrategy<Val> {
    select(pool(t)).boxed()
}

/// A value of type `t`, null one time in five except for unions, whose
/// nulls come from their children
pub(crate) fn val(t: &GeneratorType) -> BoxedStrategy<Val> {
    match t {
        GeneratorType::Exact(e) => val(&e.built_as),
        GeneratorType::Null => Just(Val::Null).boxed(),
        // A run-end encoded row is null when its value is
        GeneratorType::NotNull(_) | GeneratorType::Union(..) => value(t),
        GeneratorType::Ree(_, v) if matches!(**v, GeneratorType::NotNull(_)) => value(t),
        _ => nullable(value(t)),
    }
}

/// Long strings that recur, so view arrays can share their bytes
const LONG_STRINGS: [&str; 3] = [
    "a string longer than twelve bytes",
    "another long string value",
    "€€€ multi-byte long 𝄞",
];

/// A value of type `t` that is never null at its top level
pub(crate) fn value(t: &GeneratorType) -> BoxedStrategy<Val> {
    match t.clone() {
        GeneratorType::NotNull(t) => value(&t),
        GeneratorType::Exact(e) => value(&e.built_as),
        GeneratorType::Null => Just(Val::Null).boxed(),
        GeneratorType::Bool => any::<bool>().prop_map(Val::Bool).boxed(),
        GeneratorType::I32 => int32().prop_map(Val::I32).boxed(),
        GeneratorType::I64 => {
            prop_oneof![select(vec![0, i64::MIN, i64::MAX, 1 << 32]), any::<i64>()]
                .prop_map(Val::I64)
                .boxed()
        }
        GeneratorType::Prim(dt) => prim_value(&dt).prop_map(Val::Raw).boxed(),
        GeneratorType::Utf8 | GeneratorType::LargeUtf8 => text().prop_map(Val::Str).boxed(),
        GeneratorType::Utf8View => {
            prop_oneof![3 => text(), 1 => select(LONG_STRINGS.to_vec()).prop_map(String::from)]
                .prop_map(Val::Str)
                .boxed()
        }
        GeneratorType::Binary | GeneratorType::LargeBinary => {
            bytes((0..=30usize).boxed()).prop_map(Val::Bytes).boxed()
        }
        GeneratorType::BinaryView => prop_oneof![
            3 => bytes((0..=30usize).boxed()),
            1 => select(LONG_STRINGS.to_vec()).prop_map(|s| s.as_bytes().to_vec()),
        ]
        .prop_map(Val::Bytes)
        .boxed(),
        GeneratorType::Fsb(w) => bytes(Just(w as usize).boxed()).prop_map(Val::Bytes).boxed(),
        GeneratorType::List(c)
        | GeneratorType::LargeList(c)
        | GeneratorType::ListView(c)
        | GeneratorType::LargeListView(c) => vec(val(&c), 0..=4).prop_map(Val::List).boxed(),
        GeneratorType::Fsl(c, n) => vec(val(&c), n as usize).prop_map(Val::List).boxed(),
        // Unique keys, which the spec leaves to the application, sorted when
        // the type says so
        GeneratorType::Map(k, v, sorted) => {
            let kt = k.data_type();
            vec((value(&k), val(&v)), 0..=4)
                .prop_map(move |mut es| {
                    let mut seen: Vec<Val> = vec![];
                    es.retain(|(key, _)| {
                        !seen.contains(key) && {
                            seen.push(key.clone());
                            true
                        }
                    });
                    if sorted {
                        es.sort_by(|a, b| leaf_cmp(&kt, &a.0, &b.0));
                    }
                    Val::List(
                        es.into_iter()
                            .map(|(key, value)| Val::Struct(vec![key, value]))
                            .collect(),
                    )
                })
                .boxed()
        }
        GeneratorType::Struct(fs) => {
            let fields: Vec<_> = fs.iter().map(val).collect();
            fields.prop_map(Val::Struct).boxed()
        }
        GeneratorType::Dict(_, v) => pooled(&v),
        GeneratorType::Ree(_, v) => pooled(&v),
        GeneratorType::Union(cs, _, ids) => {
            let alts: Vec<_> = cs
                .iter()
                .zip(ids)
                .map(|(c, id)| {
                    val(c)
                        .prop_map(move |v| Val::Union(id, Box::new(v)))
                        .boxed()
                })
                .collect();
            proptest::strategy::Union::new(alts).boxed()
        }
    }
}

/// The share of top-level rows that are null, chosen per array: none, few,
/// some, half, or all (after DataFusion's `candidate_null_pcts`)
fn null_rate() -> BoxedStrategy<f64> {
    prop_oneof![6 => Just(0.0), 2 => Just(0.01), 6 => Just(0.2), 2 => Just(0.5), 1 => Just(1.0)]
        .boxed()
}

/// `n` rows. Run-end encoded columns get runs, so that runs are not all of
/// length 1. A `NotNull` type has no null rows
///
/// # Example
///
/// ```
/// use arrow_pbt::generator::{GeneratorType, build, rows, seed};
/// use arrow_array::Array;
/// use proptest::strategy::{Strategy, ValueTree};
/// use proptest::test_runner::TestRunner;
/// let mut runner = TestRunner::deterministic();
/// let r = rows(&GeneratorType::Utf8, 5).new_tree(&mut runner).unwrap().current();
/// let s = seed().new_tree(&mut runner).unwrap().current();
/// assert_eq!(build(&GeneratorType::Utf8, &r, s).len(), 5);
/// ```
pub fn rows(t: &GeneratorType, n: usize) -> BoxedStrategy<Vec<Val>> {
    match t {
        GeneratorType::NotNull(t) => rows_with(t, n, Just(0.0).boxed()),
        t => rows_with(t, n, null_rate()),
    }
}

/// `n` rows of `t`, with the share of null rows drawn from `rate`
fn rows_with(t: &GeneratorType, n: usize, rate: BoxedStrategy<f64>) -> BoxedStrategy<Vec<Val>> {
    match t {
        GeneratorType::Exact(e) => rows(&e.built_as, n),
        GeneratorType::Ree(_, v) => {
            let pad = if matches!(**v, GeneratorType::NotNull(_)) {
                default_val(v)
            } else {
                Val::Null
            };
            vec((val(t), 1..=8usize), 0..=n)
                .prop_map(move |runs| {
                    let mut out: Vec<Val> = runs
                        .into_iter()
                        .flat_map(|(v, k)| repeat_n(v, k))
                        .take(n)
                        .collect();
                    out.resize(n, pad.clone());
                    out
                })
                .boxed()
        }
        // Kernels branch on whether every row has one type id, so two in
        // seven unions take all their rows from one child, and one in seven
        // all but one, placed like the odd row of the numeric regimes
        GeneratorType::Union(cs, _, ids) => {
            let (cs, ids) = (cs.clone(), ids.clone());
            let one = (0..cs.len()).prop_flat_map(move |i| {
                let id = ids[i];
                vec(
                    val(&cs[i]).prop_map(move |v| Val::Union(id, Box::new(v))),
                    n,
                )
            });
            let all_but_one = if n > 0 {
                (one.clone(), val(t), row_at(n))
                    .prop_map(|(mut rows, v, i)| {
                        rows[i] = v;
                        rows
                    })
                    .boxed()
            } else {
                one.clone().boxed()
            };
            prop_oneof![4 => vec(val(t), n), 2 => one, 1 => all_but_one].boxed()
        }
        GeneratorType::Null => vec(val(t), n).boxed(),
        t if domain(t).is_some() || is_float(t) => numeric_rows(t, n, rate),
        // A view of at most 12 bytes holds its value inline, and a longer one
        // points into a data buffer (Columnar.rst lines 494-503). So a view
        // array mixes both, or holds only inline values and needs no data
        // buffer, or only long ones, which puts all its bytes in buffers
        GeneratorType::Utf8View | GeneratorType::BinaryView => prop_oneof![
            2 => nullable_rows(n, rate.clone(), value(t)),
            1 => nullable_rows(n, rate.clone(), value(t).prop_map(inline).boxed()),
            1 => nullable_rows(n, rate, value(t).prop_map(long).boxed()),
        ]
        .boxed(),
        _ => nullable_rows(n, rate, value(t)),
    }
}

/// `n` rows of `values`, each null at a rate drawn from `rate`
fn nullable_rows(
    n: usize,
    rate: BoxedStrategy<f64>,
    values: BoxedStrategy<Val>,
) -> BoxedStrategy<Vec<Val>> {
    rate.prop_flat_map(move |rate| vec((weighted(rate), values.clone()), n))
        .prop_map(|v| {
            v.into_iter()
                .map(|(null, x)| if null { Val::Null } else { x })
                .collect()
        })
        .boxed()
}

/// A string or bytes value too long for a view to hold inline: the value
/// itself, or the value extended past 12 bytes
fn long(v: Val) -> Val {
    match v {
        Val::Str(s) if s.len() <= 12 => Val::Str(s + " and then past twelve bytes"),
        Val::Bytes(mut b) if b.len() <= 12 => {
            b.extend_from_slice(&[0xC3; 13]);
            Val::Bytes(b)
        }
        v => v,
    }
}

/// The longest prefix of a string or bytes value that a view holds inline,
/// at most 12 bytes and whole characters
fn inline(v: Val) -> Val {
    match v {
        Val::Str(mut s) => {
            let mut end = s.len().min(12);
            while !s.is_char_boundary(end) {
                end -= 1;
            }
            s.truncate(end);
            Val::Str(s)
        }
        Val::Bytes(mut b) => {
            b.truncate(12);
            Val::Bytes(b)
        }
        v => v,
    }
}

/// The integer, decimal and float types
///
/// # Example
///
/// ```
/// use arrow_pbt::generator::numeric_types;
/// use proptest::strategy::{Strategy, ValueTree};
/// use proptest::test_runner::TestRunner;
/// let mut runner = TestRunner::deterministic();
/// let t = numeric_types().new_tree(&mut runner).unwrap().current();
/// assert!(t.data_type().is_numeric());
/// ```
pub fn numeric_types() -> BoxedStrategy<GeneratorType> {
    let types: Vec<GeneratorType> = value_types()
        .into_iter()
        .filter(|d| domain_of(d).is_some() || d.is_floating())
        .map(GeneratorType::Prim)
        .chain([GeneratorType::I32, GeneratorType::I64])
        .collect();
    select(types).boxed()
}

/// Every type this generator knows, nested to depth 2
///
/// # Example
///
/// ```
/// use arrow_pbt::generator::all_types;
/// use proptest::strategy::{Strategy, ValueTree};
/// use proptest::test_runner::TestRunner;
/// let mut runner = TestRunner::deterministic();
/// let t = all_types().new_tree(&mut runner).unwrap().current();
/// println!("{}", t.data_type());
/// ```
pub fn all_types() -> BoxedStrategy<GeneratorType> {
    ty(0)
}

/// A type from `types`, rows, and a layout seed. Kernels that accept only
/// some types pass a narrower `types`, so they never see inputs to discard
///
/// # Example
///
/// ```
/// use arrow_pbt::generator::{all_types, array_of};
/// use arrow_array::Array;
/// use proptest::strategy::{Strategy, ValueTree};
/// use proptest::test_runner::TestRunner;
/// let mut runner = TestRunner::deterministic();
/// let g = array_of(all_types()).new_tree(&mut runner).unwrap().current();
/// let a = g.build();
/// assert_eq!(a.data_type(), &g.ty.data_type());
/// assert_eq!(a.len(), g.rows.len());
/// assert!(arrow_pbt::spec::check(&a.to_data()).is_ok());
/// ```
pub fn array_of(types: BoxedStrategy<GeneratorType>) -> BoxedStrategy<GeneratedArray> {
    (types, len())
        .prop_flat_map(|(t, n)| array(&t, n, ValueDomain::Arrow))
        .boxed()
}

/// The values the generator draws for a type
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ValueDomain {
    /// Every value arrow-rs accepts, including decimals past their declared
    /// precision, which arrow-rs does not reject (apache/arrow-rs#10947)
    #[default]
    Arrow,
    /// Only values within the logical type: decimals within their declared
    /// precision, which Arrow C++'s full validation requires of every valid
    /// slot. Garbage under a primitive's own null slot may lie outside it
    Logical,
}

/// An array of type `ty` with `n` rows, whose values come from `domain`, in
/// its rows and in the valid slots its layout fills. A project that rejects
/// decimals past their precision, as DataFusion does, needs
/// [`ValueDomain::Logical`]
///
/// # Example
///
/// ```
/// use arrow_pbt::generator::{GeneratorType, ValueDomain, array};
/// use arrow_array::Array;
/// use arrow_array::cast::AsArray;
/// use arrow_array::types::Decimal32Type;
/// use arrow_schema::DataType;
/// use proptest::strategy::{Strategy, ValueTree};
/// use proptest::test_runner::TestRunner;
/// let mut runner = TestRunner::deterministic();
/// let t = GeneratorType::try_from(&DataType::Decimal32(3, 1)).unwrap();
/// for _ in 0..100 {
///     let a = array(&t, 20, ValueDomain::Logical)
///         .new_tree(&mut runner)
///         .unwrap()
///         .current()
///         .build();
///     assert!(a.as_primitive::<Decimal32Type>().validate_decimal_precision(3).is_ok());
/// }
/// ```
pub fn array(ty: &GeneratorType, n: usize, domain: ValueDomain) -> BoxedStrategy<GeneratedArray> {
    let ty = ty.clone();
    (rows(&ty, n), seed())
        .prop_map(move |(rows, seed)| GeneratedArray {
            rows: match domain {
                ValueDomain::Arrow => rows,
                ValueDomain::Logical => rows.into_iter().map(|v| fit_decimals(&ty, v)).collect(),
            },
            ty: ty.clone(),
            seed,
            domain,
        })
        .boxed()
}

/// An array as `array_of` generates it: a type, logical rows, the seed
/// that picks the physical layout, and the domain of the values the layout
/// adds
#[derive(Clone, Debug)]
pub struct GeneratedArray {
    pub ty: GeneratorType,
    pub rows: Vec<Val>,
    pub seed: u64,
    pub domain: ValueDomain,
}

impl GeneratedArray {
    /// The array, as [`build`] makes it, with the values the layout adds
    /// from `domain`
    pub fn build(&self) -> ArrayRef {
        build_node(
            &self.ty,
            &self.rows,
            &mut Layout::new(self.seed, self.domain),
        )
    }
}

/// Zero to four fields of any types, named from `c0`, each nullable three
/// times in four. A field whose type holds `Null` is always nullable, since
/// a `Null` array is all null
///
/// # Example
///
/// ```
/// use arrow_pbt::generator::{ValueDomain, batch, schemas};
/// use proptest::strategy::{Strategy, ValueTree};
/// use proptest::test_runner::TestRunner;
/// let mut runner = TestRunner::deterministic();
/// let s = schemas().new_tree(&mut runner).unwrap().current();
/// assert!(s.fields().len() <= 4);
/// assert!(batch(&s, 10, ValueDomain::Arrow).is_ok());
/// ```
pub fn schemas() -> BoxedStrategy<SchemaRef> {
    let nullable = prop_oneof![3 => Just(true), 1 => Just(false)];
    prop_oneof![1 => Just(0usize), 7 => 1..=4usize]
        .prop_flat_map(move |k| vec((all_types(), nullable.clone()), k))
        .prop_map(|cols| {
            let fields: Vec<Field> = cols
                .iter()
                .enumerate()
                .map(|(i, (t, nullable))| {
                    let nullable = *nullable || any_type(&t.data_type(), &DataType::is_null);
                    Field::new(format!("c{i}"), t.data_type(), nullable)
                })
                .collect();
            Arc::new(Schema::new(fields))
        })
        .boxed()
}

/// A batch of `schema` with `n` rows, each column with its own layout and
/// with values from `domain`
///
/// # Errors
///
/// When the generator cannot build the type of a field, such as `Float16`
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use arrow_pbt::generator::{ValueDomain, batch};
/// use arrow_schema::{DataType, Field, Schema};
/// use proptest::strategy::{Strategy, ValueTree};
/// use proptest::test_runner::TestRunner;
/// let s = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
/// let mut runner = TestRunner::deterministic();
/// let g = batch(&s, 5, ValueDomain::Arrow).unwrap().new_tree(&mut runner).unwrap().current();
/// let b = g.build();
/// assert_eq!(b.schema(), s);
/// assert_eq!(b.num_rows(), 5);
/// assert_eq!(b.column(0).null_count(), 0);
/// ```
pub fn batch(
    schema: &SchemaRef,
    n: usize,
    domain: ValueDomain,
) -> Result<BoxedStrategy<GeneratedBatch>, String> {
    Ok(batch_of(schema, &column_types(schema)?, n, domain))
}

/// Zero to four batches of `schema`, none one time in ten, each with its own
/// length as [`len`] draws it, which includes no rows
///
/// # Errors
///
/// When the generator cannot build the type of a field, such as `Float16`
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use arrow_pbt::generator::{ValueDomain, batches};
/// use arrow_schema::{DataType, Field, Schema};
/// use proptest::strategy::{Strategy, ValueTree};
/// use proptest::test_runner::TestRunner;
/// let s = Arc::new(Schema::new(vec![Field::new("a", DataType::Utf8, true)]));
/// let mut runner = TestRunner::deterministic();
/// let gs = batches(&s, ValueDomain::Arrow).unwrap().new_tree(&mut runner).unwrap().current();
/// assert!(gs.len() <= 4);
/// assert!(gs.iter().all(|g| g.build().schema() == s));
/// ```
pub fn batches(
    schema: &SchemaRef,
    domain: ValueDomain,
) -> Result<BoxedStrategy<Vec<GeneratedBatch>>, String> {
    let types = column_types(schema)?;
    let schema = schema.clone();
    let one = len()
        .prop_flat_map(move |n| batch_of(&schema, &types, n, domain))
        .boxed();
    Ok(prop_oneof![1 => Just(0usize), 9 => 1..=4usize]
        .prop_flat_map(move |k| vec(one.clone(), k))
        .boxed())
}

fn column_types(schema: &Schema) -> Result<Vec<GeneratorType>, String> {
    schema
        .fields()
        .iter()
        .map(|f| {
            GeneratorType::try_from(f.as_ref()).map_err(|e| format!("field {}: {e}", f.name()))
        })
        .collect()
}

fn batch_of(
    schema: &SchemaRef,
    types: &[GeneratorType],
    n: usize,
    domain: ValueDomain,
) -> BoxedStrategy<GeneratedBatch> {
    let schema = schema.clone();
    let columns: Vec<_> = types.iter().map(|t| array(t, n, domain)).collect();
    columns
        .prop_map(move |columns| GeneratedBatch {
            schema: schema.clone(),
            columns,
            num_rows: n,
        })
        .boxed()
}

/// A batch as [`batch`] generates it: its schema, an array for each field,
/// and its row count, which a batch with no columns needs
#[derive(Clone, Debug)]
pub struct GeneratedBatch {
    pub schema: SchemaRef,
    pub columns: Vec<GeneratedArray>,
    pub num_rows: usize,
}

impl GeneratedBatch {
    /// The batch, with each column as [`GeneratedArray::build`] makes it
    ///
    /// # Panics
    ///
    /// When the columns do not match the schema or the row count. A batch
    /// from [`batch`] or [`batches`] always matches
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use arrow_pbt::generator::{GeneratedBatch, ValueDomain, batch};
    /// use arrow_schema::Schema;
    /// use proptest::strategy::{Strategy, ValueTree};
    /// use proptest::test_runner::TestRunner;
    /// let s = Arc::new(Schema::empty());
    /// let mut runner = TestRunner::deterministic();
    /// let g: GeneratedBatch = batch(&s, 3, ValueDomain::Arrow).unwrap().new_tree(&mut runner).unwrap().current();
    /// assert_eq!(g.build().num_rows(), 3);
    /// ```
    pub fn build(&self) -> RecordBatch {
        let columns = self.columns.iter().map(GeneratedArray::build).collect();
        let options = RecordBatchOptions::new().with_row_count(Some(self.num_rows));
        RecordBatch::try_new_with_options(self.schema.clone(), columns, &options)
            .unwrap_or_else(|e| panic!("columns that do not match {}: {e}", self.schema))
    }
}

/// `v`, a row of `t`, with each decimal value past its declared precision
/// moved to the nearest limit. Map keys that become equal keep the first
fn fit_decimals(t: &GeneratorType, v: Val) -> Val {
    match (t, v) {
        (_, Val::Null) => Val::Null,
        (GeneratorType::NotNull(t), v) => fit_decimals(t, v),
        (GeneratorType::Exact(e), v) => fit_decimals(&e.built_as, v),
        (GeneratorType::Prim(dt), Val::Raw(b)) => Val::Raw(fit_precision(dt, b)),
        (
            GeneratorType::List(c)
            | GeneratorType::LargeList(c)
            | GeneratorType::ListView(c)
            | GeneratorType::LargeListView(c)
            | GeneratorType::Fsl(c, _),
            Val::List(xs),
        ) => Val::List(xs.into_iter().map(|x| fit_decimals(c, x)).collect()),
        (GeneratorType::Map(k, v, _), Val::List(es)) => {
            let mut keys: Vec<Val> = vec![];
            Val::List(
                es.into_iter()
                    .filter_map(|e| {
                        let Val::Struct(kv) = e else { return Some(e) };
                        let [key, value] = <[Val; 2]>::try_from(kv).ok()?;
                        let key = fit_decimals(k, key);
                        if keys.contains(&key) {
                            return None;
                        }
                        keys.push(key.clone());
                        Some(Val::Struct(vec![key, fit_decimals(v, value)]))
                    })
                    .collect(),
            )
        }
        (GeneratorType::Struct(fs), Val::Struct(xs)) => {
            Val::Struct(fs.iter().zip(xs).map(|(f, x)| fit_decimals(f, x)).collect())
        }
        (GeneratorType::Dict(_, v) | GeneratorType::Ree(_, v), x) => fit_decimals(v, x),
        (GeneratorType::Union(cs, _, ids), Val::Union(id, x)) => {
            let x = match ids.iter().position(|i| *i == id) {
                Some(i) => fit_decimals(&cs[i], *x),
                None => *x,
            };
            Val::Union(id, Box::new(x))
        }
        (_, v) => v,
    }
}

/// `b`, a value of the fixed-width type `dt`, moved to the nearest limit of
/// `dt`'s precision if `dt` is a decimal and `b` lies past it
fn fit_precision(dt: &DataType, b: Vec<u8>) -> Vec<u8> {
    let decimal = matches!(
        dt,
        DataType::Decimal32(..)
            | DataType::Decimal64(..)
            | DataType::Decimal128(..)
            | DataType::Decimal256(..)
    );
    let Some(d) = domain_of(dt).filter(|_| decimal) else {
        return b;
    };
    let v = BigInt::from_signed_bytes_le(&b).clamp(d.lo, d.hi);
    match int_val(&GeneratorType::Prim(dt.clone()), &v) {
        Val::Raw(b) => b,
        _ => unreachable!("int_val of a primitive type is raw"),
    }
}

/// A layout seed. Seed 0 is the plain layout, so shrinking moves toward it
///
/// # Example
///
/// ```
/// use arrow_pbt::generator::{GeneratorType, build, seed};
/// use arrow_pbt::val::{Val, vals};
/// use proptest::strategy::{Strategy, ValueTree};
/// use proptest::test_runner::TestRunner;
/// let mut runner = TestRunner::deterministic();
/// let s = seed().new_tree(&mut runner).unwrap().current();
/// let rows = [Val::I32(1), Val::Null];
/// // The layout changes with the seed, the values do not
/// assert_eq!(vals(build(&GeneratorType::I32, &rows, s).as_ref()), rows);
/// ```
pub fn seed() -> BoxedStrategy<u64> {
    prop_oneof![1 => Just(0u64), 4 => any::<u64>()].boxed()
}

// ---------------------------------------------------------------- masks and indices

/// Run-based, constant, or independent bits, with optional nulls
///
/// # Example
///
/// ```
/// use arrow_pbt::generator::mask;
/// use arrow_array::Array;
/// use proptest::strategy::{Strategy, ValueTree};
/// use proptest::test_runner::TestRunner;
/// let mut runner = TestRunner::deterministic();
/// let m = mask(10).new_tree(&mut runner).unwrap().current();
/// assert_eq!(m.build().len(), 10);
/// ```
pub fn mask(n: usize) -> BoxedStrategy<Mask> {
    // On short masks runs are at most half the mask, so it mixes values
    let short = (n / 2).clamp(1, 8);
    let run = prop_oneof![
        3 => 1..=short,
        2 => select(vec![7usize, 8, 9, 63, 64, 65, 127, 128]),
        1 => 1..=300usize,
    ];
    // Runs alternate, starting from a random value. Long masks can have many
    // runs, which kernels such as `zip` and `filter` pick a strategy by
    let runs = (any::<bool>(), vec(run, 0..=24.max(n / 2))).prop_map(move |(first, rs)| {
        let mut bits: Vec<bool> = rs
            .into_iter()
            .enumerate()
            .flat_map(|(i, k)| repeat_n(first ^ (i % 2 == 1), k))
            .take(n)
            .collect();
        let pad = !bits.last().copied().unwrap_or(first);
        bits.resize(n, pad);
        bits
    });
    // Two adjacent words keeping k and 64 - k bits, which packs exactly one
    // full output word across a word boundary
    let words = (1..64usize, any::<u64>()).prop_map(move |(k, noise)| {
        let mut bits: Vec<bool> = (0..n).map(|i| (noise >> (i % 64)) & 1 == 1).collect();
        if n >= 128 {
            for (i, b) in bits.iter_mut().take(128).enumerate() {
                *b = if i < 64 { i < k } else { i - 64 < 64 - k };
            }
        }
        bits
    });
    // A few scattered rows, at most one in 16 and at least one, where
    // kernels switch to index-based copying
    let sparse = vec(0..n.max(1), 1..=(n / 16).max(1)).prop_map(move |keep| {
        let mut bits = vec![false; n];
        for i in keep.into_iter().filter(|&i| i < n) {
            bits[i] = true;
        }
        bits
    });
    // Keeping exactly one row, or all but one, are the boundaries next to
    // keeping none or all
    let one = (0..n.max(1)).prop_map(move |i| (0..n).map(|j| j == i).collect::<Vec<_>>());
    let all_but_one = (0..n.max(1)).prop_map(move |i| (0..n).map(|j| j != i).collect::<Vec<_>>());
    let bits = prop_oneof![
        1 => Just(vec![true; n]),
        1 => Just(vec![false; n]),
        1 => one,
        1 => all_but_one,
        6 => runs,
        1 => words,
        2 => select(vec![0.05, 0.5, 0.9, 0.99])
            .prop_flat_map(move |p| vec(weighted(p), n)),
        2 => sparse,
    ];
    let nulls = prop_oneof![
        8 => Just(None),
        3 => vec(weighted(0.8), n).prop_map(Some),
        1 => Just(Some(vec![false; n])),
    ];
    (bits, nulls, seed())
        .prop_map(|(bits, nulls, seed)| {
            let m = match nulls {
                None => bits.into_iter().map(Some).collect(),
                Some(valid) => bits
                    .into_iter()
                    .zip(valid)
                    .map(|(b, v)| v.then_some(b))
                    .collect(),
            };
            Mask { bits: m, seed }
        })
        .boxed()
}

/// A filter mask as `mask` generates it: optional bits and the seed that
/// picks the physical layout
#[derive(Clone, Debug)]
pub struct Mask {
    pub bits: Vec<Option<bool>>,
    pub seed: u64,
}

impl Mask {
    /// The mask as a Boolean array
    ///
    /// # Example
    ///
    /// ```
    /// use arrow_pbt::generator::Mask;
    /// let m = Mask { bits: vec![Some(true), None], seed: 0 };
    /// assert_eq!(m.build().iter().collect::<Vec<_>>(), [Some(true), None]);
    /// ```
    pub fn build(&self) -> BooleanArray {
        let rows: Vec<Val> = self
            .bits
            .iter()
            .map(|b| b.map_or(Val::Null, Val::Bool))
            .collect();
        build(&GeneratorType::Bool, &rows, self.seed)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap()
            .clone()
    }
}

/// `count` indices into `n` rows: random, runs of consecutive or repeated
/// rows, or the identity. A null index can hide any value, including one out
/// of bounds
///
/// # Example
///
/// ```
/// use arrow_pbt::generator::indices;
/// use proptest::strategy::{Strategy, ValueTree};
/// use proptest::test_runner::TestRunner;
/// let mut runner = TestRunner::deterministic();
/// let i = indices(4).new_tree(&mut runner).unwrap().current();
/// // Valid indices are in bounds, a null may hide any value
/// assert!(i.build().iter().flatten().all(|i| (0..4).contains(&i)));
/// ```
pub fn indices(n: usize) -> BoxedStrategy<Indices> {
    let count = len();
    let idx = move || -> BoxedStrategy<i32> {
        if n == 0 {
            Just(0).boxed()
        } else {
            (0..n as i32).boxed()
        }
    };
    let values = count.prop_flat_map(move |c| {
        let runs = vec((idx(), 1..=8usize, any::<bool>()), 0..=c.max(1)).prop_map(move |rs| {
            let mut out: Vec<i32> = rs
                .into_iter()
                .flat_map(|(start, k, step)| {
                    (0..k as i32).map(move |j| {
                        if step && n > 0 {
                            (start + j) % n as i32
                        } else {
                            start
                        }
                    })
                })
                .take(c)
                .collect();
            out.resize(c, 0);
            out
        });
        prop_oneof![
            3 => vec(idx(), c),
            2 => runs,
            1 => Just((0..c).map(|i| if n == 0 { 0 } else { (i % n) as i32 }).collect()),
        ]
    });
    (values, any::<bool>(), seed())
        .prop_flat_map(move |(vs, nullable, seed)| {
            let c = vs.len();
            let valid: BoxedStrategy<Vec<bool>> = if n == 0 {
                Just(vec![false; c]).boxed()
            } else if nullable {
                vec(weighted(0.75), c).boxed()
            } else {
                Just(vec![true; c]).boxed()
            };
            (Just(vs), valid, Just(seed))
        })
        .prop_map(|(vs, valid, seed)| Indices {
            values: vs
                .into_iter()
                .zip(valid)
                .map(|(v, ok)| ok.then_some(v))
                .collect(),
            seed,
        })
        .boxed()
}

/// Take indices as `indices` generates them: optional values and the seed
/// that picks the physical layout
#[derive(Clone, Debug)]
pub struct Indices {
    pub values: Vec<Option<i32>>,
    pub seed: u64,
}

impl Indices {
    /// The indices as an `Int32` array
    ///
    /// # Example
    ///
    /// ```
    /// use arrow_pbt::generator::Indices;
    /// let i = Indices { values: vec![Some(3), None], seed: 0 };
    /// assert_eq!(i.build().iter().collect::<Vec<_>>(), [Some(3), None]);
    /// ```
    pub fn build(&self) -> Int32Array {
        let rows: Vec<Val> = self
            .values
            .iter()
            .map(|x| x.map_or(Val::Null, Val::I32))
            .collect();
        build(&GeneratorType::I32, &rows, self.seed)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .clone()
    }
}

// ---------------------------------------------------------------- building

struct Layout {
    rng: Option<SmallRng>,
    /// Where buffers sit in memory, a stream apart from `rng` so that it
    /// does not shift the other layout choices
    placement: Option<SmallRng>,
    /// The values garbage in valid slots comes from
    domain: ValueDomain,
}

impl Layout {
    fn new(seed: u64, domain: ValueDomain) -> Self {
        Layout {
            rng: (seed != 0).then(|| SmallRng::seed_from_u64(seed)),
            placement: (seed != 0).then(|| SmallRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15)),
            domain,
        }
    }
    fn chance(&mut self, p: f64) -> bool {
        self.rng.as_mut().is_some_and(|r| r.random_bool(p))
    }
    /// Whether the next buffer sits at exactly the alignment arrow-rs requires
    fn at_minimum_alignment(&mut self) -> bool {
        self.placement.as_mut().is_some_and(|r| r.random_bool(0.3))
    }
    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        match self.rng.as_mut() {
            Some(r) => xs[r.random_range(0..xs.len())],
            None => xs[0],
        }
    }
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        match self.rng.as_mut() {
            Some(r) => r.random_range(lo..=hi),
            None => lo,
        }
    }
    /// A random non-null value of the layout's domain, used for padding rows
    /// and data under nulls
    fn garbage(&mut self, t: &GeneratorType) -> Val {
        self.garbage_in(t, self.domain)
    }

    /// The same, with a primitive value from `domain`. Nested values come
    /// from the layout's domain, since they fill valid slots of the children
    fn garbage_in(&mut self, t: &GeneratorType, domain: ValueDomain) -> Val {
        if let GeneratorType::NotNull(t) = t {
            return self.garbage_in(t, domain);
        }
        if let GeneratorType::Exact(e) = t {
            return self.garbage_in(&e.built_as, domain);
        }
        let Some(r) = self.rng.as_mut() else {
            return default_val(t);
        };
        match t {
            GeneratorType::NotNull(_) | GeneratorType::Exact(_) => unreachable!("unwrapped above"),
            GeneratorType::Null => Val::Null,
            GeneratorType::Bool => Val::Bool(r.random()),
            GeneratorType::I32 => Val::I32(r.random()),
            GeneratorType::I64 => Val::I64(r.random()),
            // Garbage can land in slots a child array considers valid (under a
            // null struct row, in an unselected sparse union child), so it
            // must satisfy the type's value rules
            GeneratorType::Prim(dt) => {
                use arrow_schema::TimeUnit::*;
                let w = dt.primitive_width().unwrap();
                let v: Option<i128> = match dt {
                    DataType::Date64 => {
                        Some(r.random_range(-1_000_000i128..1_000_000) * 86_400_000)
                    }
                    DataType::Time32(u) | DataType::Time64(u) => {
                        let per_day: i128 = match u {
                            Second => 86_400,
                            Millisecond => 86_400_000,
                            Microsecond => 86_400_000_000,
                            Nanosecond => 86_400_000_000_000,
                        };
                        Some(r.random_range(0..per_day))
                    }
                    _ => None,
                };
                let b = match v {
                    Some(v) => le(v, w),
                    None => (0..w).map(|_| r.random()).collect(),
                };
                Val::Raw(match domain {
                    ValueDomain::Arrow => b,
                    ValueDomain::Logical => fit_precision(dt, b),
                })
            }
            GeneratorType::Utf8 | GeneratorType::LargeUtf8 | GeneratorType::Utf8View => {
                let k = r.random_range(0..=20);
                Val::Str(
                    (0..k)
                        .map(|_| ['x', 'y', 'é'][r.random_range(0..3)])
                        .collect(),
                )
            }
            GeneratorType::Binary | GeneratorType::LargeBinary | GeneratorType::BinaryView => {
                Val::Bytes((0..r.random_range(0..=20)).map(|_| r.random()).collect())
            }
            GeneratorType::Fsb(w) => Val::Bytes((0..*w).map(|_| r.random()).collect()),
            GeneratorType::List(c)
            | GeneratorType::LargeList(c)
            | GeneratorType::ListView(c)
            | GeneratorType::LargeListView(c) => {
                let k = self.range(0, 3);
                Val::List((0..k).map(|_| self.garbage(c)).collect())
            }
            GeneratorType::Fsl(c, n) => Val::List((0..*n).map(|_| self.garbage(c)).collect()),
            // One entry, so a null map row can cover a non-empty range
            GeneratorType::Map(k, v, _) => {
                Val::List(vec![Val::Struct(vec![self.garbage(k), self.garbage(v)])])
            }
            GeneratorType::Struct(fs) => Val::Struct(fs.iter().map(|f| self.garbage(f)).collect()),
            GeneratorType::Dict(_, v) | GeneratorType::Ree(_, v) => {
                let p = pool(v);
                let x = p[r.random_range(0..p.len())].clone();
                match self.domain {
                    ValueDomain::Arrow => x,
                    ValueDomain::Logical => fit_decimals(v, x),
                }
            }
            GeneratorType::Union(cs, _, ids) => {
                let i = self.range(0, cs.len() - 1);
                Val::Union(ids[i], Box::new(self.garbage(&cs[i])))
            }
        }
    }
}

/// A non-null value of type `t`, for support probes and the plain layout
pub(crate) fn default_val(t: &GeneratorType) -> Val {
    match t {
        GeneratorType::NotNull(t) => default_val(t),
        GeneratorType::Exact(e) => default_val(&e.built_as),
        GeneratorType::Null => Val::Null,
        GeneratorType::Bool => Val::Bool(true),
        GeneratorType::I32 => Val::I32(1),
        GeneratorType::I64 => Val::I64(1),
        GeneratorType::Prim(dt) => Val::Raw(vec![0; dt.primitive_width().unwrap()]),
        GeneratorType::Utf8 | GeneratorType::LargeUtf8 | GeneratorType::Utf8View => {
            Val::Str("a".into())
        }
        GeneratorType::Binary | GeneratorType::LargeBinary | GeneratorType::BinaryView => {
            Val::Bytes(vec![1])
        }
        GeneratorType::Fsb(w) => Val::Bytes(vec![1; *w as usize]),
        GeneratorType::List(c)
        | GeneratorType::LargeList(c)
        | GeneratorType::ListView(c)
        | GeneratorType::LargeListView(c) => Val::List(vec![default_val(c)]),
        GeneratorType::Fsl(c, n) => Val::List((0..*n).map(|_| default_val(c)).collect()),
        GeneratorType::Map(..) => Val::List(vec![]),
        GeneratorType::Struct(fs) => Val::Struct(fs.iter().map(default_val).collect()),
        GeneratorType::Dict(_, v) | GeneratorType::Ree(_, v) => default_val(v),
        GeneratorType::Union(cs, _, ids) => Val::Union(ids[0], Box::new(default_val(&cs[0]))),
    }
}

/// Builds `rows` with the physical layout chosen by `seed`
///
/// # Example
///
/// ```
/// use arrow_pbt::generator::{GeneratorType, build};
/// use arrow_pbt::val::{Val, vals};
/// let rows = [Val::Str("a".into()), Val::Null];
/// let a = build(&GeneratorType::Utf8, &rows, 0);
/// assert_eq!(vals(a.as_ref()), rows);
/// ```
pub fn build(t: &GeneratorType, rows: &[Val], seed: u64) -> ArrayRef {
    build_node(t, rows, &mut Layout::new(seed, ValueDomain::Arrow))
}

/// How many arrays `misalign_by_8` produced, for the coverage report
#[doc(hidden)]
pub static MISALIGNED: AtomicUsize = AtomicUsize::new(0);

/// `data` with its values buffer moved to an address that is 8-byte but not
/// 16-byte aligned, built without the alignment check a foreign producer never
/// ran
fn misalign_by_8(data: ArrayData) -> ArrayData {
    MISALIGNED.fetch_add(1, Ordering::Relaxed);
    let values = &data.buffers()[0];
    let shifted = placed(values.as_slice(), 8);
    debug_assert_eq!(shifted.as_ptr() as usize % 16, 8);
    debug_assert_eq!(shifted.as_slice(), values.as_slice());
    let builder = data.into_builder().buffers(vec![shifted]);
    // SAFETY: only the address of the values changed; the data is exported
    // through FFI, whose import is responsible for realigning it
    unsafe { builder.build_unchecked() }
}

/// How a padded array is cut down to its rows
#[derive(Clone, Copy)]
enum Slicing {
    /// `Array::slice`, which re-slices buffers and children
    Array,
    /// `ArrayData::slice`, which keeps the buffers and children and sets the
    /// data offset
    Data,
    /// `ArrayData::slice`, then an FFI export and import
    Ffi,
    /// The offset set on the parent and the buffers and children left
    /// whole, as pyarrow exports a sliced array through the C Data Interface
    Parent,
}

fn build_node(t: &GeneratorType, rows: &[Val], l: &mut Layout) -> ArrayRef {
    if let GeneratorType::NotNull(t) = t {
        return build_node(t, rows, l);
    }
    if let GeneratorType::Exact(e) = t {
        return retype(&build_node(&e.built_as, rows, l), &e.dt);
    }
    // Sometimes build a longer array and slice it, at a byte or word boundary
    if l.chance(0.3) {
        let pre = l.pick(&[1usize, 2, 3, 7, 8, 9, 63, 64, 65]);
        let post = l.pick(&[0usize, 1, 5, 64]);
        let mut all: Vec<Val> = (0..pre).map(|_| l.garbage(t)).collect();
        all.extend_from_slice(rows);
        all.extend((0..post).map(|_| l.garbage(t)));
        let padded = build_plain(t, &all, l);
        let mode = l.pick(&[Slicing::Array, Slicing::Data, Slicing::Ffi, Slicing::Parent]);
        let n = rows.len();
        return match mode {
            Slicing::Array => padded.slice(pre, n),
            Slicing::Data => make_array(padded.to_data().slice(pre, n)),
            Slicing::Parent => {
                let data = padded.to_data();
                let nulls = data.nulls().map(|b| b.slice(pre, n));
                match data
                    .clone()
                    .into_builder()
                    .offset(data.offset() + pre)
                    .len(n)
                    .nulls(nulls)
                    .build()
                {
                    Ok(d) => make_array(d),
                    Err(_) => make_array(data.slice(pre, n)),
                }
            }
            Slicing::Ffi => {
                let mut data = padded.to_data().slice(pre, n);
                // A foreign producer may align buffers to 8 bytes only, as
                // arrow-java does; i128 and i256 values need 16
                // (apache/arrow-rs#10028)
                if matches!(t, GeneratorType::Prim(dt) if dt.primitive_width().is_some_and(|w| w >= 16))
                    && l.chance(0.5)
                {
                    data = misalign_by_8(data);
                }
                let (array, schema) = to_ffi(&data).unwrap();
                // SAFETY: exported just above and imported once
                let imported = unsafe { from_ffi(array, &schema) }.unwrap();
                make_array(imported)
            }
        };
    }
    let a = build_plain(t, rows, l);
    // Independently of slicing: arrive through FFI from a producer that
    // aligns to 8 bytes only (apache/arrow-rs#10028)
    if matches!(t, GeneratorType::Prim(dt) if dt.primitive_width().is_some_and(|w| w >= 16))
        && l.chance(0.3)
    {
        let data = misalign_by_8(a.to_data());
        let (array, schema) = to_ffi(&data).unwrap();
        // SAFETY: exported just above and imported once
        let imported = unsafe { from_ffi(array, &schema) }.unwrap();
        return make_array(imported);
    }
    a
}

/// Replaces null rows by garbage when the layout says so, and returns the
/// validity to attach afterwards
/// A bitmap of `bits`, sometimes with the unused bits of its last byte set,
/// which the spec asks producers to leave unset (line 329) but does not require
fn bitmap(bits: &[bool], l: &mut Layout) -> BooleanBuffer {
    let mut bytes = vec![0u8; bits.len().div_ceil(8)];
    for (i, b) in bits.iter().enumerate() {
        if *b {
            bytes[i / 8] |= 1 << (i % 8);
        }
    }
    if !bits.len().is_multiple_of(8) && l.chance(0.3) {
        let last = bytes.len() - 1;
        bytes[last] |= !0u8 << (bits.len() % 8);
    }
    BooleanBuffer::new(Buffer::from_vec(bytes), 0, bits.len())
}

fn under_nulls(t: &GeneratorType, rows: &[Val], l: &mut Layout) -> (Vec<Val>, Option<NullBuffer>) {
    let valid: Vec<bool> = rows.iter().map(|v| *v != Val::Null).collect();
    let nulls = valid
        .iter()
        .any(|v| !v)
        .then(|| NullBuffer::new(bitmap(&valid, l)));
    let garbage = l.chance(0.5);
    // Garbage under nulls stays inline when every value of a view array is,
    // so that the array can still have no data buffer
    let all_inline = matches!(t, GeneratorType::Utf8View | GeneratorType::BinaryView)
        && rows.iter().all(|v| *v == inline(v.clone()));
    let filled = rows
        .iter()
        .map(|v| match v {
            Val::Null if garbage && all_inline => inline(l.garbage(t)),
            // A null slot of a primitive holds no value, so its garbage may lie
            // outside the domain. A nested type's garbage fills valid slots of
            // its children, so it stays within it
            Val::Null if garbage => l.garbage_in(t, ValueDomain::Arrow),
            Val::Null => default_val(t),
            v => v.clone(),
        })
        .collect();
    (filled, nulls)
}

/// A copy of `bytes` that starts `at` bytes past an aligned allocation, so its
/// address is a multiple of `at` but of nothing larger up to arrow-rs's
/// allocation alignment (at least 32)
fn placed(bytes: &[u8], at: usize) -> Buffer {
    let mut m = MutableBuffer::new(at + bytes.len());
    m.extend_zeros(at);
    m.extend_from_slice(bytes);
    Buffer::from(m).slice(at)
}

/// Rebuilds `a` through its typed constructor with a validity bitmap that has
/// no nulls, for the types whose constructors take one
fn all_valid_bitmap(a: ArrayRef) -> ArrayRef {
    use arrow_array::cast::AsArray;
    let nulls = Some(NullBuffer::new_valid(a.len()));
    match a.data_type() {
        DataType::Int32 => {
            let p = a.as_primitive::<Int32Type>();
            Arc::new(Int32Array::try_new(p.values().clone(), nulls).unwrap())
        }
        DataType::Int64 => {
            let p = a.as_primitive::<arrow_array::types::Int64Type>();
            Arc::new(Int64Array::try_new(p.values().clone(), nulls).unwrap())
        }
        DataType::Boolean => Arc::new(BooleanArray::new(a.as_boolean().values().clone(), nulls)),
        DataType::Utf8 => {
            let s = a.as_string::<i32>();
            Arc::new(StringArray::try_new(s.offsets().clone(), s.values().clone(), nulls).unwrap())
        }
        DataType::Binary => {
            let b = a.as_binary::<i32>();
            Arc::new(BinaryArray::try_new(b.offsets().clone(), b.values().clone(), nulls).unwrap())
        }
        DataType::List(f) => {
            let x = a.as_list::<i32>();
            Arc::new(
                ListArray::try_new(f.clone(), x.offsets().clone(), x.values().clone(), nulls)
                    .unwrap(),
            )
        }
        DataType::Struct(fs) if !fs.is_empty() => {
            let x = a.as_struct();
            Arc::new(StructArray::try_new(fs.clone(), x.columns().to_vec(), nulls).unwrap())
        }
        _ => a,
    }
}

fn with_nulls(a: ArrayRef, nulls: Option<NullBuffer>) -> ArrayRef {
    let data = a.to_data().into_builder().nulls(nulls).build().unwrap();
    make_array(data)
}

fn strs(rows: &[Val]) -> Vec<String> {
    rows.iter()
        .map(|v| {
            if let Val::Str(s) = v {
                s.clone()
            } else {
                unreachable!("{v:?}")
            }
        })
        .collect()
}

fn byts(rows: &[Val]) -> Vec<Vec<u8>> {
    rows.iter()
        .map(|v| {
            if let Val::Bytes(b) = v {
                b.clone()
            } else {
                unreachable!("{v:?}")
            }
        })
        .collect()
}

fn items(v: &Val) -> &[Val] {
    if let Val::List(xs) = v {
        xs
    } else {
        unreachable!("{v:?}")
    }
}

fn build_plain(t: &GeneratorType, rows: &[Val], l: &mut Layout) -> ArrayRef {
    let n = rows.len();
    match t {
        GeneratorType::Exact(e) => retype(&build_plain(&e.built_as, rows, l), &e.dt),
        GeneratorType::Null => Arc::new(NullArray::new(n)),
        // A list of key and value structs in every list layout, retyped as
        // a map
        GeneratorType::Map(k, v, sorted) => {
            let list = build_plain(
                &GeneratorType::List(Box::new(GeneratorType::Struct(vec![
                    (**k).clone(),
                    (**v).clone(),
                ]))),
                rows,
                l,
            )
            .to_data();
            let entries = list.child_data()[0]
                .clone()
                .into_builder()
                .data_type(map_entries(k, v))
                .build()
                .unwrap();
            let entries_field = Arc::new(Field::new("entries", map_entries(k, v), false));
            make_array(
                list.into_builder()
                    .data_type(DataType::Map(entries_field, *sorted))
                    .child_data(vec![entries])
                    .build()
                    .unwrap(),
            )
        }
        GeneratorType::Dict(k, v) => build_dict(k, v, rows, l),
        GeneratorType::Ree(r, v) => build_ree(r, v, rows, l),
        GeneratorType::Union(cs, mode, ids) => build_union(cs, *mode, ids, rows, l),
        _ => {
            let (filled, nulls) = under_nulls(t, rows, l);
            let a: ArrayRef = match t {
                GeneratorType::Bool => {
                    let bits: Vec<bool> = filled
                        .iter()
                        .map(|v| matches!(v, Val::Bool(true)))
                        .collect();
                    Arc::new(BooleanArray::new(bitmap(&bits, l), None))
                }
                GeneratorType::I32 => Arc::new(Int32Array::from(
                    filled
                        .iter()
                        .map(|v| {
                            if let Val::I32(x) = v {
                                *x
                            } else {
                                unreachable!()
                            }
                        })
                        .collect::<Vec<_>>(),
                )),
                GeneratorType::I64 => Arc::new(Int64Array::from(
                    filled
                        .iter()
                        .map(|v| {
                            if let Val::I64(x) = v {
                                *x
                            } else {
                                unreachable!("{v:?}")
                            }
                        })
                        .collect::<Vec<_>>(),
                )),
                GeneratorType::Prim(dt) => {
                    let bytes: Vec<u8> = filled
                        .iter()
                        .flat_map(|v| {
                            if let Val::Raw(b) = v {
                                b.clone()
                            } else {
                                unreachable!("{v:?}")
                            }
                        })
                        .collect();
                    let values = match layout(dt).buffers[0] {
                        // Exactly the alignment arrow-rs requires, which is
                        // below the width for some types, such as 4 for the
                        // two `i32`s of `IntervalDayTime`
                        BufferSpec::FixedWidth { alignment, .. } if l.at_minimum_alignment() => {
                            placed(&bytes, alignment)
                        }
                        // `from_slice_ref` copies into an aligned allocation;
                        // a `Vec<u8>` is only byte aligned
                        _ => Buffer::from_slice_ref(&bytes),
                    };
                    make_array(
                        ArrayData::builder(dt.clone())
                            .len(n)
                            .add_buffer(values)
                            .build()
                            .unwrap(),
                    )
                }
                GeneratorType::Utf8 => Arc::new(StringArray::from(strs(&filled))),
                GeneratorType::LargeUtf8 => Arc::new(LargeStringArray::from(strs(&filled))),
                GeneratorType::Binary => Arc::new(BinaryArray::from_iter_values(byts(&filled))),
                GeneratorType::LargeBinary => {
                    Arc::new(LargeBinaryArray::from_iter_values(byts(&filled)))
                }
                GeneratorType::Utf8View => {
                    // Small blocks put long strings in several data buffers;
                    // deduplication makes equal strings share their bytes
                    let block = l.pick(&[8192u32, 16, 32]);
                    let mut b = StringViewBuilder::new().with_fixed_block_size(block);
                    if l.chance(0.5) {
                        b = b.with_deduplicate_strings();
                    }
                    strs(&filled).iter().for_each(|s| b.append_value(s));
                    let a = b.finish();
                    let (views, buffers, nulls) = a.into_parts();
                    Arc::new(
                        StringViewArray::try_new(views, extra_buffers(buffers.to_vec(), l), nulls)
                            .unwrap(),
                    )
                }
                GeneratorType::BinaryView => {
                    let block = l.pick(&[8192u32, 16, 32]);
                    let mut b = BinaryViewBuilder::new().with_fixed_block_size(block);
                    if l.chance(0.5) {
                        b = b.with_deduplicate_strings();
                    }
                    byts(&filled).iter().for_each(|s| b.append_value(s));
                    let a = b.finish();
                    let (views, buffers, nulls) = a.into_parts();
                    Arc::new(
                        arrow_array::BinaryViewArray::try_new(
                            views,
                            extra_buffers(buffers.to_vec(), l),
                            nulls,
                        )
                        .unwrap(),
                    )
                }
                // Width 0 takes its length from `len`, not from the buffer
                GeneratorType::Fsb(0) => make_array(
                    ArrayData::builder(t.data_type())
                        .len(n)
                        .add_buffer(Buffer::from_vec(Vec::<u8>::new()))
                        .build()
                        .unwrap(),
                ),
                GeneratorType::Fsb(w) => {
                    let data: Vec<u8> = byts(&filled).concat();
                    Arc::new(
                        FixedSizeBinaryArray::try_new(*w, Buffer::from_vec(data), None).unwrap(),
                    )
                }
                GeneratorType::List(c) | GeneratorType::LargeList(c) => {
                    let lens: Vec<usize> = filled.iter().map(|v| items(v).len()).collect();
                    let flat: Vec<Val> = filled.iter().flat_map(|v| items(v).to_vec()).collect();
                    let child = child_node(c, &flat, l);
                    let f = Arc::new(field("item", c));
                    match t {
                        GeneratorType::List(_) => Arc::new(ListArray::new(
                            f,
                            OffsetBuffer::from_lengths(lens),
                            child,
                            None,
                        )),
                        _ => Arc::new(LargeListArray::new(
                            f,
                            OffsetBuffer::from_lengths(lens),
                            child,
                            None,
                        )),
                    }
                }
                GeneratorType::ListView(c) => build_list_view(c, &filled, l, false),
                GeneratorType::LargeListView(c) => build_list_view(c, &filled, l, true),
                // Size 0 takes its length from `len`, not from the child
                GeneratorType::Fsl(c, 0) => {
                    let child = build_node(c, &[], l);
                    make_array(
                        ArrayData::builder(t.data_type())
                            .len(n)
                            .add_child_data(child.to_data())
                            .build()
                            .unwrap(),
                    )
                }
                GeneratorType::Fsl(c, size) => {
                    let flat: Vec<Val> = filled.iter().flat_map(|v| items(v).to_vec()).collect();
                    let child = child_node(c, &flat, l);
                    Arc::new(
                        FixedSizeListArray::try_new(Arc::new(field("item", c)), *size, child, None)
                            .unwrap(),
                    )
                }
                GeneratorType::Struct(fs) => {
                    let cols: Vec<ArrayRef> = (0..fs.len())
                        .map(|j| {
                            let col: Vec<Val> = filled
                                .iter()
                                .map(|v| {
                                    if let Val::Struct(xs) = v {
                                        xs[j].clone()
                                    } else {
                                        unreachable!()
                                    }
                                })
                                .collect();
                            build_node(&fs[j], &col, l)
                        })
                        .collect();
                    if fs.is_empty() {
                        Arc::new(StructArray::new_empty_fields(n, None))
                    } else {
                        Arc::new(StructArray::new(struct_fields(fs), cols, None))
                    }
                }
                _ => unreachable!(),
            };
            match nulls {
                Some(_) => with_nulls(a, nulls),
                // A validity bitmap with no nulls, which `ArrayData` drops, so
                // only typed constructors can carry it
                None if n > 0 && l.chance(0.4) => all_valid_bitmap(a),
                None => a,
            }
        }
    }
}

/// A child array for `rows`, which sometimes starts at a nonzero offset of
/// its own. The C data interface gives every array, children included, a
/// logical offset into its buffers (CDataInterface.rst lines 456-464)
fn child_node(c: &GeneratorType, rows: &[Val], l: &mut Layout) -> ArrayRef {
    if !l.chance(0.2) {
        return build_node(c, rows, l);
    }
    let pad = l.range(1, 3);
    let mut padded: Vec<Val> = (0..pad).map(|_| l.garbage(c)).collect();
    padded.extend_from_slice(rows);
    build_node(c, &padded, l).slice(pad, rows.len())
}

/// Data buffers no view points into: an empty one, or one of garbage
fn extra_buffers(mut buffers: Vec<Buffer>, l: &mut Layout) -> Vec<Buffer> {
    if l.chance(0.2) {
        // Appended, as inserting would shift the indices views refer to
        buffers.push(Buffer::from_vec(Vec::<u8>::new()));
    }
    if l.chance(0.2) {
        buffers.push(Buffer::from_vec(vec![0xAB_u8; 40]));
    }
    buffers
}

fn build_list_view(c: &GeneratorType, rows: &[Val], l: &mut Layout, large: bool) -> ArrayRef {
    // Place each row's items in a random order, with gaps, so offsets are
    // neither sorted nor contiguous
    let mut order: Vec<usize> = (0..rows.len()).collect();
    if l.chance(0.5) {
        order.reverse();
    }
    let mut flat: Vec<Val> = vec![];
    let mut offsets = vec![0i32; rows.len()];
    let mut sizes = vec![0i32; rows.len()];
    let share = l.chance(0.3);
    for &i in &order {
        let xs = items(&rows[i]);
        // Rows with the same items can share one range of the child
        let found = (share && !xs.is_empty() && flat.len() >= xs.len())
            .then(|| {
                (0..=flat.len().saturating_sub(xs.len())).find(|&p| flat[p..p + xs.len()] == *xs)
            })
            .flatten();
        if let Some(p) = found {
            offsets[i] = p as i32;
            sizes[i] = xs.len() as i32;
            continue;
        }
        if l.chance(0.3) {
            flat.push(l.garbage(c));
        }
        offsets[i] = flat.len() as i32;
        sizes[i] = xs.len() as i32;
        flat.extend(xs.iter().cloned());
    }
    let child = child_node(c, &flat, l);
    let f = Arc::new(field("item", c));
    if large {
        let widen =
            |v: Vec<i32>| ScalarBuffer::from(v.into_iter().map(i64::from).collect::<Vec<_>>());
        return Arc::new(LargeListViewArray::new(
            f,
            widen(offsets),
            widen(sizes),
            child,
            None,
        ));
    }
    Arc::new(ListViewArray::new(
        f,
        ScalarBuffer::from(offsets),
        ScalarBuffer::from(sizes),
        child,
        None,
    ))
}

/// The most values a dictionary with key type `k` can address
fn key_capacity(k: &DataType) -> usize {
    match k {
        DataType::Int8 => i8::MAX as usize,
        DataType::UInt8 => u8::MAX as usize,
        _ => 300,
    }
}

fn build_dict(k: &DataType, v: &GeneratorType, rows: &[Val], l: &mut Layout) -> ArrayRef {
    use arrow_array::types::*;
    let cap = key_capacity(k);
    let mut values: Vec<Val> = vec![];
    let mut keys: Vec<Option<i32>> = vec![];
    // A null row is a null key, or a key that points at a null value
    let null_value = l.chance(0.7);
    let dedup = !l.chance(0.3);
    for r in rows {
        if *r == Val::Null && !null_value {
            keys.push(None);
            continue;
        }
        // Narrow keys can't address more values than their capacity
        let pos = (dedup || values.len() + 8 >= cap)
            .then(|| values.iter().position(|x| x == r))
            .flatten();
        let k = pos.unwrap_or_else(|| {
            values.push(r.clone());
            values.len() - 1
        });
        keys.push(Some(k as i32));
    }
    // Unused values, up to the key type's capacity
    let extra = l
        .pick(&[0usize, 1, 5, cap])
        .min(cap.saturating_sub(values.len()));
    for _ in 0..extra {
        values.push(l.garbage(v));
    }
    if values.is_empty() {
        values.push(default_val(v));
    }
    let values = build_node(v, &values, l);
    macro_rules! dict {
        ($kt:ty, $at:ty, $n:ty) => {
            Arc::new(
                DictionaryArray::<$kt>::try_new(
                    // Each key is below the key type's capacity, so it fits
                    <$at>::from(
                        keys.iter()
                            .map(|k| k.map(|k| <$n>::try_from(k).unwrap()))
                            .collect::<Vec<_>>(),
                    ),
                    values,
                )
                .unwrap(),
            )
        };
    }
    match k {
        DataType::Int8 => dict!(Int8Type, Int8Array, i8),
        DataType::Int16 => dict!(Int16Type, Int16Array, i16),
        DataType::Int32 => dict!(Int32Type, Int32Array, i32),
        DataType::Int64 => dict!(Int64Type, Int64Array, i64),
        DataType::UInt8 => dict!(UInt8Type, UInt8Array, u8),
        DataType::UInt16 => dict!(UInt16Type, UInt16Array, u16),
        DataType::UInt32 => dict!(UInt32Type, UInt32Array, u32),
        DataType::UInt64 => dict!(UInt64Type, UInt64Array, u64),
        k => unreachable!("key type {k}"),
    }
}

fn build_ree(r: &DataType, v: &GeneratorType, rows: &[Val], l: &mut Layout) -> ArrayRef {
    // Runs of equal rows, sometimes split so runs are not maximal
    let split = l.chance(0.5);
    let mut ends: Vec<i32> = vec![];
    let mut values: Vec<Val> = vec![];
    for (i, r) in rows.iter().enumerate() {
        let same = values.last() == Some(r) && !(split && l.chance(0.5));
        if same {
            *ends.last_mut().unwrap() = i as i32 + 1;
        } else {
            values.push(r.clone());
            ends.push(i as i32 + 1);
        }
    }
    let values = build_node(v, &values, l);
    let a: ArrayRef = match r {
        DataType::Int16 => {
            let e = Int16Array::from(ends.iter().map(|&x| x as i16).collect::<Vec<_>>());
            Arc::new(
                RunArray::<arrow_array::types::Int16Type>::try_new(&e, values.as_ref()).unwrap(),
            )
        }
        DataType::Int32 => Arc::new(
            RunArray::<Int32Type>::try_new(&Int32Array::from(ends), values.as_ref()).unwrap(),
        ),
        DataType::Int64 => {
            let e = Int64Array::from(ends.iter().map(|&x| x as i64).collect::<Vec<_>>());
            Arc::new(
                RunArray::<arrow_array::types::Int64Type>::try_new(&e, values.as_ref()).unwrap(),
            )
        }
        r => unreachable!("run end type {r}"),
    };
    // `RunArray::try_new` declares the values field nullable, so a values
    // field that is not takes the declared type
    let dt = GeneratorType::Ree(r.clone(), Box::new(v.clone())).data_type();
    if a.data_type() == &dt {
        return a;
    }
    make_array(a.to_data().into_builder().data_type(dt).build().unwrap())
}

fn build_union(
    cs: &[GeneratorType],
    mode: UnionMode,
    ids: &[i8],
    rows: &[Val],
    l: &mut Layout,
) -> ArrayRef {
    let fields = union_fields(cs, ids);
    let tids: Vec<i8> = rows
        .iter()
        .map(|v| {
            if let Val::Union(t, _) = v {
                *t
            } else {
                unreachable!("{v:?}")
            }
        })
        .collect();
    let child_of = |t: i8| ids.iter().position(|&x| x == t).unwrap();
    match mode {
        UnionMode::Sparse => {
            let children = cs
                .iter()
                .enumerate()
                .map(|(c, ct)| {
                    let col: Vec<Val> = rows
                        .iter()
                        .map(|v| match v {
                            Val::Union(t, x) if child_of(*t) == c => (**x).clone(),
                            _ => l.garbage(ct),
                        })
                        .collect();
                    build_node(ct, &col, l)
                })
                .collect();
            Arc::new(UnionArray::try_new(fields, tids.into(), None, children).unwrap())
        }
        UnionMode::Dense => {
            let mut cols: Vec<Vec<Val>> = vec![vec![]; cs.len()];
            let mut offsets = vec![];
            // Kernels slice a child instead of taking from it when its offsets
            // are sequential, so each array's offsets take one of three
            // shapes. Plain offsets reach those fast paths, and mixed ones,
            // with gaps and shared slots anywhere, reach the general paths.
            // Nearly sequential offsets, with one shared slot and one gap,
            // pass a cheap check of the first and last offsets but are not
            // sequential, so only a full check sends them down the right path.
            // Without the shared slot, the lone gap lands at row 64 or at the
            // last multiple of 64 as often as anywhere, since checks that scan
            // offsets in chunks of 64 compare each chunk's first offset
            let mode = l.range(0, 2);
            let n = rows.len();
            let gap_at = match l.range(0, 2) {
                0 => l.range(0, n),
                1 => 64.min(n),
                _ => n - n % 64,
            };
            let mut share_left = mode == 2 && l.chance(0.5);
            for (i, v) in rows.iter().enumerate() {
                let Val::Union(t, x) = v else { unreachable!() };
                let c = child_of(*t);
                // Rows with the same value may share a child slot. The format
                // requires each child's offsets "in order / increasing"
                // (Columnar.rst lines 891-894), and Arrow C++'s ValidateFull
                // rejects only an offset below the previous one
                let share = cols[c].last() == Some(&**x)
                    && match mode {
                        1 => l.chance(0.5),
                        2 => share_left,
                        _ => false,
                    };
                if share {
                    share_left = false;
                    offsets.push(cols[c].len() as i32 - 1);
                    continue;
                }
                let gap = match mode {
                    1 => l.chance(0.2),
                    2 => i == gap_at,
                    _ => false,
                };
                if gap {
                    cols[c].push(l.garbage(&cs[c]));
                }
                offsets.push(cols[c].len() as i32);
                cols[c].push((**x).clone());
            }
            // A child may be longer than its offsets use, even when no row
            // refers to it. Some children are padded, half of them to the
            // union's length
            for (col, ct) in cols.iter_mut().zip(cs) {
                if l.chance(0.2) {
                    let to = if l.chance(0.5) {
                        rows.len()
                    } else {
                        col.len() + l.range(1, 3)
                    };
                    while col.len() < to {
                        col.push(l.garbage(ct));
                    }
                }
            }
            let children = cs
                .iter()
                .zip(&cols)
                .map(|(ct, col)| build_node(ct, col, l))
                .collect();
            Arc::new(
                UnionArray::try_new(fields, tids.into(), Some(offsets.into()), children).unwrap(),
            )
        }
    }
}

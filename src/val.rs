//! The oracle's view of an array: one logical value per row, read through
//! typed accessors. It shares no code with the kernels under test or with
//! `ArrayData` equality.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType;
use proptest::test_runner::TestCaseError;
use std::cell::Cell;

use crate::spec;

/// The logical value of one row, which [`vals`] reads and models are written
/// in. A dictionary or run-end encoded row reads as its value, and a null
/// union row reads as `Null`, without its type id
#[derive(Clone, Debug, PartialEq)]
pub enum Val {
    /// A null row, of any type
    Null,
    /// A `Boolean` row
    Bool(bool),
    /// An `Int32` row
    I32(i32),
    /// An `Int64` row
    I64(i64),
    /// A `Utf8`, `LargeUtf8` or `Utf8View` row
    Str(String),
    /// A `Binary`, `LargeBinary`, `BinaryView` or `FixedSizeBinary` row
    Bytes(Vec<u8>),
    /// Any other fixed-width row (the other integers, floats, decimals,
    /// temporal types and intervals), as its little-endian bytes
    Raw(Vec<u8>),
    /// A row of any list type, or of a `Map`, whose entries are `Struct`s of
    /// the key and the value
    List(Vec<Val>),
    /// A `Struct` row, one value per field
    Struct(Vec<Val>),
    /// The type id and the value in the selected child
    Union(i8, Box<Val>),
}

/// The logical value of every row of `a`, read without arrow-rs's
/// comparison or cast kernels
///
/// # Example
///
/// ```
/// use arrow_array::Int32Array;
/// use arrow_pbt::val::{Val, vals};
/// let a = Int32Array::from(vec![Some(1), None]);
/// assert_eq!(vals(&a), [Val::I32(1), Val::Null]);
/// ```
pub fn vals(a: &dyn Array) -> Vec<Val> {
    (0..a.len()).map(|i| val_at(a, i)).collect()
}

fn list_vals(child: ArrayRef) -> Val {
    Val::List(vals(child.as_ref()))
}

thread_local! {
    /// Whether a null union row reads as `Union(type id, Null)` instead of
    /// `Null`, for modeling kernels that order null union rows by type id
    static KEEP_UNION_IDS: Cell<bool> = const { Cell::new(false) };
}

/// The values of `a`, with null union rows keeping their type id
pub(crate) fn vals_keep_union_ids(a: &dyn Array) -> Vec<Val> {
    KEEP_UNION_IDS.set(true);
    let v = vals(a);
    KEEP_UNION_IDS.set(false);
    v
}

pub(crate) fn val_at(a: &dyn Array, i: usize) -> Val {
    use DataType::*;
    match a.data_type() {
        Null => Val::Null,
        Union(..) => {
            let u = a.as_union();
            let tid = u.type_id(i);
            let child = u.child(tid);
            // A union has no validity of its own: a row is null when the
            // selected child's slot is
            match val_at(child.as_ref(), u.value_offset(i)) {
                Val::Null if !KEEP_UNION_IDS.get() => Val::Null,
                v => Val::Union(tid, Box::new(v)),
            }
        }
        RunEndEncoded(r, _) => {
            // Physical index by hand: the first run that ends after the
            // logical index, counting the slice offset
            macro_rules! physical {
                ($t:ty) => {{
                    let r = a.as_run::<$t>();
                    let ends = r.run_ends();
                    let logical = (ends.offset() + i) as i64;
                    let p = ends.values().partition_point(|&e| i64::from(e) <= logical);
                    (p, r.values().clone())
                }};
            }
            let (p, values) = match r.data_type() {
                Int16 => physical!(Int16Type),
                Int32 => physical!(Int32Type),
                Int64 => physical!(Int64Type),
                t => panic!("unsupported run end type {t}"),
            };
            val_at(values.as_ref(), p)
        }
        Dictionary(k, _) => {
            macro_rules! key {
                ($t:ty) => {{
                    let d = a.as_dictionary::<$t>();
                    (
                        d.keys().is_valid(i).then(|| d.keys().value(i) as usize),
                        d.values().clone(),
                    )
                }};
            }
            let (key, values) = match k.as_ref() {
                Int8 => key!(Int8Type),
                Int16 => key!(Int16Type),
                Int32 => key!(Int32Type),
                Int64 => key!(Int64Type),
                UInt8 => key!(UInt8Type),
                UInt16 => key!(UInt16Type),
                UInt32 => key!(UInt32Type),
                UInt64 => key!(UInt64Type),
                k => panic!("unsupported key type {k}"),
            };
            key.map_or(Val::Null, |k| val_at(values.as_ref(), k))
        }
        _ if a.is_null(i) => Val::Null,
        Boolean => Val::Bool(a.as_boolean().value(i)),
        Int32 => Val::I32(a.as_primitive::<Int32Type>().value(i)),
        Int64 => Val::I64(a.as_primitive::<Int64Type>().value(i)),
        // Every other fixed-width type, read straight from the values buffer
        t if t.primitive_width().is_some() && !matches!(t, Boolean) => {
            let d = a.to_data();
            let w = t.primitive_width().unwrap();
            let start = (d.offset() + i) * w;
            Val::Raw(d.buffers()[0][start..start + w].to_vec())
        }
        Utf8 => Val::Str(a.as_string::<i32>().value(i).to_string()),
        LargeUtf8 => Val::Str(a.as_string::<i64>().value(i).to_string()),
        Utf8View => Val::Str(a.as_string_view().value(i).to_string()),
        Binary => Val::Bytes(a.as_binary::<i32>().value(i).to_vec()),
        LargeBinary => Val::Bytes(a.as_binary::<i64>().value(i).to_vec()),
        BinaryView => Val::Bytes(a.as_binary_view().value(i).to_vec()),
        FixedSizeBinary(_) => Val::Bytes(a.as_fixed_size_binary().value(i).to_vec()),
        List(_) => list_vals(a.as_list::<i32>().value(i)),
        LargeList(_) => list_vals(a.as_list::<i64>().value(i)),
        ListView(_) => list_vals(a.as_list_view::<i32>().value(i)),
        LargeListView(_) => list_vals(a.as_list_view::<i64>().value(i)),
        FixedSizeList(..) => list_vals(a.as_fixed_size_list().value(i)),
        Map(..) => list_vals(Arc::new(a.as_map().value(i))),
        Struct(_) => Val::Struct(
            a.as_struct()
                .columns()
                .iter()
                .map(|c| val_at(c.as_ref(), i))
                .collect(),
        ),
        t => panic!("unsupported type {t}"),
    }
}

impl Val {
    /// A `Float32` or `Float64` value as `f64`, which holds every `f32` exactly
    ///
    /// # Example
    ///
    /// ```
    /// use arrow_pbt::val::Val;
    /// assert_eq!(Val::Raw(1.5f32.to_le_bytes().to_vec()).as_f64(), 1.5);
    /// ```
    pub fn as_f64(&self) -> f64 {
        match self {
            Val::Raw(b) if b.len() == 4 => f64::from(f32::from_le_bytes(b[..].try_into().unwrap())),
            Val::Raw(b) => f64::from_le_bytes(b[..].try_into().unwrap()),
            v => panic!("not a float: {v:?}"),
        }
    }
}

/// Whether two floats are the same value: equal bits, or both NaN
///
/// # Example
///
/// ```
/// use arrow_pbt::val::same_float;
/// assert!(same_float(f64::NAN, -f64::NAN));
/// assert!(!same_float(0.0, -0.0));
/// ```
pub fn same_float(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// Checks an output: valid under `validate_full` and the spec checker, of
/// type `dt`, and holding `expected` as `vals` reads it. A union row whose
/// child is null reads as `Null` and may be expected as `Union(id, Null)`
/// too, which is how the generator's rows write it
///
/// # Errors
///
/// A failure that names the first difference
///
/// # Example
///
/// ```
/// use arrow_array::Int32Array;
/// use arrow_pbt::val::{Val, check_rows};
/// use arrow_schema::DataType;
/// let a = Int32Array::from(vec![Some(1), None]);
/// assert!(check_rows(&a, &DataType::Int32, &[Val::I32(1), Val::Null]).is_ok());
/// assert!(check_rows(&a, &DataType::Int32, &[Val::I32(2), Val::Null]).is_err());
/// ```
pub fn check_rows(out: &dyn Array, dt: &DataType, expected: &[Val]) -> Result<(), TestCaseError> {
    let data = out.to_data();
    data.validate_full()
        .map_err(|e| TestCaseError::fail(format!("invalid output: {e}")))?;
    spec::check(&data)
        .map_err(|e| TestCaseError::fail(format!("output violates the spec: {e}")))?;
    if out.data_type() != dt {
        return Err(TestCaseError::fail(format!(
            "type {} != {dt}",
            out.data_type()
        )));
    }
    if out.len() != expected.len() {
        return Err(TestCaseError::fail(format!(
            "len {} != {}",
            out.len(),
            expected.len()
        )));
    }
    let got = vals(out);
    if let Some(i) = (0..got.len()).find(|&i| !same_row(&got[i], &expected[i])) {
        return Err(TestCaseError::fail(format!(
            "row {i}: got {:?}, expected {:?}",
            got[i], expected[i]
        )));
    }
    Ok(())
}

/// Whether `got`, as `vals` reads it, equals `expected`, which may write a
/// union row whose child is null as `Union(id, Null)`
fn same_row(got: &Val, expected: &Val) -> bool {
    match (got, expected) {
        (Val::Null, Val::Union(_, c)) => **c == Val::Null,
        (Val::Union(a, x), Val::Union(b, y)) => a == b && same_row(x, y),
        (Val::List(xs), Val::List(ys)) | (Val::Struct(xs), Val::Struct(ys)) => {
            xs.len() == ys.len() && xs.iter().zip(ys).all(|(x, y)| same_row(x, y))
        }
        _ => got == expected,
    }
}

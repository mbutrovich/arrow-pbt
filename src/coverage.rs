//! Measures which features (see docs/design.md) generated arrays actually exhibit, by
//! inspecting the built arrays rather than trusting the generator's choices:
//! arrow-rs normalizes some layouts away (an empty validity bitmap, a
//! struct's parent offset), so intent and outcome can differ.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{Array, ArrayRef, BooleanArray, Int32Array};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType, UnionMode};
use num_bigint::{BigInt, Sign};
use num_traits::{Signed, ToPrimitive};

use crate::generator::{Domain, domain_of, narrow, narrower};
use crate::val::{Val, val_at};
use arrow_array::types::Int16Type;
use arrow_array::types::Int64Type;

/// Every feature the generator is meant to produce, so that one never seen
/// is reported as 0% rather than missing
pub const EXPECTED: &[&str] = &[
    "empty array",
    "length at a byte or word boundary",
    "length >= 1024",
    "all slots null",
    "validity bitmap present with no nulls",
    "validity bitmap padding bits set",
    "Boolean values padding bits set",
    "nonzero ArrayData offset",
    "string or binary null slot over a non-empty range",
    "string or binary offsets not starting at 0",
    "view array with no data buffers",
    "view array of only long values",
    "view array with several data buffers",
    "views sharing the same bytes",
    "view data buffer no view points into",
    "list null row over a non-empty range",
    "list child with a nonzero offset of its own",
    "list offsets not starting at 0",
    "map with sorted keys",
    "list view offsets out of order",
    "list view rows sharing a range",
    "list view gap in the child",
    "FixedSizeList or FixedSizeBinary of width 0",
    "struct with no fields",
    "struct child valid under a null parent row",
    "union with non-default type ids",
    "union of several fields whose rows all have one type id",
    "union of several fields whose rows all but one have one type id",
    "dense union child with unused slots",
    "dense union child with slots that no row references",
    "dense union rows that share a child slot",
    "dense union of one type id with sequential offsets",
    "dense union of one type id with offsets sequential only at the ends",
    "dictionary with unused values",
    "dictionary with a null value that a key points at",
    "dictionary with duplicate values",
    "dictionary with unsigned keys",
    "run-end encoded with a non-maximal run",
    "run-end encoded with Int16 or Int64 run ends",
    "child field not nullable",
    "primitive values aligned below their width",
];

/// Whether bits past `len` in the last byte of the bitmap are set
fn padding_set(n: &arrow_buffer::BooleanBuffer) -> bool {
    let end = n.offset() + n.len();
    if end.is_multiple_of(8) {
        return false;
    }
    let bytes = n.values();
    bytes.len() > end / 8 && bytes[end / 8] >> (end % 8) != 0
}

pub fn features(a: &dyn Array, out: &mut BTreeSet<&'static str>) {
    let n = a.len();
    if n == 0 {
        out.insert("empty array");
    }
    if matches!(n, 7 | 8 | 9 | 15 | 16 | 17 | 63 | 64 | 65 | 127 | 128 | 129) {
        out.insert("length at a byte or word boundary");
    }
    if n >= 1024 {
        out.insert("length >= 1024");
    }
    let nulls: Option<&NullBuffer> = a.nulls();
    if let Some(nb) = nulls {
        if nb.null_count() == 0 {
            out.insert("validity bitmap present with no nulls");
        }
        if n > 0 && nb.null_count() == n {
            out.insert("all slots null");
        }
        if padding_set(nb.inner()) {
            out.insert("validity bitmap padding bits set");
        }
    }
    if a.to_data().offset() > 0 {
        out.insert("nonzero ArrayData offset");
    }
    let is_null = |i: usize| nulls.is_some_and(|nb| nb.is_null(i));
    // Child fields the spec leaves to the schema: list items, struct and
    // union fields, map values, and run-end encoded values
    let child_fields: Vec<&arrow_schema::FieldRef> = match a.data_type() {
        DataType::List(f)
        | DataType::LargeList(f)
        | DataType::ListView(f)
        | DataType::LargeListView(f)
        | DataType::FixedSizeList(f, _) => vec![f],
        DataType::Struct(fs) => fs.iter().collect(),
        DataType::Union(fs, _) => fs.iter().map(|(_, f)| f).collect(),
        DataType::RunEndEncoded(_, v) => vec![v],
        DataType::Map(e, _) => match e.data_type() {
            DataType::Struct(kv) => kv.iter().skip(1).collect(),
            _ => vec![],
        },
        _ => vec![],
    };
    if child_fields.iter().any(|f| !f.is_nullable()) {
        out.insert("child field not nullable");
    }
    match a.data_type() {
        DataType::Boolean => {
            if padding_set(a.as_boolean().values()) {
                out.insert("Boolean values padding bits set");
            }
        }
        DataType::Utf8 | DataType::Binary | DataType::LargeUtf8 | DataType::LargeBinary => {
            let o: Vec<i64> = match a.data_type() {
                DataType::Utf8 => a
                    .as_string::<i32>()
                    .value_offsets()
                    .iter()
                    .map(|&x| x as i64)
                    .collect(),
                DataType::Binary => a
                    .as_binary::<i32>()
                    .value_offsets()
                    .iter()
                    .map(|&x| x as i64)
                    .collect(),
                DataType::LargeUtf8 => a.as_string::<i64>().value_offsets().to_vec(),
                _ => a.as_binary::<i64>().value_offsets().to_vec(),
            };
            if (0..n).any(|i| is_null(i) && o[i + 1] > o[i]) {
                out.insert("string or binary null slot over a non-empty range");
            }
            if o[0] > 0 {
                out.insert("string or binary offsets not starting at 0");
            }
        }
        DataType::Utf8View | DataType::BinaryView => {
            let (views, buffers) = match a.data_type() {
                DataType::Utf8View => (
                    a.as_string_view().views().clone(),
                    a.as_string_view().data_buffers().to_vec(),
                ),
                _ => (
                    a.as_binary_view().views().clone(),
                    a.as_binary_view().data_buffers().to_vec(),
                ),
            };
            if buffers.len() > 1 {
                out.insert("view array with several data buffers");
            }
            // A view's first four bytes are its length (Columnar.rst line 506)
            if n > 0 && views.iter().all(|v| (*v as u32) > 12) {
                out.insert("view array of only long values");
            }
            if n > 0 && buffers.is_empty() {
                out.insert("view array with no data buffers");
            }
            let mut seen = HashSet::new();
            let mut used = HashSet::new();
            for v in &views {
                let len = *v as u32;
                if len > 12 {
                    let (idx, off) = ((v >> 64) as u32, (v >> 96) as u32);
                    used.insert(idx);
                    if !seen.insert((idx, off)) {
                        out.insert("views sharing the same bytes");
                    }
                }
            }
            if (0..buffers.len() as u32).any(|i| !used.contains(&i)) {
                out.insert("view data buffer no view points into");
            }
        }
        DataType::List(_) | DataType::LargeList(_) | DataType::Map(..) => {
            let (o, child): (Vec<i64>, ArrayRef) = match a.data_type() {
                DataType::Map(..) => {
                    let m = a.as_map();
                    (
                        m.value_offsets().iter().map(|&x| x as i64).collect(),
                        Arc::new(m.entries().clone()) as ArrayRef,
                    )
                }
                DataType::List(_) => {
                    let l = a.as_list::<i32>();
                    (
                        l.value_offsets().iter().map(|&x| x as i64).collect(),
                        l.values().clone(),
                    )
                }
                _ => {
                    let l = a.as_list::<i64>();
                    (l.value_offsets().to_vec(), l.values().clone())
                }
            };
            if (0..n).any(|i| is_null(i) && o[i + 1] > o[i]) {
                out.insert("list null row over a non-empty range");
            }
            if child.offset() > 0 {
                out.insert("list child with a nonzero offset of its own");
            }
            if o[0] > 0 {
                out.insert("list offsets not starting at 0");
            }
            if matches!(a.data_type(), DataType::Map(_, true)) {
                out.insert("map with sorted keys");
            }
            features(child.as_ref(), out);
        }
        DataType::ListView(_) | DataType::LargeListView(_) => {
            let (o, s, child): (Vec<i64>, Vec<i64>, ArrayRef) = match a.data_type() {
                DataType::ListView(_) => {
                    let l = a.as_list_view::<i32>();
                    (
                        l.offsets().iter().map(|&x| x as i64).collect(),
                        l.sizes().iter().map(|&x| x as i64).collect(),
                        l.values().clone(),
                    )
                }
                _ => {
                    let l = a.as_list_view::<i64>();
                    (l.offsets().to_vec(), l.sizes().to_vec(), l.values().clone())
                }
            };
            if o.windows(2).any(|w| w[1] < w[0]) {
                out.insert("list view offsets out of order");
            }
            let mut ranges = HashSet::new();
            if (0..n).any(|i| s[i] > 0 && !ranges.insert((o[i], s[i]))) {
                out.insert("list view rows sharing a range");
            }
            let mut covered = vec![false; child.len()];
            for i in 0..n {
                for j in o[i]..o[i] + s[i] {
                    covered[j as usize] = true;
                }
            }
            if covered.iter().any(|c| !c) {
                out.insert("list view gap in the child");
            }
            features(child.as_ref(), out);
        }
        DataType::FixedSizeList(_, size) => {
            if *size == 0 {
                out.insert("FixedSizeList or FixedSizeBinary of width 0");
            }
            features(a.as_fixed_size_list().values().as_ref(), out);
        }
        DataType::FixedSizeBinary(0) => {
            out.insert("FixedSizeList or FixedSizeBinary of width 0");
        }
        DataType::Struct(fs) => {
            if fs.is_empty() {
                out.insert("struct with no fields");
            }
            for c in a.as_struct().columns() {
                if (0..n).any(|i| is_null(i) && c.is_valid(i)) {
                    out.insert("struct child valid under a null parent row");
                }
                features(c.as_ref(), out);
            }
        }
        DataType::Union(fs, mode) => {
            let u = a.as_union();
            if fs.iter().enumerate().any(|(i, (id, _))| id as usize != i) {
                out.insert("union with non-default type ids");
            }
            if fs.len() > 1 && n > 1 && (1..n).all(|i| u.type_id(i) == u.type_id(0)) {
                out.insert("union of several fields whose rows all have one type id");
            }
            if fs.len() > 1
                && n > 2
                && (0..n).filter(|&i| u.type_id(i) != u.type_id(n / 2)).count() == 1
            {
                out.insert("union of several fields whose rows all but one have one type id");
            }
            if *mode == UnionMode::Dense {
                // Offsets into a child are in order, so a shared slot repeats
                // the child's previous offset
                let mut last = HashMap::new();
                if (0..n).any(|i| {
                    last.insert(u.type_id(i), u.value_offset(i)) == Some(u.value_offset(i))
                }) {
                    out.insert("dense union rows that share a child slot");
                }
                // The shape `union_extract` checks before slicing: the first
                // and last offsets of a single-child run are `n - 1` apart
                if n > 1 && (1..n).all(|i| u.type_id(i) == u.type_id(0)) {
                    let o: Vec<i32> = (0..n).map(|i| u.value_offset(i) as i32).collect();
                    if o.windows(2).all(|w| w[1] == w[0] + 1) {
                        out.insert("dense union of one type id with sequential offsets");
                    } else if o[n - 1] - o[0] == n as i32 - 1 {
                        out.insert(
                            "dense union of one type id with offsets sequential only at the ends",
                        );
                    }
                }
                for (id, _) in fs.iter() {
                    let refs = (0..n).filter(|&i| u.type_id(i) == id).count();
                    if u.child(id).len() > refs {
                        out.insert("dense union child with unused slots");
                    }
                    if refs == 0 && !u.child(id).is_empty() {
                        out.insert("dense union child with slots that no row references");
                    }
                }
            }
            for (id, _) in fs.iter() {
                features(u.child(id).as_ref(), out);
            }
        }
        DataType::Dictionary(k, _) => {
            if matches!(
                **k,
                DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64
            ) {
                out.insert("dictionary with unsigned keys");
            }
            let d = a.as_any_dictionary();
            let keys = d.normalized_keys();
            let values = d.values();
            let referenced: HashSet<usize> = (0..n)
                .filter(|&i| d.keys().is_valid(i))
                .map(|i| keys[i])
                .collect();
            if referenced.len() < values.len() {
                out.insert("dictionary with unused values");
            }
            if referenced.iter().any(|&k| values.is_null(k)) {
                out.insert("dictionary with a null value that a key points at");
            }
            let vals: Vec<_> = (0..values.len())
                .map(|i| val_at(values.as_ref(), i))
                .collect();
            if (0..vals.len()).any(|i| vals[..i].contains(&vals[i])) {
                out.insert("dictionary with duplicate values");
            }
            features(values.as_ref(), out);
        }
        DataType::RunEndEncoded(r, _) => {
            if matches!(r.data_type(), DataType::Int16 | DataType::Int64) {
                out.insert("run-end encoded with Int16 or Int64 run ends");
            }
            let values = match r.data_type() {
                DataType::Int16 => a.as_run::<Int16Type>().values().clone(),
                DataType::Int32 => a.as_run::<Int32Type>().values().clone(),
                _ => a.as_run::<Int64Type>().values().clone(),
            };
            let v: Vec<_> = (0..values.len())
                .map(|i| val_at(values.as_ref(), i))
                .collect();
            if v.windows(2).any(|w| w[0] == w[1]) {
                out.insert("run-end encoded with a non-maximal run");
            }
            features(values.as_ref(), out);
        }
        dt => {
            // arrow-rs requires only `align_of` of the native type, which is
            // below the width for `DayTime`, `MonthDayNano` and `Decimal256`
            if let Some(w) = dt.primitive_width() {
                let d = a.to_data();
                let address = d.buffers()[0].as_ptr() as usize + d.offset() * w;
                if !address.is_multiple_of(w) {
                    out.insert("primitive values aligned below their width");
                }
            }
        }
    }
}

/// The numeric features, reported over integer, decimal and float arrays
pub const NUMERIC: &[&str] = &[
    "numeric: every value fits a narrower width",
    "numeric: one value just past a narrower width",
    "numeric: within one of 2^k, k not 7, 8, 15, 16, 31, 32, 63, 64",
    "numeric: within one of a power of ten, 10 or more",
    "numeric: Decimal256 value wider than i128, not a limit",
    "numeric: one limit of the type among small values",
    "float: every value integral and fits 32 bits",
    "float: one NaN, infinity or -0.0 among small values",
    "length at a 32- or 256-row boundary",
];

/// Whether `x` is within one of 2^k for a k that is not at the limit of a
/// narrower integer width, which the width edges already give
fn near_odd_power_of_two(x: &BigInt) -> bool {
    let m = x.abs();
    [&m - 1, m.clone(), &m + 1].iter().any(|y| {
        let k = y.bits().saturating_sub(1);
        k >= 3 && *y == BigInt::from(1) << k && ![7, 8, 15, 16, 31, 32, 63, 64].contains(&k)
    })
}

/// Whether `x` is within one of 10^k for k >= 1
fn near_power_of_ten(x: &BigInt) -> bool {
    let m = x.abs();
    [&m - 1, m.clone(), &m + 1].iter().any(|y| {
        let digits = y.to_string();
        digits.len() >= 2 && digits.starts_with('1') && digits[1..].bytes().all(|c| c == b'0')
    })
}

/// Features of the values of an integer, decimal or float array: what the
/// whole batch holds, as optimized numeric kernels see it
pub fn numeric_features(a: &dyn Array, out: &mut BTreeSet<&'static str>) {
    if matches!(a.len(), 31 | 32 | 33 | 255 | 256 | 257) {
        out.insert("length at a 32- or 256-row boundary");
    }
    let valid: Vec<usize> = (0..a.len()).filter(|&i| a.is_valid(i)).collect();
    if valid.len() < 2 {
        return;
    }
    let dt = a.data_type();
    if let Some(d) = domain_of(dt) {
        let v: Vec<BigInt> = valid
            .iter()
            .map(|&i| match val_at(a, i) {
                Val::I32(x) => BigInt::from(x),
                Val::I64(x) => BigInt::from(x),
                Val::Raw(b) if dt.is_unsigned_integer() => BigInt::from_bytes_le(Sign::Plus, &b),
                Val::Raw(b) => BigInt::from_signed_bytes_le(&b),
                v => panic!("not a number {v:?}"),
            })
            .collect();
        let fits = |n: &Domain, x: &BigInt| &n.lo <= x && x <= &n.hi;
        let widths = narrower(&d);
        if widths
            .iter()
            .any(|&b| v.iter().all(|x| fits(&narrow(&d, b), x)))
        {
            out.insert("numeric: every value fits a narrower width");
        }
        if widths.iter().any(|&b| {
            let n = narrow(&d, b);
            let past: Vec<&BigInt> = v.iter().filter(|x| !fits(&n, x)).collect();
            past.len() == 1 && (*past[0] == &n.hi + 1 || *past[0] == &n.lo - 1)
        }) {
            out.insert("numeric: one value just past a narrower width");
        }
        if v.iter().any(near_odd_power_of_two) {
            out.insert("numeric: within one of 2^k, k not 7, 8, 15, 16, 31, 32, 63, 64");
        }
        if v.iter().any(near_power_of_ten) {
            out.insert("numeric: within one of a power of ten, 10 or more");
        }
        if matches!(dt, DataType::Decimal256(..))
            && v.iter()
                .any(|x| x.to_i128().is_none() && *x != d.lo && *x != d.hi)
        {
            out.insert("numeric: Decimal256 value wider than i128, not a limit");
        }
        let small = narrow(&d, 8);
        let limit = |x: &BigInt| *x == d.lo || *x == d.hi;
        if v.iter().filter(|x| limit(x)).count() == 1
            && v.iter().all(|x| limit(x) || fits(&small, x))
        {
            out.insert("numeric: one limit of the type among small values");
        }
    } else if matches!(dt, DataType::Float32 | DataType::Float64) {
        let v: Vec<f64> = valid.iter().map(|&i| val_at(a, i).as_f64()).collect();
        if v.iter()
            .all(|x| x.fract() == 0.0 && x.abs() < 2f64.powi(31))
        {
            out.insert("float: every value integral and fits 32 bits");
        }
        let special = |x: &f64| !x.is_finite() || (*x == 0.0 && x.is_sign_negative());
        if v.iter().filter(|x| special(x)).count() == 1
            && v.iter()
                .all(|x| special(x) || (x.fract() == 0.0 && x.abs() < 128.0))
        {
            out.insert("float: one NaN, infinity or -0.0 among small values");
        }
    }
}

/// Features of a predicate for `filter`
pub fn mask_features(p: &BooleanArray, out: &mut BTreeSet<&'static str>) {
    let n = p.len();
    let kept = (0..n).filter(|&i| p.is_valid(i) && p.value(i)).count();
    if n > 0 && kept == n {
        out.insert("mask: all selected");
    }
    if n > 0 && kept == 0 {
        out.insert("mask: none selected");
    }
    if n > 0 && kept > n * 8 / 10 && kept < n {
        out.insert("mask: selectivity in (0.8, 1)");
    }
    if kept > 0 && kept <= n / 16 {
        out.insert("mask: 1 row to 1/16 of rows selected");
    }
    if p.null_count() > 0 {
        out.insert("mask: has nulls");
    }
    // A whole aligned 64-bit word selected, and one with 64 kept bits in two
    let words: Vec<usize> = (0..n / 64)
        .map(|w| {
            (w * 64..w * 64 + 64)
                .filter(|&i| p.is_valid(i) && p.value(i))
                .count()
        })
        .collect();
    if words.contains(&64) {
        out.insert("mask: a whole 64-bit word selected");
    }
    if words
        .windows(2)
        .any(|w| w[0] + w[1] == 64 && w[0] != 0 && w[1] != 0)
    {
        out.insert("mask: two words keeping exactly 64 bits");
    }
}

/// Features of an index array for `take`
pub fn index_features(idx: &Int32Array, n_values: usize, out: &mut BTreeSet<&'static str>) {
    if idx.is_empty() {
        out.insert("indices: empty");
    }
    if (0..idx.len())
        .any(|i| idx.is_null(i) && (idx.value(i) < 0 || idx.value(i) as usize >= n_values))
    {
        out.insert("indices: null hiding an out-of-bounds value");
    }
    let valid: Vec<i32> = idx.iter().flatten().collect();
    if valid.windows(2).any(|w| w[0] == w[1]) {
        out.insert("indices: repeated in a row");
    }
    if valid.windows(2).any(|w| w[1] == w[0] + 1) {
        out.insert("indices: consecutive run");
    }
}

pub fn report(
    name: &str,
    cases: usize,
    counts: &BTreeMap<&'static str, usize>,
    expected: &[&'static str],
) {
    println!("{name} ({cases} cases):");
    let mut keys: Vec<&'static str> = expected.to_vec();
    keys.extend(counts.keys().filter(|k| !expected.contains(k)));
    for k in keys {
        let c = counts.get(k).copied().unwrap_or(0);
        let flag = if c == 0 { "  <- never" } else { "" };
        println!("  {:5.1}%  {k}{flag}", 100.0 * c as f64 / cases as f64);
    }
}

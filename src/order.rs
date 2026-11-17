//! Properties for multi-column ordering: the row format and `lexsort`,
//! against an ordering oracle over the values `val.rs` reads

use std::cmp::Ordering;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, BinaryArray, StringArray, UInt32Array};
use arrow_row::{RowConverter, SortField};
use arrow_schema::{DataType, Field, SortOptions};
use proptest::test_runner::TestCaseError;

use crate::known::known;
use crate::props::R;
use crate::props::leaf_cmp;
use crate::props::{check_in, fail};
use crate::spec::check;
use crate::val::check_rows;
use crate::val::vals_keep_union_ids;
use crate::val::{Val, vals};
use arrow_row::Rows;
use std::mem::replace;

pub type SortCase = (Vec<ArrayRef>, Vec<SortOptions>, Option<usize>);

/// The order of two values of type `dt`. Sort options apply at every level
/// of nesting: nulls go where `nulls_first` says, and `descending` reverses
/// every comparison of non-null values, including list lengths and union
/// type ids. Lists compare element by element and then by length, structs
/// field by field, unions by type id and then by value, and dictionaries and
/// run-end encoded arrays by their values. Floats use IEEE totalOrder
///
/// With `union_bug`, unions follow the order the row format gives them
/// (apache/arrow-rs#11598): their values compare
/// ascending with nulls first when `nulls_first != descending`, and only the
/// type id follows `descending`. With values read by `vals_keep_union_ids`,
/// a null union row then sorts by its type id like any other
fn ord_with(dt: &DataType, a: &Val, b: &Val, o: SortOptions, union_bug: bool) -> Ordering {
    let ord_val =
        |dt: &DataType, a: &Val, b: &Val, o: SortOptions| ord_with(dt, a, b, o, union_bug);
    let null = if o.nulls_first {
        Ordering::Less
    } else {
        Ordering::Greater
    };
    let dir = |x: Ordering| if o.descending { x.reverse() } else { x };
    match (a, b) {
        (Val::Null, Val::Null) => return Ordering::Equal,
        (Val::Null, _) => return null,
        (_, Val::Null) => return null.reverse(),
        _ => {}
    }
    let elements = |child: &DataType, xs: &[Val], ys: &[Val]| {
        xs.iter()
            .zip(ys)
            .map(|(x, y)| ord_val(child, x, y, o))
            .find(|c| c.is_ne())
            .unwrap_or_else(|| dir(xs.len().cmp(&ys.len())))
    };
    match (dt, a, b) {
        (DataType::Dictionary(_, v), a, b) => ord_val(v, a, b, o),
        (DataType::RunEndEncoded(_, v), a, b) => ord_val(v.data_type(), a, b, o),
        (
            _,
            Val::Bool(_) | Val::I32(_) | Val::I64(_) | Val::Raw(_) | Val::Str(_) | Val::Bytes(_),
            _,
        ) => dir(leaf_cmp(dt, a, b)),
        (
            DataType::List(c)
            | DataType::LargeList(c)
            | DataType::ListView(c)
            | DataType::LargeListView(c)
            | DataType::FixedSizeList(c, _)
            | DataType::Map(c, _),
            Val::List(xs),
            Val::List(ys),
        ) => elements(c.data_type(), xs, ys),
        (DataType::Struct(fs), Val::Struct(xs), Val::Struct(ys)) => fs
            .iter()
            .zip(xs.iter().zip(ys))
            .map(|(f, (x, y))| ord_val(f.data_type(), x, y, o))
            .find(|c| c.is_ne())
            .unwrap_or(Ordering::Equal),
        (DataType::Union(fs, _), Val::Union(i, x), Val::Union(j, y)) => match i.cmp(j) {
            Ordering::Equal => {
                let child = fs.iter().find(|(id, _)| id == i).unwrap().1.data_type();
                if union_bug {
                    ord_val(
                        child,
                        x,
                        y,
                        SortOptions {
                            descending: false,
                            nulls_first: o.nulls_first != o.descending,
                        },
                    )
                } else {
                    ord_val(child, x, y, o)
                }
            }
            c => dir(c),
        },
        (dt, a, b) => panic!("no order for {dt}: {a:?}, {b:?}"),
    }
}

/// Lexicographic order of row `i` and row `j` over the columns
fn ord_rows(
    types: &[DataType],
    cols: &[Vec<Val>],
    opts: &[SortOptions],
    i: usize,
    j: usize,
) -> Ordering {
    ord_rows_with(types, cols, opts, i, j, false)
}

fn ord_rows_with(
    types: &[DataType],
    cols: &[Vec<Val>],
    opts: &[SortOptions],
    i: usize,
    j: usize,
    union_bug: bool,
) -> Ordering {
    (0..cols.len())
        .map(|c| ord_with(&types[c], &cols[c][i], &cols[c][j], opts[c], union_bug))
        .find(|c| c.is_ne())
        .unwrap_or(Ordering::Equal)
}

/// The rows of the columns in the oracle's order. Rows that compare equal
/// have equal values, so the sorted values do not depend on how ties break
fn oracle_sorted(
    types: &[DataType],
    cols: &[Vec<Val>],
    opts: &[SortOptions],
    n: usize,
) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| ord_rows(types, cols, opts, i, j));
    idx
}

/// `dt` with every dictionary replaced by its value type, which is what
/// `convert_rows` returns, as the arrow-row docs say
fn hydrated(dt: &DataType) -> DataType {
    let f = |f: &Arc<Field>| Arc::new(f.as_ref().clone().with_data_type(hydrated(f.data_type())));
    match dt {
        DataType::Dictionary(_, v) => hydrated(v),
        DataType::List(c) => DataType::List(f(c)),
        DataType::LargeList(c) => DataType::LargeList(f(c)),
        DataType::ListView(c) => DataType::ListView(f(c)),
        DataType::LargeListView(c) => DataType::LargeListView(f(c)),
        DataType::FixedSizeList(c, n) => DataType::FixedSizeList(f(c), *n),
        DataType::Map(c, sorted) => DataType::Map(f(c), *sorted),
        DataType::Struct(fs) => DataType::Struct(fs.iter().map(f).collect()),
        DataType::Union(fs, m) => {
            DataType::Union(fs.iter().map(|(id, x)| (id, f(x))).collect(), *m)
        }
        DataType::RunEndEncoded(r, v) => DataType::RunEndEncoded(r.clone(), f(v)),
        dt => dt.clone(),
    }
}

/// Rows compare as the oracle orders the columns, including rows converted
/// in separate calls, and `convert_rows` gives back the columns' values
pub fn row_format((cols, opts, split): SortCase) -> R {
    for c in &cols {
        check_in(c.as_ref())?;
    }
    let n = cols.first().map_or(0, |c| c.len());
    let types: Vec<DataType> = cols.iter().map(|c| c.data_type().clone()).collect();
    let v: Vec<Vec<Val>> = cols.iter().map(|c| vals(c.as_ref())).collect();
    let fields = types
        .iter()
        .zip(&opts)
        .map(|(t, o)| SortField::new_with_options(t.clone(), *o))
        .collect();
    let conv = RowConverter::new(fields).map_err(|e| fail(format!("RowConverter::new: {e}")))?;
    // Rows from two separate conversions of a split, compared across
    let k = split.unwrap_or(n / 2).min(n);
    let part = |from: usize, len: usize| -> Result<Rows, TestCaseError> {
        let cs: Vec<ArrayRef> = cols.iter().map(|c| c.slice(from, len)).collect();
        conv.convert_columns(&cs)
            .map_err(|e| fail(format!("convert_columns: {e}")))
    };
    let (head, tail) = (part(0, k)?, part(k, n - k)?);
    let row = |i: usize| if i < k { head.row(i) } else { tail.row(i - k) };
    let v_ids: Vec<Vec<Val>> = cols
        .iter()
        .map(|c| vals_keep_union_ids(c.as_ref()))
        .collect();
    let sorted = oracle_sorted(&types, &v, &opts, n);
    // Pairs that match the known union bug are skipped, and the case is
    // discarded at the end only if every other check passes
    let mut union_known = false;
    for w in sorted.windows(2) {
        let (want, got) = (
            ord_rows(&types, &v, &opts, w[0], w[1]),
            row(w[0]).cmp(&row(w[1])),
        );
        if want != got {
            if got == ord_rows_with(&types, &v_ids, &opts, w[0], w[1], true) {
                union_known = true;
                continue;
            }
            return Err(fail(format!(
                "rows {} and {}: compare {got:?}, expected {want:?}",
                w[0], w[1]
            )));
        }
    }
    // Pairs that are not neighbors in the order too
    for i in 0..n {
        let j = (i * 7 + 3) % n;
        let (want, got) = (ord_rows(&types, &v, &opts, i, j), row(i).cmp(&row(j)));
        if want != got {
            if got == ord_rows_with(&types, &v_ids, &opts, i, j, true) {
                union_known = true;
                continue;
            }
            return Err(fail(format!(
                "rows {i} and {j}: compare {got:?}, expected {want:?}"
            )));
        }
    }
    let all = conv
        .convert_columns(&cols)
        .map_err(|e| fail(format!("convert_columns: {e}")))?;
    let back = conv
        .convert_rows(&all)
        .map_err(|e| fail(format!("convert_rows: {e}")))?;
    for (c, b) in back.iter().enumerate() {
        b.to_data()
            .validate_full()
            .map_err(|e| fail(format!("convert_rows column {c} invalid: {e}")))?;
        check(&b.to_data())
            .map_err(|e| fail(format!("convert_rows column {c} violates the spec: {e}")))?;
        if b.data_type() != &hydrated(&types[c]) {
            return Err(fail(format!(
                "convert_rows column {c}: type {} for {}",
                b.data_type(),
                types[c]
            )));
        }
        if vals(b.as_ref()) != v[c] {
            let got = vals(b.as_ref());
            let i = (0..n).find(|&i| got[i] != v[c][i]).unwrap_or(n);
            return Err(fail(format!(
                "convert_rows column {c} row {i}: got {:?}, expected {:?}",
                got.get(i),
                v[c].get(i)
            )));
        }
    }
    if union_known {
        return Err(known(
            "known: row format orders unions unlike lexsort (apache/arrow-rs#11598)",
        ));
    }
    Ok(())
}

/// The bytes of each row of `a` as `arrow-row` documents them (lib.rs for
/// integers and booleans, variable.rs for strings and binary), or `None` for
/// a type this model does not cover. Floats, intervals, nested types and
/// dictionaries are not covered.
///
/// Integers and booleans are a validity byte and the big-endian value with
/// the sign bit flipped for signed types. A string or binary value is a
/// sentinel, 8-byte mini-blocks up to 32 bytes and 32-byte blocks after, each
/// padded with zeros and followed by `0xFF`, or by its length if it is the
/// last. Descending inverts every byte of a valid value but not a null.
fn model_rows(a: &dyn Array, o: SortOptions) -> Option<Vec<Vec<u8>>> {
    use DataType as D;
    let null_byte = if o.nulls_first { 0u8 } else { 0xFF };
    let invert = |mut v: Vec<u8>| {
        if o.descending {
            v.iter_mut().for_each(|b| *b = !*b);
        }
        v
    };
    let dt = a.data_type();
    let signed = matches!(
        dt,
        D::Int8
            | D::Int16
            | D::Int32
            | D::Int64
            | D::Date32
            | D::Date64
            | D::Time32(_)
            | D::Time64(_)
            | D::Timestamp(..)
            | D::Duration(_)
            | D::Decimal128(..)
            | D::Decimal256(..)
    );
    let unsigned = matches!(
        dt,
        D::UInt8 | D::UInt16 | D::UInt32 | D::UInt64 | D::Boolean
    );
    let variable = matches!(
        dt,
        D::Utf8 | D::LargeUtf8 | D::Utf8View | D::Binary | D::LargeBinary | D::BinaryView
    );
    let width = if *dt == D::Boolean {
        1
    } else {
        dt.primitive_width().unwrap_or(0)
    };
    let mut out = Vec::with_capacity(a.len());
    for v in vals(a) {
        let row = match v {
            Val::Null if variable => vec![null_byte],
            Val::Null if signed || unsigned => {
                let mut r = vec![null_byte];
                r.resize(1 + width, 0);
                r
            }
            _ if signed || unsigned => {
                let mut be = match v {
                    Val::Bool(b) => vec![b as u8],
                    Val::I32(x) => x.to_le_bytes().to_vec(),
                    Val::I64(x) => x.to_le_bytes().to_vec(),
                    Val::Raw(b) => b,
                    _ => return None,
                };
                be.reverse();
                if signed {
                    be[0] ^= 0x80;
                }
                let mut r = vec![1];
                r.extend(be);
                // The validity byte is not inverted
                let mut inverted = invert(r.split_off(1));
                r.append(&mut inverted);
                r
            }
            Val::Str(s) if variable => model_variable(s.as_bytes(), o),
            Val::Bytes(b) if variable => model_variable(&b, o),
            _ => return None,
        };
        out.push(row);
    }
    Some(out)
}

fn model_variable(v: &[u8], o: SortOptions) -> Vec<u8> {
    let mut out = vec![if v.is_empty() { 1 } else { 2 }];
    let (mini, rest) = v.split_at(v.len().min(32));
    for (blocks, size, more) in [(mini, 8, !rest.is_empty()), (rest, 32, false)] {
        let count = blocks.len().div_ceil(size);
        for (i, block) in blocks.chunks(size).enumerate() {
            out.extend_from_slice(block);
            out.resize(out.len() + size - block.len(), 0);
            let last = i + 1 == count && !more;
            out.push(if last { block.len() as u8 } else { 0xFF });
        }
    }
    if o.descending {
        out.iter_mut().for_each(|b| *b = !*b);
    }
    out
}

/// Each row equals the model's bytes, column by column and for the whole row
/// when the model covers every column, and a view or large array encodes the
/// bytes of the same values in a `Utf8` or `Binary` array
fn row_bytes_match_format(cols: &[ArrayRef], opts: &[SortOptions], all: &arrow_row::Rows) -> R {
    let n = cols.first().map_or(0, |c| c.len());
    let models: Vec<Option<Vec<Vec<u8>>>> = cols
        .iter()
        .zip(opts)
        .map(|(c, o)| model_rows(c.as_ref(), *o))
        .collect();
    let converter = |c: &ArrayRef, o: SortOptions| {
        RowConverter::new(vec![SortField::new_with_options(c.data_type().clone(), o)])
            .map_err(|e| fail(format!("RowConverter::new: {e}")))
    };
    for (c, (col, model)) in cols.iter().zip(&models).enumerate() {
        let Some(model) = model else { continue };
        let rows = converter(col, opts[c])?
            .convert_columns(&[Arc::clone(col)])
            .map_err(|e| fail(format!("convert_columns: {e}")))?;
        for (i, want) in model.iter().enumerate() {
            if rows.row(i).as_ref() != want.as_slice() {
                return Err(fail(format!(
                    "column {c} ({}) row {i} of {n}, {:?}: {:?}, the format says {want:?}",
                    col.data_type(),
                    vals(col.as_ref())[i],
                    rows.row(i).as_ref()
                )));
            }
        }
        // The same values in the array type that does not use views or
        // 64-bit offsets
        let plain: Option<ArrayRef> = match col.data_type() {
            DataType::LargeUtf8 | DataType::Utf8View => Some(Arc::new(StringArray::from_iter(
                vals(col.as_ref()).iter().map(|v| match v {
                    Val::Str(s) => Some(s.as_str()),
                    _ => None,
                }),
            ))),
            DataType::LargeBinary | DataType::BinaryView => Some(Arc::new(BinaryArray::from_iter(
                vals(col.as_ref()).iter().map(|v| match v {
                    Val::Bytes(b) => Some(b.as_slice()),
                    _ => None,
                }),
            ))),
            _ => None,
        };
        if let Some(plain) = plain {
            let rows_plain = converter(&plain, opts[c])?
                .convert_columns(&[plain])
                .map_err(|e| fail(format!("convert_columns: {e}")))?;
            for i in 0..n {
                if rows.row(i).as_ref() != rows_plain.row(i).as_ref() {
                    return Err(fail(format!(
                        "column {c} ({}) row {i} of {n}: {:?}, {:?} as Utf8 or Binary",
                        col.data_type(),
                        rows.row(i).as_ref(),
                        rows_plain.row(i).as_ref()
                    )));
                }
            }
        }
    }
    if models.iter().all(Option::is_some) {
        for i in 0..n {
            let want: Vec<u8> = models.iter().flatten().flat_map(|m| m[i].clone()).collect();
            if all.row(i).as_ref() != want.as_slice() {
                return Err(fail(format!(
                    "row {i} of {n}: {:?}, the format says {want:?}",
                    all.row(i).as_ref()
                )));
            }
        }
    }
    Ok(())
}

/// Each row's bytes depend on the row alone: row `i` of one conversion equals
/// the row from converting that row by itself, the rows of a conversion of
/// a slice at a nonzero offset equal the same rows of the whole, and rows
/// appended to existing rows equal the rows of one conversion. This checks
/// the bytes, which the order and the round trip in `row_format` do not, and
/// that fixed-width columns encode the same bytes in the block-at-a-time
/// encoder and the general one. Rows also equal the bytes `model_rows` writes
/// from the format's documentation, and view and large arrays equal the same
/// values in a `Utf8` or `Binary` array
pub fn row_encode((cols, opts, split): SortCase) -> R {
    let n = cols.first().map_or(0, |c| c.len());
    let fields = cols
        .iter()
        .zip(&opts)
        .map(|(c, o)| SortField::new_with_options(c.data_type().clone(), *o))
        .collect();
    let conv = RowConverter::new(fields).map_err(|e| fail(format!("RowConverter::new: {e}")))?;
    let convert = |cs: &[ArrayRef]| {
        conv.convert_columns(cs)
            .map_err(|e| fail(format!("convert_columns: {e}")))
    };
    let slices = |from: usize, len: usize| -> Vec<ArrayRef> {
        cols.iter().map(|c| c.slice(from, len)).collect()
    };
    let all = convert(&cols)?;
    for i in 0..n {
        let one = convert(&slices(i, 1))?;
        if all.row(i) != one.row(0) {
            return Err(fail(format!(
                "row {i} of {n}: {:?} in one conversion, {:?} converted alone",
                all.row(i).as_ref(),
                one.row(0).as_ref()
            )));
        }
    }
    // When every column is fixed width, a Utf8 column after them makes the
    // rows variable length, which takes the general encoder. A row's bytes
    // are its columns' bytes in order, so the fixed columns' bytes must not
    // change
    if cols
        .iter()
        .all(|c| c.data_type().is_primitive() || c.data_type() == &DataType::Boolean)
    {
        let mut with_text = cols.clone();
        with_text.push(Arc::new(StringArray::from(vec![""; n])));
        let mut fields: Vec<SortField> = cols
            .iter()
            .zip(&opts)
            .map(|(c, o)| SortField::new_with_options(c.data_type().clone(), *o))
            .collect();
        fields.push(SortField::new(DataType::Utf8));
        let general = RowConverter::new(fields)
            .map_err(|e| fail(format!("RowConverter::new: {e}")))?
            .convert_columns(&with_text)
            .map_err(|e| fail(format!("convert_columns: {e}")))?;
        for i in 0..n {
            let (fixed_row, general_row) = (all.row(i), general.row(i));
            let (fixed, prefix) = (fixed_row.as_ref(), general_row.as_ref());
            if prefix.get(..fixed.len()) != Some(fixed) {
                return Err(fail(format!(
                    "row {i} of {n}: {fixed:?} for fixed-width columns, {prefix:?} with a Utf8 column after them"
                )));
            }
        }
    }
    row_bytes_match_format(&cols, &opts, &all)?;
    let k = split.unwrap_or(n / 2).min(n);
    let (head, tail) = (convert(&slices(0, k))?, convert(&slices(k, n - k))?);
    let mut appended = conv.empty_rows(0, 0);
    conv.append(&mut appended, &slices(0, k))
        .map_err(|e| fail(format!("append: {e}")))?;
    conv.append(&mut appended, &slices(k, n - k))
        .map_err(|e| fail(format!("append: {e}")))?;
    if appended.num_rows() != n || head.num_rows() != k || tail.num_rows() != n - k {
        return Err(fail(format!("row counts for {n} rows split at {k}")));
    }
    for i in 0..n {
        let part = if i < k { head.row(i) } else { tail.row(i - k) };
        if part != all.row(i) {
            return Err(fail(format!(
                "row {i} of {n}: converting a slice split at {k} gives different bytes"
            )));
        }
        if appended.row(i) != all.row(i) {
            return Err(fail(format!(
                "row {i} of {n}: appending after {k} rows gives different bytes"
            )));
        }
    }
    let back = conv
        .convert_rows(&appended)
        .map_err(|e| fail(format!("convert_rows: {e}")))?;
    for (c, b) in back.iter().enumerate() {
        if vals(b.as_ref()) != vals(cols[c].as_ref()) {
            return Err(fail(format!("convert_rows of appended rows, column {c}")));
        }
    }
    Ok(())
}

/// `lexsort_to_indices` returns the first `limit` rows in the oracle's order,
/// and `lexsort` returns those rows' values
pub fn lexsort_prop((cols, opts, limit): SortCase) -> R {
    use arrow_ord::sort::{SortColumn, lexsort, lexsort_to_indices};
    for c in &cols {
        check_in(c.as_ref())?;
    }
    let n = cols.first().map_or(0, |c| c.len());
    let types: Vec<DataType> = cols.iter().map(|c| c.data_type().clone()).collect();
    let v: Vec<Vec<Val>> = cols.iter().map(|c| vals(c.as_ref())).collect();
    let sort_cols: Vec<SortColumn> = cols
        .iter()
        .zip(&opts)
        .map(|(c, o)| SortColumn {
            values: c.clone(),
            options: Some(*o),
        })
        .collect();
    let want = oracle_sorted(&types, &v, &opts, n);
    let want = &want[..limit.unwrap_or(n).min(n)];
    let idx: UInt32Array = lexsort_to_indices(&sort_cols, limit)
        .map_err(|e| fail(format!("lexsort_to_indices: {e}")))?;
    if idx.len() != want.len() || idx.null_count() > 0 {
        return Err(fail(format!(
            "lexsort_to_indices: {} indices with {} nulls, expected {}",
            idx.len(),
            idx.null_count(),
            want.len()
        )));
    }
    let mut seen = vec![false; n];
    for (p, (&w, g)) in want.iter().zip(idx.values().iter()).enumerate() {
        if (*g as usize) < n && replace(&mut seen[*g as usize], true) {
            return Err(fail(format!(
                "lexsort_to_indices position {p}: row {g} repeated"
            )));
        }
        let g = *g as usize;
        if g >= n || (0..cols.len()).any(|c| v[c][g] != v[c][w]) {
            return Err(fail(format!(
                "lexsort_to_indices position {p}: row {g}, expected a row equal to {w}"
            )));
        }
    }
    let sorted = lexsort(&sort_cols, limit).map_err(|e| fail(format!("lexsort: {e}")))?;
    for (c, s) in sorted.iter().enumerate() {
        let expected: Vec<Val> = want.iter().map(|&w| v[c][w].clone()).collect();
        check_rows(s, &types[c], &expected)
            .map_err(|e| fail(format!("lexsort column {c}: {e:?}")))?;
    }
    Ok(())
}

//! Properties of the selection, comparison, sort, arithmetic and cast kernels,
//! the laws between kernels, and the IPC, FFI and `ArrayData` round trips,
//! with the property runner. Each compares a kernel's output with the
//! row-by-row values the oracle in `val.rs` reads from the inputs.

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BooleanArray, Int32Array, RecordBatch, make_array, new_null_array,
};
use arrow_schema::{ArrowError, DataType, Field, FieldRef, Schema};
use proptest::test_runner::TestCaseError;

use crate::run::{caught, panic_message};
use crate::val::{Val, check_rows, same_float, vals};

pub type Interleave = (Vec<ArrayRef>, Vec<(usize, usize)>);

pub(crate) type R = Result<(), TestCaseError>;

pub(crate) fn fail(msg: impl Into<String>) -> TestCaseError {
    TestCaseError::fail(msg.into())
}

/// An error from a kernel is a failure, unless the kernel also rejects a
/// null array of the same type, which means the arrow-rs under test does
/// not support the type, so the case is discarded
pub(crate) fn kernel_err(
    e: ArrowError,
    dt: &DataType,
    probe: impl Fn(&ArrayRef) -> Result<ArrayRef, ArrowError>,
) -> TestCaseError {
    let msg = e.to_string();
    if msg.contains("Dictionary key bigger than the key type") && any_type(dt, &is_view_dict) {
        return known(
            "known: dictionaries of views are not merged, so keys overflow (apache/arrow-rs#10925)",
        );
    }
    let unsupported = matches!(e, ArrowError::NotYetImplemented(_))
        || msg.to_lowercase().contains("not supported")
        || msg.to_lowercase().contains("unsupported");
    if unsupported && probe(&new_null_array(dt, 2)).is_err() {
        return TestCaseError::reject(format!("unsupported: {msg}"));
    }
    fail(format!("kernel error: {msg}"))
}

/// Writes `a` as a one-column IPC stream and reads it back. `None` when IPC
/// does not support the type in the arrow-rs under test
fn ipc_round_trip(a: &ArrayRef) -> Result<Option<ArrayRef>, String> {
    use arrow_ipc::reader::StreamReader;
    use arrow_ipc::writer::StreamWriter;
    let schema = Arc::new(Schema::new(vec![Field::new(
        "v",
        a.data_type().clone(),
        true,
    )]));
    let batch = RecordBatch::try_new(schema.clone(), vec![a.clone()]).map_err(|e| e.to_string())?;
    let mut bytes = vec![];
    let written = StreamWriter::try_new(&mut bytes, &schema).and_then(|mut w| {
        w.write(&batch)?;
        w.finish()
    });
    if let Err(e) = written {
        let msg = e.to_string();
        let unsupported = matches!(e, ArrowError::NotYetImplemented(_))
            || msg.to_lowercase().contains("not supported")
            || msg.to_lowercase().contains("unsupported");
        return if unsupported {
            Ok(None)
        } else {
            Err(format!("IPC write: {msg}"))
        };
    }
    let mut reader =
        StreamReader::try_new(Cursor::new(bytes), None).map_err(|e| format!("IPC read: {e}"))?;
    let back = reader
        .next()
        .ok_or("IPC read: no batch")?
        .map_err(|e| format!("IPC read: {e}"))?;
    Ok(Some(back.column(0).clone()))
}

/// Exports `a` through the C Data Interface and imports it back
fn ffi_round_trip(a: &ArrayRef) -> Result<ArrayRef, String> {
    let (array, schema) = to_ffi(&a.to_data()).map_err(|e| format!("FFI export: {e}"))?;
    // SAFETY: `array` and `schema` were exported just above and are imported once
    let data = unsafe { from_ffi(array, &schema) }.map_err(|e| format!("FFI import: {e}"))?;
    Ok(make_array(data))
}

/// Inputs must be valid per the spec; an invalid one is a generator bug
pub(crate) fn check_in(a: &dyn Array) -> R {
    match check(&a.to_data()) {
        Ok(()) => Ok(()),
        Err(e) => Err(fail(format!("invalid input (generator bug): {e}"))),
    }
}

/// IPC write then read gives the same rows, for a slice of a generated array
pub fn round_trip_ipc((a, offset, len): (ArrayRef, usize, usize)) -> R {
    check_in(a.as_ref())?;
    let s = a.slice(offset, len);
    match ipc_round_trip(&s) {
        Ok(Some(back)) => check_rows(&back, s.data_type(), &vals(s.as_ref())),
        Ok(None) => Err(TestCaseError::reject("IPC does not support this type")),
        Err(e) => Err(fail(e)),
    }
}

/// FFI export then import gives the same rows, for a slice of a generated array
pub fn round_trip_ffi((a, offset, len): (ArrayRef, usize, usize)) -> R {
    check_in(a.as_ref())?;
    let s = a.slice(offset, len);
    let back = ffi_round_trip(&s).map_err(fail)?;
    check_rows(&back, s.data_type(), &vals(s.as_ref()))
}

/// Runs a kernel, turning the panic form of a known bug into a discard
pub(crate) fn guarded(
    dt: &DataType,
    f: impl FnOnce() -> Result<ArrayRef, ArrowError>,
) -> Result<Result<ArrayRef, ArrowError>, TestCaseError> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|p| known_panic(dt, &panic_message(&*p)))
}

/// The discard for a panic that is a known bug's on arrays of `dt`, or else
/// the failure
pub(crate) fn known_panic(dt: &DataType, msg: &str) -> TestCaseError {
    if msg.contains("DictionaryKeyOverflowError") && any_type(dt, &is_view_dict) {
        return known(
            "known: dictionaries of views are not merged, so keys overflow (apache/arrow-rs#10925)",
        );
    }
    // interleave of a list view copies its child with MutableArrayData::new,
    // which concatenates dictionaries without merging them and panics when
    // the key type overflows. A dictionary anywhere below the list view
    // reaches it. Found by this harness on main at 152f0d33c
    // (apache/arrow-rs#11620)
    let is_dict = |d: &DataType| matches!(d, DataType::Dictionary(..));
    let is_list_view_of_dict = |d: &DataType| {
        matches!(d, DataType::ListView(c) | DataType::LargeListView(c)
            if any_type(c.data_type(), &is_dict))
    };
    if msg.contains("DictionaryKeyOverflowError") && any_type(dt, &is_list_view_of_dict) {
        return known(
            "known: interleave of a list view of dictionaries panics on key overflow (apache/arrow-rs#11620)",
        );
    }
    fail(format!("panic: {msg}"))
}

/// Whether an array of `dt` can hold a null at its top level. A union holds
/// one only through a child whose field may be null, and a run-end encoded
/// array only through a values field that may be null
pub fn can_hold_null(dt: &DataType) -> bool {
    match dt {
        DataType::RunEndEncoded(_, v) => v.is_nullable() && can_hold_null(v.data_type()),
        DataType::Union(fs, _) => fs
            .iter()
            .any(|(_, f)| f.is_nullable() && can_hold_null(f.data_type())),
        _ => true,
    }
}

pub fn take((values, indices): (ArrayRef, Int32Array)) -> R {
    use arrow_select::take::{TakeOptions, take, take_arrays};
    check_in(values.as_ref())?;
    check_in(&indices)?;
    // A null index needs a null in the output, which this type cannot hold
    if indices.null_count() > 0 && !can_hold_null(values.data_type()) {
        return match take(values.as_ref(), &indices, None) {
            Err(_) => Ok(()),
            Ok(out) => Err(fail(format!(
                "take of a null index into {} returned {} nulls",
                values.data_type(),
                out.logical_null_count()
            ))),
        };
    }
    let checked = Some(TakeOptions { check_bounds: true });
    // A valid index outside 0..len is an error when bounds are checked, and
    // a documented panic when they are not, so only the checked call runs
    let n = values.len() as i64;
    if let Some(i) = (0..indices.len())
        .find(|&i| indices.is_valid(i) && !(0..n).contains(&(indices.value(i) as i64)))
    {
        return match catch_unwind(AssertUnwindSafe(|| {
            take(values.as_ref(), &indices, checked.clone())
        })) {
            Ok(Err(_)) => Ok(()),
            Ok(Ok(_)) => Err(fail(format!(
                "take with checked bounds returned a result for index {} of {n} rows",
                indices.value(i)
            ))),
            Err(_) => Err(fail(format!(
                "take with checked bounds panicked for index {} of {n} rows",
                indices.value(i)
            ))),
        };
    }
    let v = vals(values.as_ref());
    let expected: Vec<Val> = (0..indices.len())
        .map(|i| {
            if indices.is_valid(i) {
                v[indices.value(i) as usize].clone()
            } else {
                Val::Null
            }
        })
        .collect();
    let run = |a: &ArrayRef| take(a.as_ref(), &indices, None);
    let out = run(&values).map_err(|e| kernel_err(e, values.data_type(), run))?;
    check_rows(&out, values.data_type(), &expected)?;
    maximal_runs(&out)?;
    // The other entry points give the same rows
    let out = take(values.as_ref(), &indices, checked)
        .map_err(|e| fail(format!("take with checked bounds: {e}")))?;
    check_rows(&out, values.data_type(), &expected)?;
    let outs = take_arrays(&[values.clone(), values.clone()], &indices, None)
        .map_err(|e| fail(format!("take_arrays: {e}")))?;
    if outs.len() != 2 {
        return Err(fail(format!(
            "take_arrays returned {} arrays for 2",
            outs.len()
        )));
    }
    outs.iter()
        .try_for_each(|o| check_rows(o, values.data_type(), &expected))
}

/// A run-end encoded result of `take` merges adjacent runs of equal values,
/// nulls included. This is arrow-rs's intent as `take_run`'s comment and
/// tests state it, not a documented public contract: the format allows
/// runs that are not maximal
fn maximal_runs(out: &ArrayRef) -> R {
    if !matches!(out.data_type(), DataType::RunEndEncoded(..)) {
        return Ok(());
    }
    let values = make_array(out.to_data().child_data()[1].clone());
    let v = vals(values.as_ref());
    match (1..v.len()).find(|&i| v[i] == v[i - 1]) {
        Some(i) => Err(fail(format!(
            "take on run-end encoded: runs {} and {i} hold the same value {:?}",
            i - 1,
            v[i]
        ))),
        None => Ok(()),
    }
}

pub fn filter((values, predicate): (ArrayRef, BooleanArray)) -> R {
    use arrow_select::filter::{FilterBuilder, filter, filter_record_batch};
    check_in(values.as_ref())?;
    check_in(&predicate)?;
    let v = vals(values.as_ref());
    let kept: Vec<usize> = (0..predicate.len())
        .filter(|&i| predicate.is_valid(i) && predicate.value(i))
        .collect();
    let expected: Vec<Val> = kept.iter().map(|&i| v[i].clone()).collect();
    let dt = values.data_type();
    let run = |a: &ArrayRef| filter(a.as_ref(), &predicate);
    let out = run(&values).map_err(|e| kernel_err(e, dt, run))?;
    check_rows(&out, dt, &expected)?;

    // The other entry points: an optimized predicate, its null filtering, and
    // a record batch with a second column
    let p = FilterBuilder::new(&predicate).optimize().build();
    let out = p
        .filter(values.as_ref())
        .map_err(|e| fail(format!("FilterPredicate::filter: {e}")))?;
    check_rows(&out, dt, &expected)?;
    if let Some(nulls) = values.nulls() {
        let got: Option<Vec<bool>> = p.filter_nulls(Some(nulls)).map(|n| n.iter().collect());
        let want: Vec<bool> = kept.iter().map(|&i| nulls.is_valid(i)).collect();
        let got = got.unwrap_or_else(|| vec![true; kept.len()]);
        if got != want {
            return Err(fail(format!(
                "filter_nulls: got {got:?}, expected {want:?}"
            )));
        }
    }
    let ids: ArrayRef = Arc::new(Int32Array::from_iter_values(0..values.len() as i32));
    let schema = Arc::new(Schema::new(vec![
        Field::new("v", dt.clone(), true),
        Field::new("id", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(schema, vec![values.clone(), ids])
        .map_err(|e| TestCaseError::reject(format!("batch: {e}")))?;
    let out = filter_record_batch(&batch, &predicate)
        .map_err(|e| fail(format!("filter_record_batch: {e}")))?;
    check_rows(out.column(0), dt, &expected)?;
    let want_ids: Vec<Val> = kept.iter().map(|&i| Val::I32(i as i32)).collect();
    check_rows(out.column(1), &DataType::Int32, &want_ids)
}

pub fn interleave((arrays, indices): Interleave) -> R {
    arrays.iter().try_for_each(|a| check_in(a.as_ref()))?;
    let vs: Vec<Vec<Val>> = arrays.iter().map(|a| vals(a.as_ref())).collect();
    let expected: Vec<Val> = indices.iter().map(|&(a, r)| vs[a][r].clone()).collect();
    let refs: Vec<&dyn Array> = arrays.iter().map(|a| a.as_ref()).collect();
    let dt = arrays[0].data_type();
    let out =
        guarded(dt, || arrow_select::interleave::interleave(&refs, &indices))?.map_err(|e| {
            kernel_err(e, dt, |a| {
                arrow_select::interleave::interleave(&[a.as_ref()], &[(0, 0)])
            })
        })?;
    check_rows(&out, dt, &expected)
}

pub fn concat(arrays: Vec<ArrayRef>) -> R {
    arrays.iter().try_for_each(|a| check_in(a.as_ref()))?;
    let expected: Vec<Val> = arrays.iter().flat_map(|a| vals(a.as_ref())).collect();
    let refs: Vec<&dyn Array> = arrays.iter().map(|a| a.as_ref()).collect();
    let dt = arrays[0].data_type();
    let out = guarded(dt, || arrow_select::concat::concat(&refs))?.map_err(|e| {
        kernel_err(e, dt, |a| {
            arrow_select::concat::concat(&[a.as_ref(), a.as_ref()])
        })
    })?;
    check_rows(&out, dt, &expected)
}

/// `shift` moves every row `offset` places, right when positive, and fills
/// the rows left behind with nulls
pub fn shift((a, offset): (ArrayRef, i64)) -> R {
    check_in(a.as_ref())?;
    let v = vals(a.as_ref());
    let n = v.len() as i64;
    let expected: Vec<Val> = (0..n)
        .map(|i| match i.checked_sub(offset) {
            Some(j) if (0..n).contains(&j) => v[j as usize].clone(),
            _ => Val::Null,
        })
        .collect();
    let dt = a.data_type();
    let r = guarded(dt, || arrow_select::window::shift(a.as_ref(), offset))?;
    // A null this type cannot hold is an error, as apache/arrow-rs#10992
    // settled for every kernel that introduces nulls
    if !can_hold_null(dt) && expected.contains(&Val::Null) {
        let r = match r {
            Err(_) => Ok(()),
            Ok(out) => Err(fail(format!(
                "shift by {offset} put {} nulls in {dt}, which cannot hold one",
                out.logical_null_count()
            ))),
        };
        // Known: `shift` fills with `new_null_array`, which writes the nulls
        // into a union's or run-end encoded array's non-nullable children.
        // Found by this harness on main at 01a991765 (apache/arrow-rs#11835)
        if let Err(TestCaseError::Fail(m)) = &r
            && m.to_string().contains("which cannot hold one")
        {
            return Err(known(
                "known: shift puts nulls in fields that cannot hold one (apache/arrow-rs#11835)",
            ));
        }
        return r;
    }
    let out = r.map_err(|e| kernel_err(e, dt, |z| arrow_select::window::shift(z.as_ref(), 1)))?;
    check_rows(&out, dt, &expected).map_err(|e| fail(format!("shift by {offset}: {e:?}")))
}

/// `union_extract_by_id` returns the target field's value where a row
/// selects it and null elsewhere, and an error when no field has the type
/// id. `union_extract` does the same by name, where a repeated name may
/// select any field that has it
pub fn union_extract((a, target, by_name): (ArrayRef, i8, bool)) -> R {
    use arrow_array::cast::AsArray;
    use arrow_select::union_extract::{union_extract, union_extract_by_id};
    check_in(a.as_ref())?;
    let u = a.as_union();
    let DataType::Union(fs, _) = a.data_type() else {
        return Err(fail(format!("not a union: {}", a.data_type())));
    };
    let Some((_, field)) = fs.iter().find(|(id, _)| *id == target) else {
        let r = if by_name {
            union_extract(u, "missing")
        } else {
            union_extract_by_id(u, target)
        };
        return match r {
            Err(_) => Ok(()),
            Ok(_) => Err(fail(format!(
                "union_extract of missing type id {target} returned a result"
            ))),
        };
    };
    let v = vals(a.as_ref());
    let expected = |id: i8| -> Vec<Val> {
        (0..u.len())
            .map(|i| match &v[i] {
                Val::Union(t, x) if *t == id => (**x).clone(),
                _ => Val::Null,
            })
            .collect()
    };
    let dt = field.data_type();
    let r = guarded(dt, || union_extract_by_id(u, target))?;
    let want = expected(target);
    // A null this type cannot hold is an error, as apache/arrow-rs#10992
    // settled for every kernel that introduces nulls
    if !can_hold_null(dt) && want.contains(&Val::Null) {
        let r = match r {
            Err(_) => Ok(()),
            Ok(out) => Err(fail(format!(
                "union_extract put {} nulls in {dt}, which cannot hold one",
                out.logical_null_count()
            ))),
        };
        // Known: `union_extract` fills unselected rows with `new_null_array`
        // when the target cannot hold a null mask, as `shift` does. Found by
        // this harness on main at 01a991765 (apache/arrow-rs#11835)
        if let Err(TestCaseError::Fail(m)) = &r
            && m.to_string().contains("which cannot hold one")
        {
            return Err(known(
                "known: union_extract puts nulls in fields that cannot hold one (apache/arrow-rs#11835)",
            ));
        }
        return r;
    }
    let out = r.map_err(|e| fail(format!("union_extract_by_id: {e}")))?;
    check_rows(&out, dt, &want).map_err(|e| fail(format!("union_extract_by_id: {e:?}")))?;
    if by_name {
        let r = guarded(dt, || union_extract(u, field.name()))?;
        let same: Vec<_> = fs
            .iter()
            .filter(|(_, f)| f.name() == field.name())
            .collect();
        // The name may resolve to another field of that name, whose type may
        // not hold the nulls of the rows that select another field. That is
        // an error, as for the type id above
        let matches = |id: i8, f: &FieldRef| {
            let want = expected(id);
            match &r {
                Ok(out) => check_rows(out, f.data_type(), &want).is_ok(),
                Err(_) => !can_hold_null(f.data_type()) && want.contains(&Val::Null),
            }
        };
        if !same.iter().any(|(id, f)| matches(*id, f)) {
            let ids: Vec<i8> = same.iter().map(|(id, _)| *id).collect();
            return Err(fail(format!(
                "union_extract of {:?} matches none of type ids {ids:?}: {:?}",
                field.name(),
                r.map(|out| check_rows(&out, dt, &want))
            )));
        }
    }
    Ok(())
}

pub fn slice((a, offset, len): (ArrayRef, usize, usize)) -> R {
    check_in(a.as_ref())?;
    let v = vals(a.as_ref());
    let expected = &v[offset..offset + len];
    check_rows(&a.slice(offset, len), a.data_type(), expected)?;
    let data = a.to_data().slice(offset, len);
    check_rows(&make_array(data), a.data_type(), expected)
}

/// `ArrayData` equality treats a null key and a key pointing at a null value
/// as different, which the oracle does not, so for those types only check
/// that "equal" implies equal values
pub fn physical_nulls_differ(dt: &DataType) -> bool {
    match dt {
        DataType::Dictionary(..) | DataType::RunEndEncoded(..) | DataType::Union(..) => true,
        DataType::List(f)
        | DataType::LargeList(f)
        | DataType::ListView(f)
        | DataType::LargeListView(f)
        | DataType::FixedSizeList(f, _)
        | DataType::Map(f, _) => physical_nulls_differ(f.data_type()),
        DataType::Struct(fs) => fs.iter().any(|f| physical_nulls_differ(f.data_type())),
        _ => false,
    }
}

pub fn eq((x, y): (ArrayRef, ArrayRef)) -> R {
    check_in(x.as_ref())?;
    check_in(y.as_ref())?;
    let (vx, vy) = (vals(x.as_ref()), vals(y.as_ref()));
    let same = vx == vy;
    let data_eq = x.to_data() == y.to_data();
    if data_eq && !same {
        return Err(fail("ArrayData equality says equal, values differ"));
    }
    if same && !data_eq && !physical_nulls_differ(x.data_type()) {
        return Err(fail("values equal, ArrayData equality says different"));
    }
    // Law: equality is reflexive and symmetric
    if x.to_data() != x.slice(0, x.len()).to_data() {
        return Err(fail("ArrayData equality is not reflexive"));
    }
    if data_eq != (y.to_data() == x.to_data()) {
        return Err(fail("ArrayData equality is not symmetric"));
    }
    // `cmp::eq` compares floats by IEEE total order, which is equality of
    // bits, so the oracle's byte comparison is the expected result
    let leaf = x.data_type().is_primitive()
        || matches!(
            x.data_type(),
            DataType::Boolean
                | DataType::Int32
                | DataType::Int64
                | DataType::Utf8
                | DataType::LargeUtf8
                | DataType::Utf8View
                | DataType::Binary
                | DataType::BinaryView
                | DataType::FixedSizeBinary(_)
        );
    if leaf && x.len() == y.len() {
        let out = arrow_ord::cmp::eq(&x, &y).map_err(|e| fail(format!("cmp::eq: {e}")))?;
        let expected: Vec<Val> = vx
            .iter()
            .zip(&vy)
            .map(|(a, b)| {
                if *a == Val::Null || *b == Val::Null {
                    Val::Null
                } else {
                    Val::Bool(a == b)
                }
            })
            .collect();
        let out: ArrayRef = Arc::new(out);
        check_rows(&out, &DataType::Boolean, &expected)?;
    }
    Ok(())
}

pub fn bools((a, b): (BooleanArray, BooleanArray)) -> R {
    use arrow_arith::boolean::{and, and_not, not, or};
    check_in(&a)?;
    check_in(&b)?;
    let bit = |x: &BooleanArray, i: usize| x.is_valid(i).then(|| x.value(i));
    let both = |f: fn(bool, bool) -> bool| -> Vec<Val> {
        (0..a.len())
            .map(|i| match (bit(&a, i), bit(&b, i)) {
                (Some(x), Some(y)) => Val::Bool(f(x, y)),
                _ => Val::Null,
            })
            .collect()
    };
    let wrap =
        |r: Result<BooleanArray, ArrowError>, name: &str| -> Result<ArrayRef, TestCaseError> {
            r.map(|x| Arc::new(x) as ArrayRef)
                .map_err(|e| fail(format!("{name}: {e}")))
        };
    check_rows(
        &wrap(and(&a, &b), "and")?,
        &DataType::Boolean,
        &both(|x, y| x & y),
    )?;
    check_rows(
        &wrap(or(&a, &b), "or")?,
        &DataType::Boolean,
        &both(|x, y| x | y),
    )?;
    check_rows(
        &wrap(and_not(&a, &b), "and_not")?,
        &DataType::Boolean,
        &both(|x, y| x & !y),
    )?;
    let want: Vec<Val> = (0..a.len())
        .map(|i| bit(&a, i).map_or(Val::Null, |x| Val::Bool(!x)))
        .collect();
    check_rows(&wrap(not(&a), "not")?, &DataType::Boolean, &want)?;
    // Law: and_not(a, b) == and(a, not(b))
    let lhs = and_not(&a, &b).map_err(|e| fail(format!("and_not: {e}")))?;
    let rhs = not(&b)
        .and_then(|nb| and(&a, &nb))
        .map_err(|e| fail(format!("and(a, not(b)): {e}")))?;
    if vals(&lhs) != vals(&rhs) {
        return Err(fail("and_not(a, b) != and(a, not(b))"));
    }
    Ok(())
}

/// Known: `filter`, `interleave` and the row format rebuild run-end encoded
/// arrays with `RunArray::try_new`, which declares the values field
/// nullable (apache/arrow-rs#11607). Such a
/// failure names one run-end encoded type in both forms, through
/// `Display` in a type comparison or `Debug` in the coalescer's assertion
fn known_ree_values_nullable(r: R) -> R {
    let Err(TestCaseError::Fail(m)) = &r else {
        return r;
    };
    let m = m.to_string();
    // The run ends and values of each `RunEndEncoded(..)` in the message
    let mut forms = vec![];
    for (i, _) in m.match_indices("RunEndEncoded(") {
        let rest = &m[i + "RunEndEncoded(".len()..];
        let mut depth = 1;
        let Some(end) = rest.char_indices().find_map(|(k, c)| {
            depth += (c == '(') as i32 - (c == ')') as i32;
            (depth == 0).then_some(k)
        }) else {
            continue;
        };
        if let Some((ends, values)) = rest[..end].split_once(", ") {
            forms.push((
                ends.to_string(),
                values
                    .strip_prefix("non-null ")
                    .map_or_else(|| (values.to_string(), true), |v| (v.to_string(), false)),
            ));
        }
    }
    let display = forms
        .iter()
        .any(|(e, (v, nullable))| !nullable && forms.contains(&(e.clone(), (v.clone(), true))));
    let debug = m.contains("RunEndEncoded(Field")
        && m.contains("assertion `left == right` failed")
        && m.contains("\"values\"");
    if display || debug {
        return Err(known(
            "known: run-end encoded values field made nullable (apache/arrow-rs#11607)",
        ));
    }
    r
}

/// Known: `take` rejects a null index into a struct or fixed-size list
/// whose run-end encoded or union child cannot hold a null, though the
/// parent's validity can (apache/arrow-rs#11608, the follow-up agreed in
/// apache/arrow-rs#10994). A type that cannot hold a null at its top level is checked
/// before this, so the error here comes from a child. The message alone
/// would also match that error from a top-level array, so the case must
/// have such a parent
fn known_take_null_under_parent(r: R, inputs: &[ArrayRef]) -> R {
    let child_cannot_hold = |f: &FieldRef| {
        matches!(
            f.data_type(),
            DataType::RunEndEncoded(..) | DataType::Union(..)
        ) && !can_hold_null(f.data_type())
    };
    let parent = |d: &DataType| match d {
        DataType::Struct(fs) => fs.iter().any(child_cannot_hold),
        DataType::FixedSizeList(f, _) => child_cannot_hold(f),
        _ => false,
    };
    match r {
        Err(TestCaseError::Fail(m))
            if m.to_string().contains("Cannot take null indices")
                && inputs.iter().any(|a| any_type(a.data_type(), &parent)) =>
        {
            Err(known(
                "known: take rejects a null index a parent could hold (apache/arrow-rs#11608)",
            ))
        }
        r => r,
    }
}

pub(crate) fn with_known<T: Inputs>(property: fn(T) -> R, input: T) -> R {
    let inputs = input.arrays();
    known_take_null_under_parent(
        known_ree_values_nullable(caught(|| property(input))),
        &inputs,
    )
}

// ---------------------------------------------------------------- laws

fn same_rows(name: &str, x: &ArrayRef, y: &ArrayRef) -> R {
    let (vx, vy) = (vals(x.as_ref()), vals(y.as_ref()));
    if vx.len() != vy.len() {
        return Err(fail(format!(
            "{name}: lengths {} and {}",
            vx.len(),
            vy.len()
        )));
    }
    if let Some(i) = (0..vx.len()).find(|&i| vx[i] != vy[i]) {
        return Err(fail(format!("{name}: row {i}: {:?} vs {:?}", vx[i], vy[i])));
    }
    check_rows(x, x.data_type(), &vy)?;
    check_rows(y, y.data_type(), &vx)
}

/// filter(a, p) == take(a, indices of the kept rows)
pub fn law_filter_take((values, predicate): (ArrayRef, BooleanArray)) -> R {
    check_in(values.as_ref())?;
    let dt = values.data_type();
    let kept = Int32Array::from_iter_values(
        (0..predicate.len())
            .filter(|&i| predicate.is_valid(i) && predicate.value(i))
            .map(|i| i as i32),
    );
    let f = arrow_select::filter::filter(values.as_ref(), &predicate).map_err(|e| {
        kernel_err(e, dt, |a| {
            arrow_select::filter::filter(a.as_ref(), &BooleanArray::from(vec![true, false]))
        })
    })?;
    let t = arrow_select::take::take(values.as_ref(), &kept, None).map_err(|e| {
        kernel_err(e, dt, |a| {
            arrow_select::take::take(a.as_ref(), &Int32Array::from(vec![0]), None)
        })
    })?;
    same_rows("filter vs take", &f, &t)
}

/// take(take(a, i), j) == take(a, take(i, j)), feeding one output into the
/// next kernel
pub fn law_take_take((values, i, j): (ArrayRef, Int32Array, Int32Array)) -> R {
    check_in(values.as_ref())?;
    let dt = values.data_type();
    // The law needs both sides defined, and a null index into a type that
    // cannot hold a null is an error (checked in `take`)
    if (i.null_count() > 0 || j.null_count() > 0) && !can_hold_null(dt) {
        return Err(TestCaseError::reject(
            "outside the law: a null index into a type that cannot hold a null",
        ));
    }
    let probe =
        |a: &ArrayRef| arrow_select::take::take(a.as_ref(), &Int32Array::from(vec![0]), None);
    let ti = arrow_select::take::take(values.as_ref(), &i, None)
        .map_err(|e| kernel_err(e, dt, probe))?;
    let lhs =
        arrow_select::take::take(ti.as_ref(), &j, None).map_err(|e| kernel_err(e, dt, probe))?;
    let ij =
        arrow_select::take::take(&i, &j, None).map_err(|e| fail(format!("take(i, j): {e}")))?;
    let ij = ij.as_any().downcast_ref::<Int32Array>().unwrap();
    let rhs = arrow_select::take::take(values.as_ref(), ij, None)
        .map_err(|e| kernel_err(e, dt, probe))?;
    same_rows("take(take(a, i), j) vs take(a, take(i, j))", &lhs, &rhs)
}

/// concat of two adjacent slices == the whole
pub fn law_concat_slices((a, k, _): (ArrayRef, usize, usize)) -> R {
    check_in(a.as_ref())?;
    let dt = a.data_type();
    let (l, r) = (a.slice(0, k), a.slice(k, a.len() - k));
    let out = guarded(dt, || {
        arrow_select::concat::concat(&[l.as_ref(), r.as_ref()])
    })?
    .map_err(|e| {
        kernel_err(e, dt, |x| {
            arrow_select::concat::concat(&[x.as_ref(), x.as_ref()])
        })
    })?;
    same_rows("concat of slices vs whole", &out, &a)
}

/// a.slice(o1, l1).slice(o2, l2) == a.slice(o1 + o2, l2), through arrays
/// and through ArrayData
pub fn law_slice_slice((a, o1, l1, o2, l2): (ArrayRef, usize, usize, usize, usize)) -> R {
    check_in(a.as_ref())?;
    let twice = a.slice(o1, l1).slice(o2, l2);
    let once = a.slice(o1 + o2, l2);
    same_rows("slice of slice vs slice", &twice, &once)?;
    let data = make_array(a.to_data().slice(o1, l1).slice(o2, l2));
    same_rows("ArrayData slice of slice vs slice", &data, &once)
}

/// interleave(arrays, idx) == take(concat(arrays), the same rows by position)
pub fn law_interleave((arrays, indices): Interleave) -> R {
    arrays.iter().try_for_each(|a| check_in(a.as_ref()))?;
    let dt = arrays[0].data_type();
    let refs: Vec<&dyn Array> = arrays.iter().map(|a| a.as_ref()).collect();
    let lhs =
        guarded(dt, || arrow_select::interleave::interleave(&refs, &indices))?.map_err(|e| {
            kernel_err(e, dt, |a| {
                arrow_select::interleave::interleave(&[a.as_ref()], &[(0, 0)])
            })
        })?;
    let all = guarded(dt, || arrow_select::concat::concat(&refs))?.map_err(|e| {
        kernel_err(e, dt, |a| {
            arrow_select::concat::concat(&[a.as_ref(), a.as_ref()])
        })
    })?;
    let starts: Vec<usize> = arrays
        .iter()
        .scan(0, |acc, a| {
            let s = *acc;
            *acc += a.len();
            Some(s)
        })
        .collect();
    let mapped = Int32Array::from_iter_values(indices.iter().map(|&(a, r)| (starts[a] + r) as i32));
    let rhs = arrow_select::take::take(all.as_ref(), &mapped, None).map_err(|e| {
        kernel_err(e, dt, |a| {
            arrow_select::take::take(a.as_ref(), &Int32Array::from(vec![0]), None)
        })
    })?;
    same_rows("interleave vs take of concat", &lhs, &rhs)
}

/// The `ArrayData` APIs on raw data with the offset set on the parent, which
/// typed arrays normalize away: validation, equality with the same rows in
/// another representation, `MutableArrayData::extend`, and rejection of an
/// offset past the end
pub fn data_ops((a, offset, len): (ArrayRef, usize, usize)) -> R {
    use arrow_data::transform::MutableArrayData;
    check_in(a.as_ref())?;
    let data = a.to_data();
    let nulls = data.nulls().map(|b| b.slice(offset, len));
    let Ok(raw) = data
        .clone()
        .into_builder()
        .offset(data.offset() + offset)
        .len(len)
        .nulls(nulls)
        .build()
    else {
        return Err(TestCaseError::reject("type does not take a parent offset"));
    };
    let expected = &vals(a.as_ref())[offset..offset + len];
    raw.validate_full()
        .map_err(|e| fail(format!("raw parent offset data invalid: {e}")))?;
    check(&raw).map_err(|e| fail(format!("raw parent offset data violates the spec: {e}")))?;
    check_rows(&make_array(raw.clone()), a.data_type(), expected)
        .map_err(|e| fail(format!("make_array of raw parent offset data: {e:?}")))?;
    if !physical_nulls_differ(a.data_type()) {
        if raw != a.slice(offset, len).to_data() {
            return Err(fail("raw parent offset data != Array::slice data"));
        }
        if raw != make_array(raw.clone()).to_data() {
            return Err(fail("raw parent offset data != its typed array's data"));
        }
    }
    let extended = catch_unwind(AssertUnwindSafe(|| {
        let mut m = MutableArrayData::new(vec![&raw], false, len);
        m.try_extend(0, 0, len).unwrap();
        m.freeze()
    }))
    .map_err(|p| {
        fail(format!(
            "MutableArrayData::extend panicked on raw parent offset data: {}",
            panic_message(&*p)
        ))
    })?;
    check_rows(&make_array(extended), a.data_type(), expected).map_err(|e| {
        fail(format!(
            "MutableArrayData::extend of raw parent offset data: {e:?}"
        ))
    })?;
    // Negative case: with the offset one row past the end, arrow-rs's
    // validation and the spec checker must agree on whether it is valid
    let full = data.offset() + a.len();
    let builder = data
        .clone()
        .into_builder()
        .offset(full - len + 1)
        .len(len)
        .nulls(None);
    // SAFETY: the data is only validated and inspected, never read as an array
    let past = unsafe { builder.build_unchecked() };
    let ok = |f: &dyn Fn() -> bool| catch_unwind(AssertUnwindSafe(f)).unwrap_or(false);
    let arrow_ok = ok(&|| past.validate_full().is_ok());
    let spec_ok = ok(&|| check(&past).is_ok());
    let spec_err = catch_unwind(AssertUnwindSafe(|| check(&past).err()))
        .ok()
        .flatten();
    if arrow_ok != spec_ok {
        return Err(fail(format!(
            "offset {} length {len} over {full} rows: validate_full ok={arrow_ok}, spec check ok={spec_ok} ({spec_err:?})",
            full - len + 1
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------- comparison and sort

/// The type whose values a comparison sees: a dictionary's or run-end
/// encoded array's value type, otherwise the type itself
fn value_type(dt: &DataType) -> &DataType {
    match dt {
        DataType::Dictionary(_, v) => v,
        DataType::RunEndEncoded(_, v) => v.data_type(),
        t => t,
    }
}

/// `bytes` as a little-endian two's complement integer, compared with
/// another of the same width
fn cmp_signed_le(a: &[u8], b: &[u8]) -> Ordering {
    let (sa, sb) = (a[a.len() - 1] & 0x80 != 0, b[b.len() - 1] & 0x80 != 0);
    if sa != sb {
        return sb.cmp(&sa);
    }
    a.iter().rev().cmp(b.iter().rev())
}

/// The signed fields of an interval's bytes: one i32, two i32, or two i32
/// and an i64, as Schema.fbs lays them out
pub fn interval_fields(b: &[u8]) -> Vec<i64> {
    let i32_at = |k: usize| i32::from_le_bytes(b[k..k + 4].try_into().unwrap()) as i64;
    match b.len() {
        4 => vec![i32_at(0)],
        8 => vec![i32_at(0), i32_at(4)],
        16 => vec![
            i32_at(0),
            i32_at(4),
            i64::from_le_bytes(b[8..16].try_into().unwrap()),
        ],
        n => panic!("no interval of {n} bytes"),
    }
}

/// The order of two non-null values of type `dt`, as arrow-ord documents it:
/// IEEE 754 totalOrder for floats, byte order for strings and binary
pub fn leaf_cmp(dt: &DataType, a: &Val, b: &Val) -> Ordering {
    use DataType::*;
    match (a, b) {
        (Val::Bool(x), Val::Bool(y)) => x.cmp(y),
        (Val::I32(x), Val::I32(y)) => x.cmp(y),
        (Val::I64(x), Val::I64(y)) => x.cmp(y),
        (Val::Str(x), Val::Str(y)) => x.as_bytes().cmp(y.as_bytes()),
        (Val::Bytes(x), Val::Bytes(y)) => x.cmp(y),
        (Val::Raw(x), Val::Raw(y)) => match value_type(dt) {
            Float32 => f32::from_le_bytes(x[..].try_into().unwrap())
                .total_cmp(&f32::from_le_bytes(y[..].try_into().unwrap())),
            Float64 => f64::from_le_bytes(x[..].try_into().unwrap())
                .total_cmp(&f64::from_le_bytes(y[..].try_into().unwrap())),
            UInt8 | UInt16 | UInt32 | UInt64 => x.iter().rev().cmp(y.iter().rev()),
            // Field by field, as the derived `Ord` of arrow-buffer's interval
            // types compares them: (days, milliseconds) and (months, days,
            // nanoseconds), each signed
            Interval(_) => interval_fields(x).cmp(&interval_fields(y)),
            _ => cmp_signed_le(x, y),
        },
        (a, b) => panic!("not comparable: {a:?} {b:?}"),
    }
}

/// Values of two arrays, or of an array and a one-row scalar, compared with
/// every `cmp` kernel; nulls give null except for `distinct` and
/// `not_distinct`, which treat two nulls as equal
pub fn cmp_ops((x, y, scalar): (ArrayRef, ArrayRef, bool)) -> R {
    use Ordering::*;
    use arrow_ord::cmp;
    check_in(x.as_ref())?;
    check_in(y.as_ref())?;
    let dt = x.data_type().clone();
    let (vx, vy) = (vals(x.as_ref()), vals(y.as_ref()));
    let scalar = scalar && !y.is_empty();
    let rhs_val = |i: usize| if scalar { &vy[0] } else { &vy[i] };
    type Op = fn(&dyn Datum, &dyn Datum) -> Result<BooleanArray, ArrowError>;
    // A kernel, and whether it holds for each ordering of two values
    type CmpOp = (&'static str, Op, fn(Ordering) -> bool);
    let ops: [CmpOp; 6] = [
        ("eq", cmp::eq, |o| o == Equal),
        ("neq", cmp::neq, |o| o != Equal),
        ("lt", cmp::lt, |o| o == Less),
        ("lt_eq", cmp::lt_eq, |o| o != Greater),
        ("gt", cmp::gt, |o| o == Greater),
        ("gt_eq", cmp::gt_eq, |o| o != Less),
    ];
    // `Scalar::new` asserts one row, so build it only when used
    let scalar_y = scalar.then(|| Scalar::new(y.slice(0, 1)));
    let run = |op: Op| -> Result<BooleanArray, ArrowError> {
        match &scalar_y {
            Some(sy) => op(&x, sy),
            None => op(&x, &y),
        }
    };
    for (name, op, pred) in ops {
        let out = match run(op) {
            Ok(o) => o,
            Err(e) => {
                let probe = |a: &ArrayRef| cmp::eq(a, a).map(|r| Arc::new(r) as ArrayRef);
                return Err(kernel_err(e, &dt, probe));
            }
        };
        let expected: Vec<Val> = (0..x.len())
            .map(|i| match (&vx[i], rhs_val(i)) {
                (Val::Null, _) | (_, Val::Null) => Val::Null,
                (a, b) => Val::Bool(pred(leaf_cmp(&dt, a, b))),
            })
            .collect();
        let out: ArrayRef = Arc::new(out);
        check_rows(&out, &DataType::Boolean, &expected).map_err(|e| {
            fail(format!(
                "{name}{}: {e:?}",
                if scalar { " with a scalar" } else { "" }
            ))
        })?;
    }
    for (name, op, distinct) in [
        ("distinct", cmp::distinct as Op, true),
        ("not_distinct", cmp::not_distinct as Op, false),
    ] {
        let out = run(op).map_err(|e| fail(format!("{name}: {e}")))?;
        let expected: Vec<Val> = (0..x.len())
            .map(|i| {
                let d = match (&vx[i], rhs_val(i)) {
                    (Val::Null, Val::Null) => false,
                    (Val::Null, _) | (_, Val::Null) => true,
                    (a, b) => leaf_cmp(&dt, a, b) != Equal,
                };
                Val::Bool(d == distinct)
            })
            .collect();
        let out: ArrayRef = Arc::new(out);
        check_rows(&out, &DataType::Boolean, &expected)
            .map_err(|e| fail(format!("{name}: {e:?}")))?;
    }
    Ok(())
}

/// `sort_to_indices` gives a permutation (or, with a limit, a prefix of one)
/// in sorted order, and `sort` gives the same rows; equal values may come in
/// any order, as the sort is documented as unstable
pub fn sort_prop((a, descending, nulls_first, limit): (ArrayRef, bool, bool, Option<usize>)) -> R {
    use Ordering::*;
    use arrow_ord::sort::{SortOptions, sort, sort_to_indices};
    check_in(a.as_ref())?;
    let dt = a.data_type().clone();
    let opts = SortOptions {
        descending,
        nulls_first,
    };
    let v = vals(a.as_ref());
    let key = |x: &Val, y: &Val| match (x, y) {
        (Val::Null, Val::Null) => Equal,
        (Val::Null, _) => {
            if nulls_first {
                Less
            } else {
                Greater
            }
        }
        (_, Val::Null) => {
            if nulls_first {
                Greater
            } else {
                Less
            }
        }
        (x, y) => {
            let o = leaf_cmp(&dt, x, y);
            if descending { o.reverse() } else { o }
        }
    };
    let probe =
        |x: &ArrayRef| sort_to_indices(x.as_ref(), None, None).map(|r| Arc::new(r) as ArrayRef);
    let idx =
        sort_to_indices(a.as_ref(), Some(opts), limit).map_err(|e| kernel_err(e, &dt, probe))?;
    let want_len = limit.map_or_else(|| a.len(), |k| k.min(a.len()));
    if idx.len() != want_len || idx.null_count() > 0 {
        return Err(fail(format!(
            "{} indices with {} nulls, expected {want_len}",
            idx.len(),
            idx.null_count()
        )));
    }
    let picked: Vec<usize> = idx.values().iter().map(|&i| i as usize).collect();
    let mut seen = vec![false; a.len()];
    for &i in &picked {
        if i >= a.len() || replace(&mut seen[i], true) {
            return Err(fail(format!("index {i} out of range or repeated")));
        }
    }
    if let Some(w) = (1..picked.len()).find(|&w| key(&v[picked[w - 1]], &v[picked[w]]) == Greater) {
        return Err(fail(format!(
            "out of order at {w}: {:?} then {:?}",
            v[picked[w - 1]],
            v[picked[w]]
        )));
    }
    // With a limit, the rows are the first ones of a full sort
    let mut all: Vec<&Val> = v.iter().collect();
    all.sort_by(|x, y| key(x, y));
    if let Some(k) = (0..picked.len()).find(|&k| key(&v[picked[k]], all[k]) != Equal) {
        return Err(fail(format!(
            "row {k} of the result is {:?}, a full sort has {:?}",
            v[picked[k]], all[k]
        )));
    }
    if limit.is_none() {
        let sorted = sort(&a, Some(opts)).map_err(|e| fail(format!("sort: {e}")))?;
        let expected: Vec<Val> = picked.iter().map(|&i| v[i].clone()).collect();
        // Equal values may be distinct representations, such as NaN payloads
        // compared equal only when bits match, so compare by the sort key
        let got = vals(sorted.as_ref());
        if got.len() != expected.len()
            || (0..got.len()).any(|i| key(&got[i], &expected[i]) != Equal)
        {
            return Err(fail("sort(a) differs from take(a, sort_to_indices(a))"));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- arithmetic

use crate::inventory::Inputs;
use crate::known::known;
use crate::spec::check;
use crate::types::{any_type, is_view_dict};
use arrow_array::Datum;
use arrow_array::Scalar;
use arrow_array::ffi::from_ffi;
use arrow_array::ffi::to_ffi;
use num_bigint::{BigInt, Sign};
use num_traits::pow;
use num_traits::{Signed, ToPrimitive, Zero};
use std::cmp::Ordering;
use std::io::Cursor;
use std::mem::replace;
use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;
use std::panic::resume_unwind;
use std::sync::atomic::AtomicUsize;

fn is_unsigned(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64
    )
}

pub(crate) fn decimal_scale(dt: &DataType) -> Option<i32> {
    match dt {
        DataType::Decimal32(_, s)
        | DataType::Decimal64(_, s)
        | DataType::Decimal128(_, s)
        | DataType::Decimal256(_, s) => Some(*s as i32),
        _ => None,
    }
}

/// A non-null integer or decimal value as an exact integer
pub(crate) fn big(dt: &DataType, v: &Val) -> BigInt {
    match v {
        Val::I32(x) => BigInt::from(*x),
        Val::I64(x) => BigInt::from(*x),
        Val::Raw(b) if is_unsigned(dt) => BigInt::from_bytes_le(Sign::Plus, b),
        Val::Raw(b) => BigInt::from_signed_bytes_le(b),
        v => panic!("not an integer: {v:?}"),
    }
}

/// Whether `v` fits the storage of `dt`
pub(crate) fn fits(dt: &DataType, v: &BigInt) -> bool {
    let bits = dt.primitive_width().unwrap() as u32 * 8;
    if is_unsigned(dt) {
        !v.is_negative() && v.bits() <= bits as u64
    } else {
        let lim = BigInt::from(1) << (bits - 1);
        -&lim <= *v && *v < lim
    }
}

/// `v` reduced to the width of `dt`, as wrapping arithmetic leaves it
pub(crate) fn wrap(dt: &DataType, v: &BigInt) -> BigInt {
    let bits = dt.primitive_width().unwrap() as u32 * 8;
    let m = BigInt::from(1) << bits;
    let mut r = ((v % &m) + &m) % &m;
    if !is_unsigned(dt) && r >= (BigInt::from(1) << (bits - 1)) {
        r -= &m;
    }
    r
}

/// `v` as the oracle reads a value of type `dt`
pub(crate) fn as_val(dt: &DataType, v: &BigInt) -> Val {
    match dt {
        DataType::Int32 => Val::I32(v.to_i32().unwrap()),
        DataType::Int64 => Val::I64(v.to_i64().unwrap()),
        _ => {
            let w = dt.primitive_width().unwrap();
            let mut b = if is_unsigned(dt) {
                v.to_bytes_le().1
            } else {
                v.to_signed_bytes_le()
            };
            b.resize(w, if v.is_negative() { 0xFF } else { 0 });
            Val::Raw(b)
        }
    }
}

pub(crate) fn pow10(k: i32) -> BigInt {
    assert!(k >= 0, "negative exponent {k}");
    pow(BigInt::from(10), k as usize)
}

/// Integer division truncated toward zero, as Rust and arrow divide
fn div_trunc(a: &BigInt, b: &BigInt) -> BigInt {
    a / b
}

#[derive(Debug, PartialEq)]
enum Expect {
    Value(BigInt),
    /// Required to fail: division by zero, or a result that does not fit
    Error,
    /// The result, or an error from an intermediate scaled operand that
    /// does not fit even though the result does (documented for decimals)
    ValueOrError(BigInt),
}

/// The exact result of `op` on one row of integers or decimals
fn reference(
    op: &str,
    lt: &DataType,
    rt: &DataType,
    out: &DataType,
    l: &BigInt,
    r: &BigInt,
) -> Expect {
    let wrapping = op.ends_with("_wrapping");
    let base = op.trim_end_matches("_wrapping");
    let done = |v: BigInt, intermediate_ok: bool| -> Expect {
        if wrapping && decimal_scale(out).is_none() {
            return Expect::Value(wrap(out, &v));
        }
        match (fits(out, &v), intermediate_ok) {
            (true, true) => Expect::Value(v),
            (true, false) => Expect::ValueOrError(v),
            (false, _) => Expect::Error,
        }
    };
    match (decimal_scale(lt), decimal_scale(rt), decimal_scale(out)) {
        (Some(s1), Some(s2), Some(so)) => match base {
            "add" | "sub" => {
                if so < s1 || so < s2 {
                    return Expect::Error;
                }
                let (ls, rs) = (l * pow10(so - s1), r * pow10(so - s2));
                let v = if base == "add" { &ls + &rs } else { &ls - &rs };
                done(v, fits(out, &ls) && fits(out, &rs))
            }
            "mul" => {
                let e = so - s1 - s2;
                let v = if e >= 0 {
                    l * r * pow10(e)
                } else {
                    div_trunc(&(l * r), &pow10(-e))
                };
                done(v, true)
            }
            "div" => {
                if r.is_zero() {
                    return Expect::Error;
                }
                let e = so - s1 + s2;
                let (num, den) = if e >= 0 {
                    (l * pow10(e), r.clone())
                } else {
                    (l.clone(), r * pow10(-e))
                };
                done(div_trunc(&num, &den), fits(out, &den))
            }
            "rem" => {
                if r.is_zero() {
                    return Expect::Error;
                }
                if so < s1 || so < s2 {
                    return Expect::Error;
                }
                let (a, b) = (l * pow10(so - s1), r * pow10(so - s2));
                done(&a % &b, fits(out, &a) && fits(out, &b))
            }
            _ => unreachable!(),
        },
        _ => match base {
            "add" => done(l + r, true),
            "sub" => done(l - r, true),
            "mul" => done(l * r, true),
            "div" if r.is_zero() => Expect::Error,
            "div" => done(div_trunc(l, r), true),
            "rem" if r.is_zero() => Expect::Error,
            // Documented: signed MIN % -1 is 0, not an error
            "rem" => Expect::Value(l % r),
            _ => unreachable!(),
        },
    }
}

/// Whether `mul` of these decimal types has a result scale outside the
/// type's range, which `decimal_op` documents as an error
fn mul_scale_out_of_range(op: &str, lt: &DataType, rt: &DataType) -> bool {
    let max = match lt {
        DataType::Decimal32(..) => 9,
        DataType::Decimal64(..) => 18,
        DataType::Decimal128(..) => 38,
        DataType::Decimal256(..) => 76,
        _ => return false,
    };
    match (op.starts_with("mul"), decimal_scale(lt), decimal_scale(rt)) {
        (true, Some(a), Some(b)) => a + b > max || a + b < i8::MIN as i32,
        _ => false,
    }
}

/// Integer and decimal arithmetic against exact integers: values, errors
/// on overflow and division by zero, wrapping, and nulls, which never cause
/// an error. Floats are checked against Rust's IEEE 754 operations
/// Non-null `arith` rows seen, and rows whose value was compared, for
/// measuring how much of the input the property checks
pub static ARITH_ROWS: [AtomicUsize; 2] = [AtomicUsize::new(0), AtomicUsize::new(0)];

/// Known: decimal `div` falls back to `scaled_div` when scaling the dividend
/// overflows, and that divides magnitudes, so it fails when the dividend or
/// the quotient is the native minimum. apache/arrow-rs#10911 states the
/// dividend case as unsupported. The quotient case is asked in
/// apache/arrow-rs#11604
fn scaled_div_min(
    name: &str,
    lt: &DataType,
    out: &DataType,
    l: &Val,
    q: &Val,
    e: &ArrowError,
) -> bool {
    let min = |dt: &DataType, v: &Val| {
        *v != Val::Null
            && big(dt, v) == -(BigInt::from(1) << (dt.primitive_width().unwrap() * 8 - 1))
    };
    name == "div"
        && decimal_scale(lt).is_some()
        && matches!(e, ArrowError::ArithmeticOverflow(_))
        && (min(lt, l) || min(out, q))
}

/// Which operand of a binary kernel is a one-row scalar
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Operands {
    Arrays,
    ScalarLeft,
    ScalarRight,
}

pub fn arith((x, y, operands): (ArrayRef, ArrayRef, Operands)) -> R {
    use arrow_arith::numeric;
    check_in(x.as_ref())?;
    check_in(y.as_ref())?;
    let (lt, rt) = (x.data_type().clone(), y.data_type().clone());
    let (vx, vy) = (vals(x.as_ref()), vals(y.as_ref()));
    // `Scalar::new` asserts one row, so a scalar needs a nonempty array
    let left = operands == Operands::ScalarLeft && !x.is_empty();
    let right = operands == Operands::ScalarRight && !y.is_empty();
    let lhs = |i: usize| if left { &vx[0] } else { &vx[i] };
    let rhs = |i: usize| if right { &vy[0] } else { &vy[i] };
    type Op = fn(&dyn Datum, &dyn Datum) -> Result<ArrayRef, ArrowError>;
    let ops: [(&str, Op); 8] = [
        ("add", numeric::add),
        ("add_wrapping", numeric::add_wrapping),
        ("sub", numeric::sub),
        ("sub_wrapping", numeric::sub_wrapping),
        ("mul", numeric::mul),
        ("mul_wrapping", numeric::mul_wrapping),
        ("div", numeric::div),
        ("rem", numeric::rem),
    ];
    let float = matches!(lt, DataType::Float32 | DataType::Float64);
    // `op` on rows `i..i + len`, or on the scalar's row
    let call = |op: Op, i: usize, len: usize| {
        let (xs, ys) = (x.slice(i, len), y.slice(i, len));
        match (left, right) {
            (true, _) => op(&Scalar::new(x.slice(0, 1)), &ys),
            (_, true) => op(&xs, &Scalar::new(y.slice(0, 1))),
            _ => op(&xs, &ys),
        }
    };
    for (name, op) in ops {
        let got = call(op, 0, x.len());
        let ctx = |m: String| {
            let side = match (left, right) {
                (true, _) => " of a scalar",
                (_, true) => " by a scalar",
                _ => "",
            };
            fail(format!("{name}{side}: {m}"))
        };
        if float {
            let out = got.map_err(|e| ctx(format!("error {e}")))?;
            let got_v = vals(out.as_ref());
            for (i, g) in got_v.iter().enumerate() {
                let (a, b) = (lhs(i), rhs(i));
                let want = match (a, b) {
                    (Val::Null, _) | (_, Val::Null) => None,
                    (a, b) => {
                        let (a, b) = (a.as_f64(), b.as_f64());
                        let v = match name.trim_end_matches("_wrapping") {
                            "add" => a + b,
                            "sub" => a - b,
                            "mul" => a * b,
                            "div" => a / b,
                            _ => a % b,
                        };
                        Some(v)
                    }
                };
                let ok = match (&want, g) {
                    (None, Val::Null) => true,
                    // Float32 is computed in f32, so recompute there
                    (Some(_), g) if lt == DataType::Float32 => {
                        let (a, b) = (lhs(i).as_f64() as f32, rhs(i).as_f64() as f32);
                        let v = match name.trim_end_matches("_wrapping") {
                            "add" => a + b,
                            "sub" => a - b,
                            "mul" => a * b,
                            "div" => a / b,
                            _ => a % b,
                        };
                        same_float(g.as_f64(), f64::from(v))
                    }
                    (Some(w), g) => same_float(g.as_f64(), *w),
                    _ => false,
                };
                if !ok {
                    return Err(ctx(format!("row {i}: got {g:?}, expected {want:?}")));
                }
            }
            continue;
        }
        // The kernel decides the output type from the input types, so ask it
        // with empty inputs rather than restating its decimal rules
        let out_type = match &got {
            Ok(a) => a.data_type().clone(),
            Err(_) => match op(&x.slice(0, 0), &y.slice(0, 0)) {
                Ok(a) => a.data_type().clone(),
                Err(_) => continue,
            },
        };
        let mut expected = vec![];
        // Per row: whether it must fail, and whether it may fail
        let mut row_fail: Vec<(bool, bool)> = vec![];
        let mut must_fail = false;
        let mut may_fail = false;
        ARITH_ROWS[0].fetch_add(
            (0..x.len())
                .filter(|&i| *lhs(i) != Val::Null && *rhs(i) != Val::Null)
                .count(),
            std::sync::atomic::Ordering::Relaxed,
        );
        for i in 0..x.len() {
            match (lhs(i), rhs(i)) {
                (Val::Null, _) | (_, Val::Null) => {
                    expected.push(Val::Null);
                    row_fail.push((false, false));
                }
                (a, b) => match reference(name, &lt, &rt, &out_type, &big(&lt, a), &big(&rt, b)) {
                    Expect::Value(v) => {
                        expected.push(as_val(&out_type, &v));
                        row_fail.push((false, false));
                    }
                    Expect::ValueOrError(v) => {
                        may_fail = true;
                        expected.push(as_val(&out_type, &v));
                        row_fail.push((false, true));
                    }
                    Expect::Error => {
                        must_fail = true;
                        expected.push(Val::Null);
                        row_fail.push((true, true));
                    }
                },
            }
        }
        let whole_failed = got.is_err();
        match got {
            Err(e) if must_fail || may_fail => {
                let _ = e;
            }
            // Documented: decimal results whose scale would leave the type's
            // range are rejected from the types alone, before any values
            // Documented: a product scale outside the type's range is an
            // error decided from the types alone; checked below
            Err(e)
                if e.to_string().contains("would exceed max scale")
                    || e.to_string().contains("would be less than min scale") =>
            {
                if !mul_scale_out_of_range(name, &lt, &rt) {
                    return Err(ctx(format!("unexpected scale error {e}")));
                }
                continue;
            }
            Ok(_) if mul_scale_out_of_range(name, &lt, &rt) => {
                return Err(ctx(
                    "expected the documented error for a product scale outside the type's range"
                        .into(),
                ));
            }
            Err(e)
                if (0..x.len())
                    .any(|i| scaled_div_min(name, &lt, &out_type, lhs(i), &expected[i], &e)) =>
            {
                return Err(known(
                    "known: decimal div at the native minimum (apache/arrow-rs#11604)",
                ));
            }
            Err(e) => {
                let probe = |a: &ArrayRef| numeric::add(a, a);
                return Err(match kernel_err(e, &lt, probe) {
                    TestCaseError::Reject(r) => TestCaseError::Reject(r),
                    TestCaseError::Fail(m) => ctx(format!("unexpected {m}")),
                });
            }
            Ok(_) if must_fail => {
                return Err(ctx(
                    "expected an error (overflow or division by zero), got a result".into(),
                ));
            }
            Ok(out) => {
                ARITH_ROWS[1].fetch_add(
                    expected.iter().filter(|v| **v != Val::Null).count(),
                    std::sync::atomic::Ordering::Relaxed,
                );
                if decimal_scale(&lt).is_some() != decimal_scale(out.data_type()).is_some()
                    || out.data_type().primitive_width() != lt.primitive_width()
                {
                    return Err(ctx(format!(
                        "output type {} for {lt} and {rt}",
                        out.data_type()
                    )));
                }
                check_rows(&out, out.data_type(), &expected).map_err(|e| ctx(format!("{e:?}")))?;
            }
        }
        // One failing row makes the whole call an error, so when it failed
        // every row is also checked on its own, through one-row slices
        if whole_failed {
            for i in (0..x.len()).filter(|&i| expected[i] != Val::Null || row_fail[i].0) {
                let r = call(op, i, 1);
                let at = |m: String| ctx(format!("row {i} alone: {m}"));
                match (r, row_fail[i]) {
                    (Err(_), (_, true)) => {}
                    (Err(e), _)
                        if scaled_div_min(name, &lt, &out_type, lhs(i), &expected[i], &e) =>
                    {
                        return Err(known(
                            "known: decimal div at the native minimum (apache/arrow-rs#11604)",
                        ));
                    }
                    (Err(e), _) => return Err(at(format!("unexpected error {e}"))),
                    (Ok(_), (true, _)) => {
                        return Err(at(
                            "expected an error (overflow or division by zero), got a result".into(),
                        ));
                    }
                    (Ok(out), _) => {
                        ARITH_ROWS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        check_rows(&out, out.data_type(), &expected[i..i + 1])
                            .map_err(|e| at(format!("{e:?}")))?;
                    }
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- casts

/// A non-null numeric value as an exact rational `n / 10^scale`, or a float
enum Num {
    Exact(BigInt, i32),
    Float(f64),
}

fn num(dt: &DataType, v: &Val) -> Num {
    match (dt, v) {
        (DataType::Float32 | DataType::Float64, v) => Num::Float(v.as_f64()),
        (dt, v) => Num::Exact(big(dt, v), decimal_scale(dt).unwrap_or(0)),
    }
}

/// `n / 10^k` rounded half away from zero, as `make_downscaler` rounds
pub(crate) fn round_half_away(n: &BigInt, k: i32) -> BigInt {
    let d = pow10(k);
    let (q, r) = (n / &d, n % &d);
    if BigInt::from(2) * r.abs() >= d {
        q + n.signum()
    } else {
        q
    }
}

/// The value a cast to `to` should give, `None` when it is not
/// representable there, or `Err` when this pair is outside the reference
fn cast_reference(to: &DataType, x: Num) -> Result<Option<Val>, ()> {
    use DataType::*;
    let float_val = |f: f64| -> Val {
        match to {
            Float32 => Val::Raw((f as f32).to_le_bytes().to_vec()),
            _ => Val::Raw(f.to_le_bytes().to_vec()),
        }
    };
    match (x, to) {
        (Num::Float(f), Float32 | Float64) => Ok(Some(float_val(f))),
        (Num::Float(_), Decimal32(..) | Decimal64(..) | Decimal128(..) | Decimal256(..)) => Err(()),
        (Num::Float(f), t) => {
            if f.is_nan() || f.is_infinite() {
                return Ok(None);
            }
            // NumCast truncates toward zero and rejects values out of range
            let tr = f.trunc();
            let v = BigInt::from(tr as i128);
            if tr.abs() >= 1.7e38 || !fits(t, &v) {
                return Ok(None);
            }
            Ok(Some(as_val(t, &v)))
        }
        (Num::Exact(n, s), Float32 | Float64) => {
            // Correctly rounded: parse the decimal's own text
            let text = format!("{n}e{}", -s);
            Ok(Some(match to {
                Float32 => Val::Raw(text.parse::<f32>().unwrap().to_le_bytes().to_vec()),
                _ => Val::Raw(text.parse::<f64>().unwrap().to_le_bytes().to_vec()),
            }))
        }
        (Num::Exact(n, s), t) => match decimal_scale(t) {
            None => {
                // To an integer: truncate toward zero; a negative scale multiplies
                let v = if s >= 0 {
                    &n / pow10(s)
                } else {
                    &n * pow10(-s)
                };
                Ok(fits(t, &v).then(|| as_val(t, &v)))
            }
            Some(s2) => {
                let from_int = s == 0 && s2 < 0;
                if from_int {
                    return Err(());
                }
                let v = if s2 >= s {
                    &n * pow10(s2 - s)
                } else {
                    round_half_away(&n, s - s2)
                };
                let p2 = match t {
                    Decimal32(p, _) | Decimal64(p, _) | Decimal128(p, _) | Decimal256(p, _) => {
                        *p as i32
                    }
                    _ => unreachable!(),
                };
                Ok((v.abs() < pow10(p2) && fits(t, &v)).then(|| as_val(t, &v)))
            }
        },
    }
}

/// `cast_with_options` between numeric types: exact values where the target
/// can represent them; otherwise null with `safe`, or an error without
pub fn cast_prop((x, to, safe): (ArrayRef, DataType, bool)) -> R {
    use arrow_cast::cast::{CastOptions, can_cast_types, cast_with_options};
    check_in(x.as_ref())?;
    let from = x.data_type().clone();
    if !can_cast_types(&from, &to) {
        return Err(TestCaseError::reject("cast not supported"));
    }
    let v = vals(x.as_ref());
    let from_precision = match from {
        DataType::Decimal32(p, _)
        | DataType::Decimal64(p, _)
        | DataType::Decimal128(p, _)
        | DataType::Decimal256(p, _) => Some(p as i32),
        _ => None,
    };
    let mut expected = Vec::with_capacity(v.len());
    // Rows whose result is open: a valid value past its own declared
    // precision, which apache/arrow-rs#10947 leaves undecided
    let mut open = vec![false; v.len()];
    let mut unrepresentable = false;
    let mut may_fail = false;
    for (i, r) in v.iter().enumerate() {
        if let (Some(p), r) = (from_precision, r)
            && *r != Val::Null
            && big(&from, r).abs() >= pow10(p)
        {
            open[i] = true;
            may_fail = true;
            expected.push(Val::Null);
            continue;
        }
        match r {
            Val::Null => expected.push(Val::Null),
            r => match cast_reference(&to, num(&from, r)) {
                Ok(Some(e)) => expected.push(e),
                Ok(None) => {
                    unrepresentable = true;
                    expected.push(Val::Null)
                }
                Err(()) => return Err(TestCaseError::reject("pair outside the reference")),
            },
        }
    }
    let opts = CastOptions {
        safe,
        ..Default::default()
    };
    let ctx = |m: String| fail(format!("{from} to {to} (safe: {safe}): {m}"));
    let is_dec = |d: &DataType| decimal_scale(d).is_some();
    let result = match catch_unwind(AssertUnwindSafe(|| cast_with_options(&x, &to, &opts))) {
        Ok(r) => r,
        // Known: the infallible upscale path unwraps every slot, null or past
        // its precision (apache/arrow-rs#10947)
        Err(p)
            if is_dec(&from)
                && is_dec(&to)
                && p.downcast_ref::<&str>()
                    .is_some_and(|m| m.contains("unwrap()"))
                || p.downcast_ref::<String>()
                    .is_some_and(|m| m.contains("unwrap()"))
                    && is_dec(&from)
                    && is_dec(&to) =>
        {
            return Err(known(
                "known: decimal upscale unwraps slots that do not fit (apache/arrow-rs#10947)",
            ));
        }
        Err(p) => resume_unwind(p),
    };
    // Known: a scale increase beyond the target width's maximum precision
    // errors for the whole cast, even with `safe` and even for zeros
    // (apache/arrow-rs#11589, related to apache/arrow-rs#10946)
    if let (Err(e), Some(s1), Some(s2)) = (&result, decimal_scale(&from), decimal_scale(&to)) {
        let max_p = match to {
            DataType::Decimal32(..) => 9,
            DataType::Decimal64(..) => 18,
            DataType::Decimal128(..) => 38,
            _ => 76,
        };
        if s2 - s1 > max_p && e.to_string().contains("Value overflows for output scale") {
            return Err(known(
                "known: scale increase beyond the table (apache/arrow-rs#11589)",
            ));
        }
    }
    match result {
        Err(_) if !safe && (unrepresentable || may_fail) => Ok(()),
        Err(e) => Err(ctx(format!("unexpected error {e}"))),
        Ok(_) if !safe && unrepresentable => Err(ctx(
            "expected an error for a value the target cannot represent".into(),
        )),
        Ok(out) => {
            if out.data_type() != &to {
                return Err(ctx(format!("output type {}", out.data_type())));
            }
            let got = vals(out.as_ref());
            // Floats: equal bits, or both NaN
            let same = |g: &Val, e: &Val| match (g, e, &to) {
                (Val::Raw(_), Val::Raw(_), DataType::Float32 | DataType::Float64) => {
                    same_float(g.as_f64(), e.as_f64())
                }
                (g, e, _) => g == e,
            };
            if got.len() != expected.len() {
                return Err(ctx(format!("len {} != {}", got.len(), expected.len())));
            }
            if let Some(i) = (0..got.len()).find(|&i| !open[i] && !same(&got[i], &expected[i])) {
                return Err(ctx(format!(
                    "row {i}: {:?} gave {:?}, expected {:?}",
                    v[i], got[i], expected[i]
                )));
            }
            out.to_data()
                .validate_full()
                .map_err(|e| ctx(format!("invalid output: {e}")))?;
            Ok(())
        }
    }
}

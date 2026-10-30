//! Validity per the Arrow Columnar spec (apache/arrow
//! docs/source/format/Columnar.rst, blob f95d7e8), written from the spec text
//! and independent of arrow-rs's `ArrayData::validate_full`. Only slots the
//! spec defines are checked: content under nulls is undefined (lines 442-445).
//! Line numbers without a file name refer to Columnar.rst. Schema.fbs line
//! numbers here and in `gen.rs` refer to apache/arrow format/Schema.fbs, blob
//! 67b7719.

use arrow_buffer::{ArrowNativeType, Buffer};
use arrow_data::ArrayData;
use arrow_schema::TimeUnit;
use arrow_schema::{DataType, UnionMode};
use std::str::from_utf8;

type R = Result<(), String>;

/// Slot `i` of an integer buffer of type `t`, widened to i64. A `u64` above
/// `i64::MAX` comes back negative, which every caller treats as invalid
fn int_at(b: &Buffer, t: &DataType, i: usize) -> i64 {
    match t {
        DataType::Int8 => ints::<i8>(b)[i] as i64,
        DataType::Int16 => ints::<i16>(b)[i] as i64,
        DataType::Int32 => ints::<i32>(b)[i] as i64,
        DataType::Int64 => ints::<i64>(b)[i],
        DataType::UInt8 => ints::<u8>(b)[i] as i64,
        DataType::UInt16 => ints::<u16>(b)[i] as i64,
        DataType::UInt32 => ints::<u32>(b)[i] as i64,
        DataType::UInt64 => ints::<u64>(b)[i] as i64,
        t => panic!("not an integer type: {t}"),
    }
}

fn err(d: &ArrayData, msg: String) -> R {
    Err(format!("{}: {msg}", d.data_type()))
}

fn valid(d: &ArrayData, i: usize) -> bool {
    d.nulls().is_none_or(|n| n.is_valid(i))
}

fn ints<T: ArrowNativeType>(b: &Buffer) -> &[T] {
    b.typed_data::<T>()
}

/// Lines 430-432 and 447-449: `length + 1` offsets give each slot's position
/// in the values, and they increase monotonically, even for null slots
fn offsets<T: ArrowNativeType + Ord>(d: &ArrayData, values_len: usize) -> R {
    let o = ints::<T>(&d.buffers()[0]);
    let (start, len) = (d.offset(), d.len());
    if o.len() < start + len + 1 {
        return err(
            d,
            format!(
                "offsets buffer has {} entries, needs {}",
                o.len(),
                start + len + 1
            ),
        );
    }
    let o = &o[start..start + len + 1];
    if o.windows(2).any(|w| w[1] < w[0]) {
        return err(d, "offsets decrease".into());
    }
    if o[0].as_usize() > values_len || o[len].as_usize() > values_len {
        return err(
            d,
            format!(
                "offsets {}..{} exceed values length {values_len}",
                o[0].as_usize(),
                o[len].as_usize()
            ),
        );
    }
    Ok(())
}

/// Lines 505-524: a short view is inline and zero padded; a long view's
/// bytes lie within its buffer and its prefix copies their first four bytes
fn views(d: &ArrayData, utf8: bool) -> R {
    let v = ints::<u128>(&d.buffers()[0]);
    // The views buffer covers every slot up to `offset + len`, even when the
    // array is empty
    if v.len() < d.offset() + d.len() {
        return err(
            d,
            format!(
                "views buffer has {} entries, needs {}",
                v.len(),
                d.offset() + d.len()
            ),
        );
    }
    let data = &d.buffers()[1..];
    for i in 0..d.len() {
        if !valid(d, i) {
            continue;
        }
        let view = v[d.offset() + i];
        let raw = view.to_le_bytes();
        let len = i32::from_le_bytes(raw[0..4].try_into().unwrap());
        if len < 0 {
            return err(d, format!("slot {i}: negative length {len}"));
        }
        let len = len as usize;
        let bytes: &[u8] = if len <= 12 {
            if raw[4 + len..].iter().any(|&b| b != 0) {
                return err(
                    d,
                    format!("slot {i}: inline string of {len} bytes not zero padded"),
                );
            }
            &raw[4..4 + len]
        } else {
            let idx = i32::from_le_bytes(raw[8..12].try_into().unwrap());
            let off = i32::from_le_bytes(raw[12..16].try_into().unwrap());
            if idx < 0 || idx as usize >= data.len() || off < 0 {
                return err(
                    d,
                    format!(
                        "slot {i}: buffer index {idx} offset {off} with {} buffers",
                        data.len()
                    ),
                );
            }
            let buf = &data[idx as usize];
            let (off, end) = (off as usize, off as usize + len);
            if end > buf.len() {
                return err(
                    d,
                    format!(
                        "slot {i}: bytes {off}..{end} outside buffer of {}",
                        buf.len()
                    ),
                );
            }
            if raw[4..8] != buf[off..off + 4] {
                return err(d, format!("slot {i}: prefix does not match the data"));
            }
            &buf[off..end]
        };
        if utf8 && from_utf8(bytes).is_err() {
            return err(d, format!("slot {i}: invalid UTF-8"));
        }
    }
    Ok(())
}

fn var_binary<T: ArrowNativeType + Ord>(d: &ArrayData, utf8: bool) -> R {
    let values = &d.buffers()[1];
    offsets::<T>(d, values.len())?;
    if utf8 {
        let o = &ints::<T>(&d.buffers()[0])[d.offset()..];
        for i in (0..d.len()).filter(|&i| valid(d, i)) {
            if from_utf8(&values[o[i].as_usize()..o[i + 1].as_usize()]).is_err() {
                return err(d, format!("slot {i}: invalid UTF-8"));
            }
        }
    }
    Ok(())
}

/// Lines 643-647: every list-view slot, null or not, stays within the child
fn list_view<T: ArrowNativeType>(d: &ArrayData) -> R {
    let end = d.offset() + d.len();
    let (no, ns) = (
        ints::<T>(&d.buffers()[0]).len(),
        ints::<T>(&d.buffers()[1]).len(),
    );
    if no < end || ns < end {
        return err(
            d,
            format!("offsets ({no}) or sizes ({ns}) shorter than {end}"),
        );
    }
    let child_len = d.child_data()[0].len() as i64;
    let o = &ints::<T>(&d.buffers()[0])[d.offset()..];
    let s = &ints::<T>(&d.buffers()[1])[d.offset()..];
    for i in 0..d.len() {
        let (off, size) = (o[i].to_i64().unwrap(), s[i].to_i64().unwrap());
        if off < 0 || off > child_len || size < 0 || off + size > child_len {
            return err(
                d,
                format!("slot {i}: offset {off} size {size} with child length {child_len}"),
            );
        }
    }
    Ok(())
}

fn union(d: &ArrayData, ids: &[i8], mode: UnionMode) -> R {
    let end = d.offset() + d.len();
    let n_types = ints::<i8>(&d.buffers()[0]).len();
    let n_offsets = (mode == UnionMode::Dense).then(|| ints::<i32>(&d.buffers()[1]).len());
    if n_types < end || n_offsets.is_some_and(|n| n < end) {
        return err(
            d,
            format!("type ids ({n_types}) or offsets ({n_offsets:?}) shorter than {end}"),
        );
    }
    let types = &ints::<i8>(&d.buffers()[0])[d.offset()..];
    if d.nulls().is_some() {
        return err(d, "a union has no validity bitmap (lines 873-875)".into());
    }
    for (i, t) in types.iter().take(d.len()).enumerate() {
        if !ids.contains(t) {
            return err(d, format!("slot {i}: type id {t} not declared"));
        }
    }
    match mode {
        // Lines 942-944: each child as long as the union
        UnionMode::Sparse => {
            for c in d.child_data() {
                if c.len() < d.offset() + d.len() {
                    return err(
                        d,
                        format!("sparse child length {} < {}", c.len(), d.offset() + d.len()),
                    );
                }
            }
        }
        // Lines 891-894: offsets within each child, in increasing order. Equal
        // offsets share a slot, which Arrow C++'s ValidateFull also accepts
        UnionMode::Dense => {
            let offs = &ints::<i32>(&d.buffers()[1])[d.offset()..];
            let mut last = vec![-1i64; ids.len()];
            for i in 0..d.len() {
                let c = ids.iter().position(|&x| x == types[i]).unwrap();
                let off = offs[i] as i64;
                if off < 0 || off >= d.child_data()[c].len() as i64 {
                    return err(d, format!("slot {i}: offset {off} outside child {c}"));
                }
                if off < last[c] {
                    return err(
                        d,
                        format!(
                            "slot {i}: offsets into child {c} decrease ({} then {off})",
                            last[c]
                        ),
                    );
                }
                last[c] = off;
            }
        }
    }
    Ok(())
}

/// Lines 1129-1133: run ends positive, strictly increasing, never null; the
/// parent has no nulls. That the run ends cover `offset + len` is implied by
/// lines 1108-1110 (a run end is the logical index where the run ends), not
/// stated. An empty array has no logical index to cover, as arrow-rs's
/// validation agrees (apache/arrow-rs#11668)
fn run_end(d: &ArrayData) -> R {
    if d.nulls().is_some() {
        return err(d, "run-end encoded parent has a validity bitmap".into());
    }
    let (ends, values) = (&d.child_data()[0], &d.child_data()[1]);
    if ends.nulls().is_some_and(|n| n.null_count() > 0) {
        return err(d, "null run end".into());
    }
    let e: Vec<i64> = (0..ends.len())
        .map(|i| int_at(&ends.buffers()[0], ends.data_type(), ends.offset() + i))
        .collect();
    if e.iter().any(|&x| x <= 0) || e.windows(2).any(|w| w[1] <= w[0]) {
        return err(d, "run ends not positive and strictly increasing".into());
    }
    if !d.is_empty() && e.last().map_or(0, |&x| x as usize) < d.offset() + d.len() {
        return err(
            d,
            format!(
                "run ends cover {} rows, need {}",
                e.last().copied().unwrap_or(0),
                d.offset() + d.len()
            ),
        );
    }
    if values.len() < ends.len() {
        return err(
            d,
            format!("{} run ends but {} values", ends.len(), values.len()),
        );
    }
    Ok(())
}

/// Checks `d` and its children against the Arrow columnar format, and
/// returns the first rule it breaks
///
/// # Example
///
/// ```
/// use arrow_array::{Array, Int32Array};
/// assert!(arrow_pbt::spec::check(&Int32Array::from(vec![1, 2]).to_data()).is_ok());
/// ```
pub fn check(d: &ArrayData) -> Result<(), String> {
    // Line 324: at least one validity bit per slot
    if let Some(n) = d.nulls()
        && n.len() != d.len()
    {
        return err(
            d,
            format!("validity has {} bits for {} slots", n.len(), d.len()),
        );
    }
    let end = d.offset() + d.len();
    match d.data_type() {
        DataType::Null => {}
        DataType::Boolean => {
            if d.buffers()[0].len() * 8 < end {
                return err(d, "values bitmap too short".into());
            }
        }
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
            let width = d.data_type().primitive_width().unwrap();
            if d.buffers()[0].len() < end * width {
                return err(d, "values buffer too short".into());
            }
        }
        DataType::Time32(_) | DataType::Time64(_) | DataType::Date64 => {
            let width = d.data_type().primitive_width().unwrap();
            if d.buffers()[0].len() < end * width {
                return err(d, "values buffer too short".into());
            }
            for i in (0..d.len()).filter(|&i| valid(d, i)) {
                let v = int_at(
                    &d.buffers()[0],
                    if width == 4 {
                        &DataType::Int32
                    } else {
                        &DataType::Int64
                    },
                    d.offset() + i,
                );
                match d.data_type() {
                    // Schema.fbs lines 249-250
                    DataType::Date64 if v % 86_400_000 != 0 => {
                        return err(d, format!("slot {i}: {v} is not a whole day"));
                    }
                    // Schema.fbs lines 266-268
                    DataType::Time32(u) | DataType::Time64(u) => {
                        let per_day: i64 = match u {
                            TimeUnit::Second => 86_400,
                            TimeUnit::Millisecond => 86_400_000,
                            TimeUnit::Microsecond => 86_400_000_000,
                            TimeUnit::Nanosecond => 86_400_000_000_000,
                        };
                        if !(0..per_day).contains(&v) {
                            return err(d, format!("slot {i}: {v} is outside one day"));
                        }
                    }
                    _ => {}
                }
            }
        }
        t if t.primitive_width().is_some() => {
            if d.buffers()[0].len() < end * t.primitive_width().unwrap() {
                return err(d, "values buffer too short".into());
            }
        }
        DataType::Utf8 => var_binary::<i32>(d, true)?,
        DataType::LargeUtf8 => var_binary::<i64>(d, true)?,
        DataType::Binary => var_binary::<i32>(d, false)?,
        DataType::LargeBinary => var_binary::<i64>(d, false)?,
        DataType::Utf8View => views(d, true)?,
        DataType::BinaryView => views(d, false)?,
        DataType::FixedSizeBinary(w) => {
            if d.buffers()[0].len() < end * *w as usize {
                return err(d, "values buffer too short".into());
            }
        }
        DataType::List(_) => offsets::<i32>(d, d.child_data()[0].len())?,
        // Lines 118-138: a list of entries, where neither the entries nor
        // their keys may be null
        DataType::Map(..) => {
            offsets::<i32>(d, d.child_data()[0].len())?;
            let entries = &d.child_data()[0];
            if entries.null_count() > 0
                || entries
                    .child_data()
                    .first()
                    .is_some_and(|k| k.null_count() > 0)
            {
                return err(d, "null map entry or key".into());
            }
        }
        DataType::LargeList(_) => offsets::<i64>(d, d.child_data()[0].len())?,
        DataType::ListView(_) => list_view::<i32>(d)?,
        DataType::LargeListView(_) => list_view::<i64>(d)?,
        DataType::FixedSizeList(_, n) => {
            if d.child_data()[0].len() < end * *n as usize {
                return err(d, "child shorter than length times size".into());
            }
        }
        DataType::Struct(_) => {
            for c in d.child_data() {
                if c.len() < end {
                    return err(d, format!("child length {} < {end}", c.len()));
                }
            }
        }
        DataType::Dictionary(k, _) => {
            let width = k.primitive_width().unwrap();
            if d.buffers()[0].len() < end * width {
                return err(
                    d,
                    format!(
                        "keys buffer of {} bytes, needs {}",
                        d.buffers()[0].len(),
                        end * width
                    ),
                );
            }
            let n = d.child_data().first().map_or(0, |c| c.len()) as i64;
            let key = |i: usize| -> i64 { int_at(&d.buffers()[0], k, d.offset() + i) };
            for i in (0..d.len()).filter(|&i| valid(d, i)) {
                let k = key(i);
                if k < 0 || k >= n {
                    return err(d, format!("slot {i}: key {k} outside dictionary of {n}"));
                }
            }
            // The dictionary is an array in its own right
            return d.child_data().iter().try_for_each(check);
        }
        DataType::RunEndEncoded(..) => run_end(d)?,
        DataType::Union(fs, mode) => {
            let ids: Vec<i8> = fs.iter().map(|(id, _)| id).collect();
            union(d, &ids, *mode)?
        }
        t => return Err(format!("{t}: not covered by this checker")),
    }
    d.child_data().iter().try_for_each(check)
}

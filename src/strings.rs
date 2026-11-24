//! Properties for string kernels and casts between numbers and text

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, BooleanArray, Datum, Scalar};
use arrow_schema::{ArrowError, DataType};
use num_bigint::BigInt;
use num_traits::{Signed, Zero};
use proptest::test_runner::TestCaseError;

use crate::props::R;
use crate::props::big;
use crate::props::{as_val, decimal_scale, fail, fits, pow10};
use crate::val::check_rows;
use crate::val::{Val, same_float, vals};
use arrow_array::LargeStringArray;
use arrow_array::StringArray;
use arrow_string::concat_elements::concat_elements_dyn;
use arrow_string::length::length;
use arrow_string::substring::{substring, substring_by_char};
use std::str::from_utf8;

/// The bytes of a non-null string or binary value
fn bytes_of(v: &Val) -> Option<Vec<u8>> {
    match v {
        Val::Str(s) => Some(s.as_bytes().to_vec()),
        Val::Bytes(b) => Some(b.clone()),
        _ => None,
    }
}

/// `length`: bytes of each string or binary value, elements of each list;
/// null stays null
pub fn length_prop(a: ArrayRef) -> R {
    let Ok(out) = length(a.as_ref()) else {
        return Err(TestCaseError::reject("length does not support this type"));
    };
    let v = vals(a.as_ref());
    let got = vals(out.as_ref());
    if got.len() != v.len() {
        return Err(fail(format!("length: {} rows for {}", got.len(), v.len())));
    }
    for i in 0..v.len() {
        let want: Option<i64> = match &v[i] {
            Val::Null => None,
            Val::Str(s) => Some(s.len() as i64),
            Val::Bytes(b) => Some(b.len() as i64),
            Val::List(xs) => Some(xs.len() as i64),
            other => return Err(fail(format!("length gave a result for {other:?}"))),
        };
        let g: Option<i64> = match &got[i] {
            Val::Null => None,
            Val::I32(x) => Some(*x as i64),
            Val::I64(x) => Some(*x),
            other => return Err(fail(format!("length output {other:?}"))),
        };
        if g != want {
            return Err(fail(format!(
                "length row {i}: got {g:?}, expected {want:?} for {:?}",
                v[i]
            )));
        }
    }
    Ok(())
}

/// The characters of `s` from `start` (negative counts from the end) for
/// `len` characters, as `substring_by_char` documents. A start past either
/// end clamps to it, and the length counts from the clamped start. The
/// rustdoc does not say this, and arrow-string's code does
fn sub_chars(s: &str, start: i64, len: Option<u64>) -> String {
    let cs: Vec<char> = s.chars().collect();
    let n = cs.len() as i64;
    let from = if start >= 0 {
        start.min(n)
    } else {
        (n + start).max(0)
    } as usize;
    let to = match len {
        Some(l) => (from + l as usize).min(cs.len()),
        None => cs.len(),
    };
    cs[from..to].iter().collect()
}

/// The byte range `substring` takes from a value of `n` bytes
fn sub_range(n: usize, start: i64, len: Option<u64>) -> (usize, usize) {
    let n = n as i64;
    let from = if start >= 0 {
        start.min(n)
    } else {
        (n + start).max(0)
    };
    let to = match len {
        Some(l) => (from + l as i64).min(n),
        None => n,
    };
    (from as usize, to as usize)
}

/// SQL LIKE as `regex_like` implements it: `%` any sequence, `_` one
/// character, `\` makes the next character literal, a trailing `\` is a
/// literal backslash, and the whole value must match
pub fn like_match(value: &str, pattern: &str) -> bool {
    #[derive(Clone, Copy, PartialEq)]
    enum T {
        Lit(char),
        One,
        Any,
    }
    let mut toks = vec![];
    let mut it = pattern.chars();
    while let Some(c) = it.next() {
        toks.push(match c {
            '\\' => T::Lit(it.next().unwrap_or('\\')),
            '%' => T::Any,
            '_' => T::One,
            c => T::Lit(c),
        });
    }
    // prev[j]: whether the characters so far match the first j tokens
    let m = toks.len();
    let mut prev = vec![false; m + 1];
    prev[0] = true;
    for j in 1..=m {
        prev[j] = prev[j - 1] && toks[j - 1] == T::Any;
    }
    for c in value.chars() {
        let mut cur = vec![false; m + 1];
        for j in 1..=m {
            cur[j] = match toks[j - 1] {
                T::Lit(l) => prev[j - 1] && l == c,
                T::One => prev[j - 1],
                // `%` matches nothing more, or this character too
                T::Any => cur[j - 1] || prev[j],
            };
        }
        prev = cur;
    }
    prev[m]
}

/// Known: byte `substring` checks char boundaries in the bytes under null
/// slots too, so a null slot whose cut falls inside a character makes the
/// whole call fail (`string_view_substring` skips null slots,
/// apache/arrow-rs#11590)
fn null_slot_cut(x: &dyn Array, start: i64, len: Option<u64>) -> bool {
    let (offsets, data): (Vec<usize>, &[u8]) = match x.data_type() {
        DataType::Utf8 => (
            x.as_string::<i32>()
                .value_offsets()
                .iter()
                .map(|&o| o as usize)
                .collect(),
            x.as_string::<i32>().value_data(),
        ),
        DataType::LargeUtf8 => (
            x.as_string::<i64>()
                .value_offsets()
                .iter()
                .map(|&o| o as usize)
                .collect(),
            x.as_string::<i64>().value_data(),
        ),
        _ => return false,
    };
    let (first, last) = (offsets[0], offsets[x.len()]);
    // Not valid UTF-8 is a different failure, so it does not match
    let Ok(text) = from_utf8(&data[first..last]) else {
        return false;
    };
    (0..x.len()).any(|i| {
        let (f, t) = sub_range(offsets[i + 1] - offsets[i], start, len);
        x.is_null(i)
            && !(text.is_char_boundary(offsets[i] + f - first)
                && text.is_char_boundary(offsets[i] + t - first))
    })
}

/// String kernels against Rust's `str` operations: substring by bytes and
/// characters, prefix, suffix and containment tests, LIKE, and
/// concatenation, with array or scalar right-hand sides
pub fn string_ops((x, y, scalar, start, len): (ArrayRef, ArrayRef, bool, i64, Option<u64>)) -> R {
    use arrow_string::like;
    let (vx, vy) = (vals(x.as_ref()), vals(y.as_ref()));
    let dt = x.data_type().clone();
    let is_text = matches!(
        dt,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
    );
    let scalar = scalar && !y.is_empty();
    let rhs = |i: usize| if scalar { &vy[0] } else { &vy[i] };
    let scalar_y = scalar.then(|| Scalar::new(y.slice(0, 1)));
    let datum_y: &dyn Datum = match &scalar_y {
        Some(s) => s,
        None => &y,
    };

    let mut known = false;
    // substring by bytes; for text an error when a cut is inside a character
    if !matches!(dt, DataType::Utf8View | DataType::BinaryView) {
        let mut must_err = false;
        let expected: Vec<Val> = vx
            .iter()
            .map(|v| match v {
                Val::Null => Val::Null,
                v => {
                    let b = bytes_of(v).unwrap();
                    let (f, t) = sub_range(b.len(), start, len);
                    match v {
                        Val::Str(s) => {
                            if !s.is_char_boundary(f) || !s.is_char_boundary(t) {
                                must_err = true;
                                Val::Null
                            } else {
                                Val::Str(s[f..t].to_string())
                            }
                        }
                        _ => Val::Bytes(b[f..t].to_vec()),
                    }
                }
            })
            .collect();
        match substring(x.as_ref(), start, len) {
            Err(_) if must_err => {}
            // The other checks still run, and the case is discarded at the end
            Err(e)
                if e.to_string().contains("invalid utf-8 boundary")
                    && null_slot_cut(x.as_ref(), start, len) =>
            {
                known = true
            }
            Err(e) => return Err(fail(format!("substring({start}, {len:?}): {e}"))),
            Ok(_) if must_err => {
                return Err(fail(format!(
                    "substring({start}, {len:?}) cut inside a character without an error"
                )));
            }
            Ok(out) => check_rows(&out, &dt, &expected)
                .map_err(|e| fail(format!("substring({start}, {len:?}): {e:?}")))?,
        }
    }
    if matches!(dt, DataType::Utf8 | DataType::LargeUtf8) {
        let expected: Vec<Val> = vx
            .iter()
            .map(|v| match v {
                Val::Str(s) => Val::Str(sub_chars(s, start, len)),
                _ => Val::Null,
            })
            .collect();
        let out: Result<ArrayRef, ArrowError> = match dt {
            DataType::Utf8 => substring_by_char(
                x.as_any().downcast_ref::<StringArray>().unwrap(),
                start,
                len,
            )
            .map(|a| Arc::new(a) as ArrayRef),
            _ => substring_by_char(
                x.as_any().downcast_ref::<LargeStringArray>().unwrap(),
                start,
                len,
            )
            .map(|a| Arc::new(a) as ArrayRef),
        };
        let out = out.map_err(|e| fail(format!("substring_by_char: {e}")))?;
        check_rows(&out, &dt, &expected)
            .map_err(|e| fail(format!("substring_by_char({start}, {len:?}): {e:?}")))?;
    }

    // Predicates on text
    if is_text {
        type Op = fn(&dyn Datum, &dyn Datum) -> Result<BooleanArray, ArrowError>;
        // A kernel, and the same predicate on Rust strings
        type Pred = (&'static str, Op, fn(&str, &str) -> bool);
        let preds: [Pred; 5] = [
            ("starts_with", like::starts_with, |a, b| a.starts_with(b)),
            ("ends_with", like::ends_with, |a, b| a.ends_with(b)),
            ("contains", like::contains, |a, b| a.contains(b)),
            ("like", like::like, |a, b| like_match(a, b)),
            ("nlike", like::nlike, |a, b| !like_match(a, b)),
        ];
        for (name, op, f) in preds {
            let out = op(&x, datum_y).map_err(|e| fail(format!("{name}: {e}")))?;
            let expected: Vec<Val> = (0..x.len())
                .map(|i| match (&vx[i], rhs(i)) {
                    (Val::Str(a), Val::Str(b)) => Val::Bool(f(a, b)),
                    _ => Val::Null,
                })
                .collect();
            let out: ArrayRef = Arc::new(out);
            check_rows(&out, &DataType::Boolean, &expected).map_err(|e| {
                fail(format!(
                    "{name}{}: {e:?}",
                    if scalar { " with a scalar pattern" } else { "" }
                ))
            })?;
        }
    }

    // Concatenation of two arrays of the same type
    if !scalar {
        // Errors are documented only for arrays of different types, and both
        // sides here have one type
        {
            let out = concat_elements_dyn(x.as_ref(), y.as_ref())
                .map_err(|e| fail(format!("concat_elements: {e}")))?;
            let expected: Vec<Val> = (0..x.len())
                .map(|i| match (&vx[i], &vy[i]) {
                    (Val::Str(a), Val::Str(b)) => Val::Str(format!("{a}{b}")),
                    (Val::Bytes(a), Val::Bytes(b)) => {
                        Val::Bytes([a.as_slice(), b.as_slice()].concat())
                    }
                    _ => Val::Null,
                })
                .collect();
            check_rows(&out, &dt, &expected)
                .map_err(|e| fail(format!("concat_elements: {e:?}")))?;
        }
    }
    if known {
        return Err(crate::known::known(
            "known: substring checks char boundaries under null slots (apache/arrow-rs#11590)",
        ));
    }
    Ok(())
}

/// Text accepted by `parse_decimal`'s documented grammar, as a mantissa
/// (all digits, sign applied) and the power of ten it is multiplied by
fn decimal_text(s: &str) -> Option<(BigInt, i64)> {
    let t = s.trim_matches(|c: char| c.is_ascii_whitespace());
    let (neg, t) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    let (mant, exp) = match t.find(['e', 'E']) {
        Some(k) => (&t[..k], Some(&t[k + 1..])),
        None => (t, None),
    };
    let (int, frac) = match mant.split_once('.') {
        Some((a, b)) => (a, b),
        None => (mant, ""),
    };
    let digits = |x: &str| x.bytes().all(|b| b.is_ascii_digit());
    if !digits(int) || !digits(frac) || int.len() + frac.len() == 0 {
        return None;
    }
    let e: i64 = match exp {
        None => 0,
        Some(e) => {
            let (en, ed) = match e.as_bytes().first() {
                Some(b'-') => (true, &e[1..]),
                Some(b'+') => (false, &e[1..]),
                _ => (false, e),
            };
            if ed.is_empty() || !digits(ed) {
                return None;
            }
            // Exponents this large only matter for zero, which they leave zero
            let v: i64 = ed.parse::<i64>().unwrap_or(i64::MAX / 4).min(i64::MAX / 4);
            if en { -v } else { v }
        }
    };
    let m: BigInt = format!("{int}{frac}")
        .trim_start_matches('0')
        .parse()
        .unwrap_or_else(|_| BigInt::zero());
    Some((if neg { -m } else { m }, e - frac.len() as i64))
}

/// Casts between numbers and text. Numbers to text: integers must match
/// `to_string`; floats and decimals must round-trip to the same value. Text
/// to numbers: integers as Rust parses trimmed text, decimals by the
/// grammar `parse_decimal` documents with half-away-from-zero rounding, and
/// floats where Rust parses the text, correctly rounded
pub fn cast_text((x, to, safe): (ArrayRef, DataType, bool)) -> R {
    use arrow_cast::cast::{CastOptions, cast_with_options};
    let from = x.data_type().clone();
    let opts = CastOptions {
        safe,
        ..Default::default()
    };
    let v = vals(x.as_ref());
    let text_target = matches!(
        to,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
    );
    let ctx = |m: String| fail(format!("{from} to {to} (safe: {safe}): {m}"));
    if text_target {
        let out = cast_with_options(&x, &to, &opts).map_err(|e| ctx(format!("error {e}")))?;
        if from.is_integer() {
            let expected: Vec<Val> = v
                .iter()
                .map(|r| match r {
                    Val::Null => Val::Null,
                    r => Val::Str(big(&from, r).to_string()),
                })
                .collect();
            return check_rows(&out, &to, &expected).map_err(|e| ctx(format!("{e:?}")));
        }
        // Floats and decimals: back to the same type gives the same value
        let back = match cast_with_options(
            &out,
            &from,
            &CastOptions {
                safe: false,
                ..Default::default()
            },
        ) {
            Ok(b) => b,
            Err(e) => {
                let past = v.iter().any(|r| match (&from, r) {
                    (
                        DataType::Decimal32(p, _)
                        | DataType::Decimal64(p, _)
                        | DataType::Decimal128(p, _)
                        | DataType::Decimal256(p, _),
                        Val::Raw(_),
                    ) => big(&from, r).abs() >= pow10(*p as i32),
                    _ => false,
                });
                if past {
                    return Ok(());
                }
                return Err(ctx(format!("cast back from text: {e}")));
            }
        };
        let got = vals(back.as_ref());
        // Decimals past their declared precision print, but cannot be cast
        // back; what a cast should do with them is apache/arrow-rs#10947
        let past_precision = |r: &Val| match (&from, r) {
            (
                DataType::Decimal32(p, _)
                | DataType::Decimal64(p, _)
                | DataType::Decimal128(p, _)
                | DataType::Decimal256(p, _),
                Val::Raw(_),
            ) => big(&from, r).abs() >= pow10(*p as i32),
            _ => false,
        };
        if v.iter().any(past_precision) {
            return Ok(());
        }
        for i in 0..v.len() {
            let same = match (&v[i], &got[i]) {
                (a @ Val::Raw(_), b @ Val::Raw(_))
                    if matches!(from, DataType::Float32 | DataType::Float64) =>
                {
                    same_float(a.as_f64(), b.as_f64())
                }
                (a, b) => a == b,
            };
            if !same {
                return Err(ctx(format!(
                    "row {i}: {:?} became {:?} and came back as {:?}",
                    v[i],
                    vals(out.as_ref())[i],
                    got[i]
                )));
            }
        }
        return Ok(());
    }
    // Text to a number
    let mut expected = Vec::with_capacity(v.len());
    let mut open = vec![false; v.len()];
    let mut unrepresentable = false;
    for (i, r) in v.iter().enumerate() {
        let Val::Str(s) = r else {
            expected.push(Val::Null);
            continue;
        };
        let want: Option<Option<Val>> = if to.is_integer() {
            let t = s.trim_matches(|c: char| c.is_ascii_whitespace());
            let unsigned = t.strip_prefix(['+', '-']).unwrap_or(t);
            let ok = !unsigned.is_empty() && unsigned.bytes().all(|b| b.is_ascii_digit());
            Some(if ok {
                let n: BigInt = t.parse().unwrap();
                fits(&to, &n).then(|| as_val(&to, &n))
            } else {
                None
            })
        } else if let Some(so) = decimal_scale(&to) {
            let p = match to {
                DataType::Decimal32(p, _)
                | DataType::Decimal64(p, _)
                | DataType::Decimal128(p, _)
                | DataType::Decimal256(p, _) => p as i64,
                _ => unreachable!(),
            };
            Some(decimal_text(s).and_then(|(m, e)| {
                // The value is m * 10^e; stored at scale so as m * 10^(e + so)
                let k = e + so as i64;
                let digits = m.abs().to_string().len() as i64;
                let n = if m.is_zero() {
                    BigInt::zero()
                } else if k >= 0 {
                    if digits + k > p + 1 {
                        return None;
                    }
                    &m * pow10(k as i32)
                } else if -k > digits + 1 {
                    BigInt::zero()
                } else {
                    let d = pow10((-k) as i32);
                    let (q, rem) = (&m / &d, &m % &d);
                    if BigInt::from(2) * rem.abs() >= d {
                        q + m.signum()
                    } else {
                        q
                    }
                };
                (n.abs() < pow10(p as i32) && fits(&to, &n)).then(|| as_val(&to, &n))
            }))
        } else {
            // Floats: only where Rust parses the trimmed text
            let t = s.trim_matches(|c: char| c.is_ascii_whitespace());
            match to {
                DataType::Float64 => t
                    .parse::<f64>()
                    .ok()
                    .map(|f| Some(Val::Raw(f.to_le_bytes().to_vec()))),
                DataType::Float32 => t
                    .parse::<f32>()
                    .ok()
                    .map(|f| Some(Val::Raw(f.to_le_bytes().to_vec()))),
                _ => None,
            }
        };
        match want {
            Some(Some(e)) => expected.push(e),
            Some(None) => {
                unrepresentable = true;
                expected.push(Val::Null)
            }
            None => {
                open[i] = true;
                expected.push(Val::Null)
            }
        }
    }
    match cast_with_options(&x, &to, &opts) {
        Err(_) if !safe && (unrepresentable || open.iter().any(|o| *o)) => Ok(()),
        Err(e) => Err(ctx(format!("unexpected error {e}"))),
        Ok(_) if !safe && unrepresentable => Err(ctx(
            "expected an error for text the target cannot represent".into(),
        )),
        Ok(out) => {
            let got = vals(out.as_ref());
            for i in 0..v.len() {
                if open[i] {
                    continue;
                }
                let same = match (&got[i], &expected[i]) {
                    (g @ Val::Raw(a), e @ Val::Raw(_))
                        if matches!(
                            (a.len(), &to),
                            (8, DataType::Float64) | (4, DataType::Float32)
                        ) =>
                    {
                        same_float(g.as_f64(), e.as_f64())
                    }
                    (g, e) => g == e,
                };
                if !same {
                    return Err(ctx(format!(
                        "row {i}: {:?} gave {:?}, expected {:?}",
                        v[i], got[i], expected[i]
                    )));
                }
            }
            Ok(())
        }
    }
}

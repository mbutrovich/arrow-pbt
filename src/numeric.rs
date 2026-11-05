//! Properties for `neg`, the arity helpers, the bitwise kernels, the
//! aggregate kernels, fixed-point decimal multiplication, and decimal
//! formatting

use arrow_arith::aggregate as agg;
use arrow_array::cast::AsArray;
use arrow_array::types::*;
use arrow_array::{
    Array, ArrayAccessor, ArrayRef, ArrowNativeTypeOp, ArrowNumericType, ArrowPrimitiveType,
    PrimitiveArray,
};
use arrow_buffer::{ArrowNativeType, NullBuffer, ScalarBuffer};
use arrow_schema::{ArrowError, DataType};
use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{Signed, ToPrimitive, WrappingShl, WrappingShr};
use proptest::test_runner::TestCaseError;
use std::cell::Cell;
use std::ops::{BitAnd, BitOr, BitXor, Not};

use crate::known::{known, tolerated};
use crate::props::R;
use crate::props::decimal_scale;
use crate::props::interval_fields;
use crate::props::round_half_away;
use crate::props::{as_val, big, check_in, fail, fits, pow10, wrap};
use crate::run::caught;
use crate::val::check_rows;
use crate::val::{Val, val_at, vals};

/// `neg` negates with an error on overflow and rejects unsigned types, as
/// documented. `neg_wrapping` wraps for integer types and is `neg` otherwise
pub fn neg_prop(a: ArrayRef) -> R {
    use arrow_arith::numeric::{neg, neg_wrapping};
    check_in(a.as_ref())?;
    let dt = a.data_type().clone();
    let v = vals(a.as_ref());
    let unsigned = matches!(
        dt,
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64
    );
    // Expected rows, or None when a valid row overflows
    let (checked, wrapping): (Option<Vec<Val>>, Vec<Val>) = if dt.is_floating() {
        // IEEE negation flips the sign bit, NaN included
        let flip = |x: &Val| match x {
            Val::Raw(b) => {
                let mut b = b.clone();
                *b.last_mut().unwrap() ^= 0x80;
                Val::Raw(b)
            }
            x => x.clone(),
        };
        let rows: Vec<Val> = v.iter().map(flip).collect();
        (Some(rows.clone()), rows)
    } else if let DataType::Interval(_) = dt {
        // Each field negates on its own, with an error if any overflows
        let neg_fields = |x: &Val| -> Option<Val> {
            let Val::Raw(b) = x else {
                return Some(Val::Null);
            };
            let f = interval_fields(b);
            let mut out = vec![];
            for (k, v) in f.iter().enumerate() {
                let n = v.checked_neg()?;
                if k < 2 || b.len() == 4 {
                    out.extend(i32::try_from(n).ok()?.to_le_bytes());
                } else {
                    out.extend(n.to_le_bytes());
                }
            }
            Some(Val::Raw(out))
        };
        let checked: Option<Vec<Val>> = v.iter().map(neg_fields).collect();
        (checked.clone(), checked.unwrap_or_default())
    } else {
        let negated = |x: &Val| (x != &Val::Null).then(|| -big(&dt, x));
        let checked = v.iter().map(|x| match negated(x) {
            None => Some(Val::Null),
            Some(n) => fits(&dt, &n).then(|| as_val(&dt, &n)),
        });
        let checked: Option<Vec<Val>> = if unsigned { None } else { checked.collect() };
        let wrapping = if dt.is_integer() {
            v.iter()
                .map(|x| negated(x).map_or(Val::Null, |n| as_val(&dt, &wrap(&dt, &n))))
                .collect()
        } else {
            checked.clone().unwrap_or_default()
        };
        (checked, wrapping)
    };
    match (neg(a.as_ref()), &checked) {
        (Err(_), None) => {}
        (Err(e), Some(_)) => return Err(fail(format!("neg: unexpected error {e}"))),
        (Ok(_), None) => {
            return Err(fail(format!(
                "neg: no error for {}",
                if unsigned {
                    "an unsigned type"
                } else {
                    "a row that overflows"
                }
            )));
        }
        (Ok(out), Some(rows)) => {
            check_rows(&out, &dt, rows).map_err(|e| fail(format!("neg: {e:?}")))?
        }
    }
    let wrap_ok = dt.is_integer() || checked.is_some();
    match neg_wrapping(a.as_ref()) {
        Err(_) if !wrap_ok => Ok(()),
        Err(e) => Err(fail(format!("neg_wrapping: unexpected error {e}"))),
        Ok(_) if !wrap_ok => Err(fail("neg_wrapping: no error for a row that overflows")),
        Ok(out) => {
            check_rows(&out, &dt, &wrapping).map_err(|e| fail(format!("neg_wrapping: {e:?}")))
        }
    }
}

/// The arity helpers against closures with known results. A row is null
/// when an input row is. `unary` and `binary` call `op` on every slot, null
/// or not, and the `try_` forms and `unary_opt` only on valid rows, as
/// documented. Arrays of different lengths are an error. The `_mut` forms
/// work in place on buffers no other array references, and return the array
/// unchanged when one does
pub fn arity_prop((a, b): (ArrayRef, ArrayRef)) -> R {
    check_in(a.as_ref())?;
    check_in(b.as_ref())?;
    use DataType::*;
    match a.data_type() {
        Int8 => arity::<Int8Type>(a, b),
        Int16 => arity::<Int16Type>(a, b),
        Int32 => arity::<Int32Type>(a, b),
        Int64 => arity::<Int64Type>(a, b),
        UInt8 => arity::<UInt8Type>(a, b),
        UInt16 => arity::<UInt16Type>(a, b),
        UInt32 => arity::<UInt32Type>(a, b),
        UInt64 => arity::<UInt64Type>(a, b),
        d => Err(fail(format!("arity: unexpected type {d}"))),
    }
}

/// A copy of `p` in buffers of its own, at offset 0
fn unshared<T: ArrowPrimitiveType>(p: &PrimitiveArray<T>) -> PrimitiveArray<T> {
    let nulls = p
        .nulls()
        .map(|n| NullBuffer::from(n.iter().collect::<Vec<bool>>()));
    PrimitiveArray::new(ScalarBuffer::from(p.values().to_vec()), nulls)
}

fn arity<T: ArrowPrimitiveType>(a: ArrayRef, b: ArrayRef) -> R
where
    T::Native: ArrowNativeTypeOp,
{
    use arrow_arith::arity::{
        binary, binary_mut, try_binary, try_binary_mut, try_unary, try_unary_mut, unary, unary_mut,
    };
    let dt = a.data_type().clone();
    let (va, vb) = (vals(a.as_ref()), vals(b.as_ref()));
    let n = a.len();
    // `x + 1` and `x + y`, wrapped, for valid rows, and whether a valid row
    // overflows
    let inc_of = |x: &Val| (x != &Val::Null).then(|| big(&dt, x) + 1);
    let sum_of =
        |x: &Val, y: &Val| (x != &Val::Null && y != &Val::Null).then(|| big(&dt, x) + big(&dt, y));
    let wrapped = |v: Option<BigInt>| v.map_or(Val::Null, |v| as_val(&dt, &wrap(&dt, &v)));
    let incs: Vec<Option<BigInt>> = va.iter().map(inc_of).collect();
    let sums: Vec<Option<BigInt>> = va.iter().zip(&vb).map(|(x, y)| sum_of(x, y)).collect();
    let overflows = |vs: &[Option<BigInt>]| vs.iter().flatten().any(|v| !fits(&dt, v));
    let (inc_over, sum_over) = (overflows(&incs), overflows(&sums));
    let inc: Vec<Val> = incs.iter().cloned().map(wrapped).collect();
    let sum: Vec<Val> = sums.iter().cloned().map(wrapped).collect();
    // `unary_opt` makes an overflowing row null instead
    let inc_opt: Vec<Val> = incs
        .iter()
        .map(|v| match v {
            Some(v) if fits(&dt, v) => as_val(&dt, v),
            _ => Val::Null,
        })
        .collect();
    let valid_a = va.iter().filter(|x| **x != Val::Null).count();
    let valid_ab = sums.iter().flatten().count();

    let calls = Cell::new(0usize);
    let one = T::Native::ONE;
    let inc_w = |x: T::Native| {
        calls.set(calls.get() + 1);
        x.add_wrapping(one)
    };
    let inc_c = |x: T::Native| {
        calls.set(calls.get() + 1);
        x.add_checked(one)
    };
    let sum_w = |x: T::Native, y: T::Native| {
        calls.set(calls.get() + 1);
        x.add_wrapping(y)
    };
    let sum_c = |x: T::Native, y: T::Native| {
        calls.set(calls.get() + 1);
        x.add_checked(y)
    };
    let check = |name: &str, got: &PrimitiveArray<T>, want: &[Val]| {
        check_rows(got, &dt, want).map_err(|e| fail(format!("{name}: {e:?}")))
    };
    let called = |name: &str, want: usize| match calls.replace(0) {
        c if c == want => Ok(()),
        c => Err(fail(format!("{name} called op {c} times, expected {want}"))),
    };
    // An error exactly when a valid row overflows, and otherwise `want`, with
    // `op` called once per valid row
    let fallible =
        |name: &str, got: Result<PrimitiveArray<T>, ArrowError>, over, want, valid| match (
            got, over,
        ) {
            (Err(_), true) => {
                calls.set(0);
                Ok(())
            }
            (Err(e), false) => Err(fail(format!("{name}: unexpected error {e}"))),
            (Ok(_), true) => Err(fail(format!("{name}: no error for a row that overflows"))),
            (Ok(out), false) => {
                check(name, &out, want)?;
                called(name, valid)
            }
        };
    let (pa, pb) = (a.as_primitive::<T>(), b.as_primitive::<T>());

    check("unary", &unary(pa, inc_w), &inc)?;
    called("unary", n)?;
    check(
        "unary_opt",
        &pa.unary_opt::<_, T>(|x| inc_c(x).ok()),
        &inc_opt,
    )?;
    called("unary_opt", valid_a)?;
    fallible("try_unary", try_unary(pa, inc_c), inc_over, &inc, valid_a)?;
    let out = binary(pa, pb, sum_w).map_err(|e| fail(format!("binary: {e}")))?;
    check("binary", &out, &sum)?;
    called("binary", n)?;
    fallible(
        "try_binary",
        try_binary(pa, pb, sum_c),
        sum_over,
        &sum,
        valid_ab,
    )?;
    if n > 0 {
        let short = pb.slice(1, n - 1);
        if binary::<_, _, _, T>(pa, &short, sum_w).is_ok()
            || try_binary::<_, _, _, T>(pa, &short, sum_c).is_ok()
        {
            return Err(fail("binary of different lengths returned an array"));
        }
        calls.set(0);
    }

    // In place on buffers no other array references
    match unary_mut(unshared(pa), inc_w) {
        Ok(out) => check("unary_mut", &out, &inc)?,
        Err(_) => return Err(fail("unary_mut: Err for buffers no other array references")),
    }
    called("unary_mut", n)?;
    match try_unary_mut(unshared(pa), inc_c) {
        Ok(got) => fallible("try_unary_mut", got, inc_over, &inc, valid_a)?,
        Err(_) => {
            return Err(fail(
                "try_unary_mut: Err for buffers no other array references",
            ));
        }
    }
    match binary_mut(unshared(pa), pb, sum_w) {
        Ok(Ok(out)) => check("binary_mut", &out, &sum)?,
        Ok(Err(e)) => return Err(fail(format!("binary_mut: unexpected error {e}"))),
        Err(_) => {
            return Err(fail(
                "binary_mut: Err for buffers no other array references",
            ));
        }
    }
    called("binary_mut", n)?;
    match try_binary_mut(unshared(pa), pb, sum_c) {
        Ok(got) => fallible("try_binary_mut", got, sum_over, &sum, valid_ab)?,
        Err(_) => {
            return Err(fail(
                "try_binary_mut: Err for buffers no other array references",
            ));
        }
    }
    if n > 0 {
        let short = pb.slice(1, n - 1);
        let long = matches!(binary_mut(unshared(pa), &short, sum_w), Ok(Ok(_)))
            || matches!(try_binary_mut(unshared(pa), &short, sum_c), Ok(Ok(_)));
        if long {
            return Err(fail("binary_mut of different lengths returned an array"));
        }
        calls.set(0);
    }

    // A shared array comes back unchanged, as `Err(self)`. `a` still
    // references `pa`'s buffers
    let unchanged = |name: &str, back: PrimitiveArray<T>| match back == *pa {
        true => Ok(()),
        false => Err(fail(format!("{name}: a shared array came back changed"))),
    };
    let shared_ok = |name: &str| fail(format!("{name}: Ok for a shared array of {n} rows"));
    // Known: the binary forms return `Ok` with a new empty array for an empty
    // shared array, before checking whether its buffer is shared. Found by
    // this harness on main at 9ffd63d528 (apache/arrow-rs#12002)
    let empty_ok = || {
        crate::known::known(
            "known: binary_mut returns Ok for an empty shared array (apache/arrow-rs#12002)",
        )
    };
    match unary_mut(pa.clone(), inc_w) {
        Err(back) => unchanged("unary_mut", back)?,
        Ok(_) => return Err(shared_ok("unary_mut")),
    }
    match try_unary_mut(pa.clone(), inc_c) {
        Err(back) => unchanged("try_unary_mut", back)?,
        Ok(_) => return Err(shared_ok("try_unary_mut")),
    }
    match binary_mut(pa.clone(), pb, sum_w) {
        Err(back) => unchanged("binary_mut", back)?,
        Ok(_) if n == 0 => return Err(empty_ok()),
        Ok(_) => return Err(shared_ok("binary_mut")),
    }
    match try_binary_mut(pa.clone(), pb, sum_c) {
        Err(back) => unchanged("try_binary_mut", back)?,
        Ok(_) if n == 0 => return Err(empty_ok()),
        Ok(_) => return Err(shared_ok("try_binary_mut")),
    }
    check("the shared array after the _mut forms", pa, &va)?;
    calls.set(0);

    // The generated array as its only owner. Its buffers can still be
    // sliced, imported or shared with the generator, which `Buffer::into_mutable`
    // documents as reasons to return it unchanged
    let owned = pa.clone();
    drop(a);
    match unary_mut(owned, inc_w) {
        Ok(out) => {
            check("unary_mut of the generated array", &out, &inc)?;
            called("unary_mut of the generated array", n)
        }
        Err(back) => check("unary_mut of the generated array, returned", &back, &va),
    }
}

/// The bitwise kernels against exact integers in two's complement. A row is
/// null when an input row is. A shift is `left << right` or `left >> right`
/// for an amount from 0 to one less than the bit width. The rustdoc does not
/// say what other amounts do, so those rows are checked only for being
/// valid
pub fn bitwise_prop((a, b): (ArrayRef, ArrayRef)) -> R {
    check_in(a.as_ref())?;
    check_in(b.as_ref())?;
    use DataType::*;
    match a.data_type() {
        Int8 => bitwise::<Int8Type>(&a, &b),
        Int16 => bitwise::<Int16Type>(&a, &b),
        Int32 => bitwise::<Int32Type>(&a, &b),
        Int64 => bitwise::<Int64Type>(&a, &b),
        UInt8 => bitwise::<UInt8Type>(&a, &b),
        UInt16 => bitwise::<UInt16Type>(&a, &b),
        UInt32 => bitwise::<UInt32Type>(&a, &b),
        UInt64 => bitwise::<UInt64Type>(&a, &b),
        d => Err(fail(format!("bitwise: unexpected type {d}"))),
    }
}

fn bitwise<T: ArrowNumericType>(a: &ArrayRef, b: &ArrayRef) -> R
where
    T::Native: BitAnd<Output = T::Native>
        + BitOr<Output = T::Native>
        + BitXor<Output = T::Native>
        + Not<Output = T::Native>
        + WrappingShl<Output = T::Native>
        + WrappingShr<Output = T::Native>,
{
    use arrow_arith::bitwise::*;
    let dt = a.data_type().clone();
    let bits = dt.primitive_width().unwrap() * 8;
    let (va, vb) = (vals(a.as_ref()), vals(b.as_ref()));
    let (pa, pb) = (a.as_primitive::<T>(), b.as_primitive::<T>());
    // The scalar is the second array's first value, even under a null
    let scalar = pb.values().first().copied().unwrap_or_default();
    let vs = as_oracle::<T>(&dt, scalar);
    // In-range shift amounts, from the second array's values
    let amounts = PrimitiveArray::<T>::new(
        vb.iter()
            .map(|y| match y {
                Val::Null => T::Native::default(),
                y => T::Native::from_usize(
                    big(&dt, y)
                        .mod_floor(&BigInt::from(bits))
                        .to_usize()
                        .unwrap(),
                )
                .unwrap(),
            })
            .collect::<Vec<_>>()
            .into(),
        pb.nulls().cloned(),
    );
    let va_amounts = vals(&amounts);
    // Expected rows of `f` on two rows, where `None` from `f` means the value
    // is not checked
    let model = |ys: &dyn Fn(usize) -> Val, f: &dyn Fn(BigInt, BigInt) -> Option<BigInt>| {
        (0..va.len())
            .map(|i| match (&va[i], ys(i)) {
                (Val::Null, _) | (_, Val::Null) => None,
                (x, y) => Some(f(big(&dt, x), big(&dt, &y)).map(|v| as_val(&dt, &wrap(&dt, &v)))),
            })
            .collect::<Vec<Option<Option<Val>>>>()
    };
    let check =
        |name: &str, got: Result<PrimitiveArray<T>, ArrowError>, want: Vec<Option<Option<Val>>>| {
            let got = got.map_err(|e| fail(format!("{name}: {e}")))?;
            let gv = vals(&got);
            let want: Vec<Val> = want
                .into_iter()
                .enumerate()
                .map(|(i, w)| match w {
                    None => Val::Null,
                    Some(Some(v)) => v,
                    // Not checked, but it must be a value
                    Some(None) if gv.get(i).is_some_and(|g| *g != Val::Null) => gv[i].clone(),
                    Some(None) => as_val(&dt, &BigInt::from(0)),
                })
                .collect();
            check_rows(&got, &dt, &want).map_err(|e| fail(format!("{name}: {e:?}")))
        };
    let shift = |s: &BigInt| {
        (*s >= BigInt::from(0) && *s < BigInt::from(bits)).then(|| s.to_usize().unwrap())
    };
    let shl = |x: BigInt, s: BigInt| shift(&s).map(|s| x << s);
    let shr = |x: BigInt, s: BigInt| shift(&s).map(|s| x >> s);
    let by_row = |i: usize| vb[i].clone();
    let by_amount = |i: usize| va_amounts[i].clone();
    let by_scalar = |_: usize| vs.clone();

    check(
        "bitwise_and",
        bitwise_and(pa, pb),
        model(&by_row, &|x, y| Some(x & y)),
    )?;
    check(
        "bitwise_or",
        bitwise_or(pa, pb),
        model(&by_row, &|x, y| Some(x | y)),
    )?;
    check(
        "bitwise_xor",
        bitwise_xor(pa, pb),
        model(&by_row, &|x, y| Some(x ^ y)),
    )?;
    check(
        "bitwise_and_not",
        bitwise_and_not(pa, pb),
        model(&by_row, &|x, y| Some(x & !y)),
    )?;
    check(
        "bitwise_not",
        bitwise_not(pa),
        model(&by_scalar, &|x, _| Some(!x)),
    )?;
    check(
        "bitwise_shift_left",
        bitwise_shift_left(pa, pb),
        model(&by_row, &shl),
    )?;
    check(
        "bitwise_shift_right",
        bitwise_shift_right(pa, pb),
        model(&by_row, &shr),
    )?;
    check(
        "bitwise_shift_left of in-range amounts",
        bitwise_shift_left(pa, &amounts),
        model(&by_amount, &shl),
    )?;
    check(
        "bitwise_shift_right of in-range amounts",
        bitwise_shift_right(pa, &amounts),
        model(&by_amount, &shr),
    )?;
    check(
        "bitwise_and_scalar",
        bitwise_and_scalar(pa, scalar),
        model(&by_scalar, &|x, y| Some(x & y)),
    )?;
    check(
        "bitwise_or_scalar",
        bitwise_or_scalar(pa, scalar),
        model(&by_scalar, &|x, y| Some(x | y)),
    )?;
    check(
        "bitwise_xor_scalar",
        bitwise_xor_scalar(pa, scalar),
        model(&by_scalar, &|x, y| Some(x ^ y)),
    )?;
    check(
        "bitwise_shift_left_scalar",
        bitwise_shift_left_scalar(pa, scalar),
        model(&by_scalar, &shl),
    )?;
    check(
        "bitwise_shift_right_scalar",
        bitwise_shift_right_scalar(pa, scalar),
        model(&by_scalar, &shr),
    )?;
    if !va.is_empty() {
        let short = pb.slice(1, va.len() - 1);
        if bitwise_and(pa, &short).is_ok() {
            return Err(fail("bitwise_and of different lengths returned an array"));
        }
    }
    Ok(())
}

/// The exact result of an aggregate: a value (`None` when there are no
/// non-null values), or an overflow of the result type
#[derive(Clone, Debug)]
enum Exact {
    Fits(Option<Val>),
    Overflows,
}

/// Expected integer or decimal aggregates over the non-null values `xs` of
/// type `dt`, in row order
struct Folds {
    sum: Option<Val>,
    sum_exact: Exact,
    /// Whether a prefix of the row-order sum overflows, which allows an error
    /// even when the total fits
    sum_prefix_overflows: bool,
    product: Option<Val>,
    product_exact: Exact,
    product_prefix_overflows: bool,
    min: Option<Val>,
    max: Option<Val>,
}

fn int_folds(dt: &DataType, xs: &[BigInt]) -> Folds {
    let fold = |init: i32, f: fn(BigInt, &BigInt) -> BigInt| -> (Option<Val>, Exact, bool) {
        if xs.is_empty() {
            return (None, Exact::Fits(None), false);
        }
        let mut acc = BigInt::from(init);
        let mut prefix_overflows = false;
        for x in xs {
            acc = f(acc, x);
            prefix_overflows |= !fits(dt, &acc);
        }
        let exact = if fits(dt, &acc) {
            Exact::Fits(Some(as_val(dt, &acc)))
        } else {
            Exact::Overflows
        };
        (Some(as_val(dt, &wrap(dt, &acc))), exact, prefix_overflows)
    };
    let (sum, sum_exact, sum_prefix_overflows) = fold(0, |a, x| a + x);
    let (product, product_exact, product_prefix_overflows) = fold(1, |a, x| a * x);
    Folds {
        sum,
        sum_exact,
        sum_prefix_overflows,
        product,
        product_exact,
        product_prefix_overflows,
        min: xs.iter().min().map(|m| as_val(dt, m)),
        max: xs.iter().max().map(|m| as_val(dt, m)),
    }
}

/// A checked aggregate's result against its exact value. An error is
/// required when the exact value does not fit, and allowed when it fits but
/// the kernel's evaluation order overflows on the way
fn check_checked(
    name: &str,
    got: Result<Option<Val>, String>,
    exact: &Exact,
    order_overflows: bool,
) -> R {
    match (got, exact) {
        (Err(_), Exact::Overflows) => Ok(()),
        (Err(_), Exact::Fits(_)) if order_overflows => Ok(()),
        (Err(e), Exact::Fits(_)) => Err(fail(format!("{name}: unexpected error {e}"))),
        (Ok(g), Exact::Overflows) => Err(fail(format!(
            "{name}: {g:?} for a result that does not fit"
        ))),
        (Ok(g), Exact::Fits(w)) if &g != w => {
            Err(fail(format!("{name}: got {g:?}, expected {w:?}")))
        }
        _ => Ok(()),
    }
}

fn same(name: &str, got: Option<Val>, want: Option<&Val>) -> R {
    if got.as_ref() != want {
        return Err(fail(format!("{name}: got {got:?}, expected {want:?}")));
    }
    Ok(())
}

/// The oracle's value of `x` as a value of type `dt`
fn as_oracle<T: ArrowPrimitiveType>(dt: &DataType, x: T::Native) -> Val {
    val_at(
        &PrimitiveArray::<T>::from_iter_values([x]).with_data_type(dt.clone()),
        0,
    )
}

/// Floats: min and max by IEEE total order, which apache/arrow-rs#5031
/// settled on. It puts a NaN with the sign bit set below every other value,
/// although the rustdoc says NaN is greater than any other value
fn float_min_max(name: &str, got: Option<f64>, xs: &[f64], want_max: bool) -> R {
    let want = if want_max {
        xs.iter().copied().max_by(f64::total_cmp)
    } else {
        xs.iter().copied().min_by(f64::total_cmp)
    };
    // Neither apache/arrow-rs#5031 nor the rustdoc says which NaN payload wins, so any NaN
    // of the expected sign matches
    let key = |x: f64| {
        if x.is_nan() {
            (true, x.is_sign_negative(), 0)
        } else {
            (false, false, x.to_bits())
        }
    };
    if got.map(key) != want.map(key) {
        return Err(fail(format!("{name}: got {got:?}, expected {want:?}")));
    }
    Ok(())
}

/// `sum`, `sum_checked`, `product`, `product_checked`, `min` and `max` on a
/// primitive array
fn primitive<T: ArrowNumericType>(a: &dyn Array, v: &[Val]) -> R {
    let p = a.as_primitive::<T>();
    let dt = a.data_type();
    let valid: Vec<&Val> = v.iter().filter(|x| **x != Val::Null).collect();
    let to_val = |x: Option<T::Native>| x.map(|x| as_oracle::<T>(dt, x));
    if dt.is_floating() {
        let xs: Vec<f64> = valid.iter().map(|x| x.as_f64()).collect();
        let f = |x: Option<T::Native>| to_val(x).map(|v| v.as_f64());
        float_min_max("min", f(agg::min(p)), &xs, false)?;
        float_min_max("max", f(agg::max(p)), &xs, true)?;
        // Float sums depend on the order of addition, so only their presence
        // is checked
        if agg::sum(p).is_some() == xs.is_empty() {
            return Err(fail(format!(
                "sum: {:?} for {} valid rows",
                agg::sum(p),
                xs.len()
            )));
        }
        return Ok(());
    }
    let xs: Vec<BigInt> = valid.iter().map(|x| big(dt, x)).collect();
    let f = int_folds(dt, &xs);
    same("sum", to_val(agg::sum(p)), f.sum.as_ref())?;
    same("product", to_val(agg::product(p)), f.product.as_ref())?;
    same("min", to_val(agg::min(p)), f.min.as_ref())?;
    same("max", to_val(agg::max(p)), f.max.as_ref())?;
    let sc = agg::sum_checked(p).map(to_val).map_err(|e| e.to_string());
    check_checked("sum_checked", sc, &f.sum_exact, f.sum_prefix_overflows)?;
    let pc = agg::product_checked(p)
        .map(to_val)
        .map_err(|e| e.to_string());
    check_checked(
        "product_checked",
        pc,
        &f.product_exact,
        f.product_prefix_overflows,
    )?;
    // The accessor-based kernels agree on a plain primitive array
    same(
        "sum_array",
        to_val(agg::sum_array::<T, _>(p)),
        f.sum.as_ref(),
    )?;
    same(
        "min_array",
        to_val(agg::min_array::<T, _>(p)),
        f.min.as_ref(),
    )?;
    same(
        "max_array",
        to_val(agg::max_array::<T, _>(p)),
        f.max.as_ref(),
    )
}

/// `sum_array`, `sum_array_checked`, `min_array` and `max_array` on a
/// dictionary or run-end encoded array of `T`. `runs` are the physical runs
/// of a run-end encoded array, as (value, length), whose checked sum
/// multiplies each value by its length first
fn accessor<T: ArrowNumericType, A: ArrayAccessor<Item = T::Native> + Copy>(
    acc: A,
    vdt: &DataType,
    v: &[Val],
    runs: Option<Vec<(Val, usize)>>,
) -> R {
    let xs: Vec<BigInt> = v
        .iter()
        .filter(|x| **x != Val::Null)
        .map(|x| big(vdt, x))
        .collect();
    let f = int_folds(vdt, &xs);
    let to_val = |x: Option<T::Native>| x.map(|x| as_oracle::<T>(vdt, x));
    same(
        "sum_array",
        to_val(agg::sum_array::<T, A>(acc)),
        f.sum.as_ref(),
    )?;
    same(
        "min_array",
        to_val(agg::min_array::<T, A>(acc)),
        f.min.as_ref(),
    )?;
    same(
        "max_array",
        to_val(agg::max_array::<T, A>(acc)),
        f.max.as_ref(),
    )?;
    let run_order_overflows = runs.is_some_and(|runs| {
        let mut sum = BigInt::from(0);
        runs.iter().filter(|(x, _)| *x != Val::Null).any(|(x, n)| {
            let part = big(vdt, x) * BigInt::from(*n);
            sum += &part;
            !fits(vdt, &part) || !fits(vdt, &sum)
        })
    });
    let sc = agg::sum_array_checked::<T, A>(acc)
        .map(to_val)
        .map_err(|e| e.to_string());
    check_checked(
        "sum_array_checked",
        sc,
        &f.sum_exact,
        f.sum_prefix_overflows || run_order_overflows,
    )
}

/// The physical runs of a run-end encoded array, as (value, length)
fn runs<RE: RunEndIndexType>(a: &dyn Array) -> Vec<(Val, usize)> {
    let r = a.as_run::<RE>();
    let ends = r.run_ends();
    let (start, end) = (ends.offset(), ends.offset() + ends.len());
    let mut prev = start;
    let mut out = vec![];
    for (i, e) in ends.values().iter().enumerate() {
        let e = (e.as_usize()).clamp(start, end);
        if e > prev {
            out.push((val_at(r.values().as_ref(), i), e - prev));
        }
        prev = e.max(prev);
    }
    out
}

/// Aggregates against the oracle: numeric, boolean, string and binary
/// arrays, and dictionaries and run-end encoded arrays of integers
pub fn agg_prop(a: ArrayRef) -> R {
    check_in(a.as_ref())?;
    let v = vals(a.as_ref());
    let dt = a.data_type().clone();
    macro_rules! dict {
        ($k:ty, $vt:ty, $vdt:expr) => {
            accessor::<$vt, _>(
                a.as_dictionary::<$k>()
                    .downcast_dict::<PrimitiveArray<$vt>>()
                    .unwrap(),
                $vdt,
                &v,
                None,
            )
        };
    }
    macro_rules! ree {
        ($r:ty, $vt:ty, $vdt:expr) => {
            accessor::<$vt, _>(
                a.as_run::<$r>().downcast::<PrimitiveArray<$vt>>().unwrap(),
                $vdt,
                &v,
                Some(runs::<$r>(a.as_ref())),
            )
        };
    }
    use DataType::*;
    match &dt {
        Int8 => primitive::<Int8Type>(a.as_ref(), &v),
        Int16 => primitive::<Int16Type>(a.as_ref(), &v),
        Int32 => primitive::<Int32Type>(a.as_ref(), &v),
        Int64 => primitive::<Int64Type>(a.as_ref(), &v),
        UInt8 => primitive::<UInt8Type>(a.as_ref(), &v),
        UInt16 => primitive::<UInt16Type>(a.as_ref(), &v),
        UInt32 => primitive::<UInt32Type>(a.as_ref(), &v),
        UInt64 => primitive::<UInt64Type>(a.as_ref(), &v),
        Float32 => primitive::<Float32Type>(a.as_ref(), &v),
        Float64 => primitive::<Float64Type>(a.as_ref(), &v),
        Decimal32(..) => primitive::<Decimal32Type>(a.as_ref(), &v),
        Decimal64(..) => primitive::<Decimal64Type>(a.as_ref(), &v),
        Decimal128(..) => primitive::<Decimal128Type>(a.as_ref(), &v),
        Decimal256(..) => primitive::<Decimal256Type>(a.as_ref(), &v),
        Boolean => {
            let b = a.as_boolean();
            let xs: Vec<bool> = v
                .iter()
                .filter_map(|x| if let Val::Bool(x) = x { Some(*x) } else { None })
                .collect();
            let (all, any) = (
                (!xs.is_empty()).then(|| xs.iter().all(|x| *x)),
                (!xs.is_empty()).then(|| xs.iter().any(|x| *x)),
            );
            let check = |name: &str, got: Option<bool>, want: Option<bool>| {
                same(name, got.map(Val::Bool), want.map(Val::Bool).as_ref())
            };
            check("min_boolean", agg::min_boolean(b), all)?;
            check("max_boolean", agg::max_boolean(b), any)?;
            check("bool_and", agg::bool_and(b), all)?;
            check("bool_or", agg::bool_or(b), any)
        }
        Utf8 | LargeUtf8 | Utf8View | Binary | LargeBinary | BinaryView | FixedSizeBinary(_) => {
            let valid: Vec<&Val> = v.iter().filter(|x| **x != Val::Null).collect();
            let key = |x: &&Val| match x {
                Val::Str(s) => s.as_bytes().to_vec(),
                Val::Bytes(b) => b.clone(),
                x => panic!("not text or bytes: {x:?}"),
            };
            let (lo, hi) = (
                valid.iter().copied().min_by_key(key).cloned(),
                valid.iter().copied().max_by_key(key).cloned(),
            );
            let s = |x: Option<&str>| x.map(|x| Val::Str(x.to_string()));
            let b = |x: Option<&[u8]>| x.map(|x| Val::Bytes(x.to_vec()));
            let (got_lo, got_hi) = match &dt {
                Utf8 => (
                    s(agg::min_string(a.as_string::<i32>())),
                    s(agg::max_string(a.as_string::<i32>())),
                ),
                LargeUtf8 => (
                    s(agg::min_string(a.as_string::<i64>())),
                    s(agg::max_string(a.as_string::<i64>())),
                ),
                Utf8View => (
                    s(agg::min_string_view(a.as_string_view())),
                    s(agg::max_string_view(a.as_string_view())),
                ),
                Binary => (
                    b(agg::min_binary(a.as_binary::<i32>())),
                    b(agg::max_binary(a.as_binary::<i32>())),
                ),
                LargeBinary => (
                    b(agg::min_binary(a.as_binary::<i64>())),
                    b(agg::max_binary(a.as_binary::<i64>())),
                ),
                BinaryView => (
                    b(agg::min_binary_view(a.as_binary_view())),
                    b(agg::max_binary_view(a.as_binary_view())),
                ),
                _ => (
                    b(agg::min_fixed_size_binary(a.as_fixed_size_binary())),
                    b(agg::max_fixed_size_binary(a.as_fixed_size_binary())),
                ),
            };
            same("min", got_lo, lo.as_ref())?;
            same("max", got_hi, hi.as_ref())
        }
        Dictionary(k, vt) => match (k.as_ref(), vt.as_ref()) {
            (Int8, Int32) => dict!(Int8Type, Int32Type, vt),
            (Int8, Int64) => dict!(Int8Type, Int64Type, vt),
            (UInt16, Int32) => dict!(UInt16Type, Int32Type, vt),
            (UInt16, Int64) => dict!(UInt16Type, Int64Type, vt),
            (Int32, Int32) => dict!(Int32Type, Int32Type, vt),
            (Int32, Int64) => dict!(Int32Type, Int64Type, vt),
            _ => unreachable!("agg_case generates no {dt}"),
        },
        RunEndEncoded(r, vf) => match (r.data_type(), vf.data_type()) {
            (Int16, Int32) => ree!(Int16Type, Int32Type, vf.data_type()),
            (Int16, Int64) => ree!(Int16Type, Int64Type, vf.data_type()),
            (Int32, Int32) => ree!(Int32Type, Int32Type, vf.data_type()),
            (Int32, Int64) => ree!(Int32Type, Int64Type, vf.data_type()),
            (Int64, Int32) => ree!(Int64Type, Int32Type, vf.data_type()),
            (Int64, Int64) => ree!(Int64Type, Int64Type, vf.data_type()),
            _ => unreachable!("agg_case generates no {dt}"),
        },
        _ => unreachable!("agg_case generates no {dt}"),
    }
}

// ---------------------------------------------------------------- fixed point

/// Two Decimal128 arrays of one length, and a required scale
pub type FixedPointCase = (ArrayRef, ArrayRef, i8);

/// `multiply_fixed_point`, its `_dyn` form and `multiply_fixed_point_checked`
/// round the exact product to the required scale, half away from zero, as
/// documented. A required scale above the product scale is an error, and so
/// is an output type that is not a valid decimal. The unchecked forms wrap
/// to i128, and the checked one fails when the rounded product does not fit
pub fn fixed_point_prop(case: FixedPointCase) -> R {
    let scales = match (case.0.data_type(), case.1.data_type()) {
        (DataType::Decimal128(_, s1), DataType::Decimal128(_, s2)) => {
            *s1 as i32 + *s2 as i32 - case.2 as i32
        }
        _ => 0,
    };
    let r = caught(|| fixed_point_body(case));
    // Known: the divisor 10^(product scale - required scale) is computed with
    // `pow_wrapping` in i256, which wraps past 10^76, and the exponent is
    // computed in i8, which overflows past 127 (apache/arrow-rs#11605)
    match r {
        Err(TestCaseError::Fail(_)) if scales > 76 => Err(crate::known::known(
            "known: multiply_fixed_point divisor past 10^76 (apache/arrow-rs#11605)",
        )),
        r => r,
    }
}

fn fixed_point_body((l, r, req): FixedPointCase) -> R {
    use arrow_arith::arithmetic::{
        multiply_fixed_point, multiply_fixed_point_checked, multiply_fixed_point_dyn,
    };
    use std::sync::Arc;
    check_in(l.as_ref())?;
    check_in(r.as_ref())?;
    let (DataType::Decimal128(p1, s1), DataType::Decimal128(p2, s2)) =
        (l.data_type().clone(), r.data_type().clone())
    else {
        panic!("not Decimal128: {} and {}", l.data_type(), r.data_type())
    };
    let product_scale = s1 as i32 + s2 as i32;
    let precision = (p1 + p2 + 1).min(38);
    let out_type = DataType::Decimal128(precision, req);
    // validate_decimal_precision_and_scale's documented errors for the output
    let type_error = req as i32 > product_scale || req > 38 || (req > 0 && req as u8 > precision);
    let (vl, vr) = (vals(l.as_ref()), vals(r.as_ref()));
    let mut wrapping = vec![];
    let mut checked = vec![];
    let mut overflows = false;
    for i in 0..l.len() {
        if vl[i] == Val::Null || vr[i] == Val::Null {
            wrapping.push(Val::Null);
            checked.push(Val::Null);
            continue;
        }
        let p = big(l.data_type(), &vl[i]) * big(r.data_type(), &vr[i]);
        let v = if req as i32 >= product_scale {
            p
        } else {
            round_half_away(&p, product_scale - req as i32)
        };
        wrapping.push(as_val(&out_type, &wrap(&out_type, &v)));
        overflows |= !fits(&out_type, &v);
        checked.push(if fits(&out_type, &v) {
            as_val(&out_type, &v)
        } else {
            Val::Null
        });
    }
    let (la, ra) = (
        l.as_primitive::<Decimal128Type>(),
        r.as_primitive::<Decimal128Type>(),
    );
    let unchecked = [
        (
            "multiply_fixed_point",
            multiply_fixed_point(la, ra, req).map(|a| Arc::new(a) as ArrayRef),
        ),
        (
            "multiply_fixed_point_dyn",
            multiply_fixed_point_dyn(&l, &r, req),
        ),
        (
            "multiply_fixed_point_checked",
            multiply_fixed_point_checked(la, ra, req).map(|a| Arc::new(a) as ArrayRef),
        ),
    ];
    for (name, got) in unchecked {
        let (want, must_fail) = if name.ends_with("checked") {
            (&checked, type_error || overflows)
        } else {
            (&wrapping, type_error)
        };
        let ctx = |m: String| fail(format!("{name} to scale {req}: {m}"));
        match got {
            Err(_) if must_fail => {}
            Err(e) => return Err(ctx(format!("unexpected error {e}"))),
            Ok(_) if must_fail => return Err(ctx("expected an error, got a result".into())),
            Ok(out) => check_rows(&out, &out_type, want).map_err(|e| ctx(format!("{e:?}")))?,
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- formatting

/// `n` with the decimal point `scale` digits from the right, as
/// `format_decimal` documents: leading zeros as needed, zeros appended for a
/// negative scale except to zero, and every digit kept
fn decimal_string(n: &BigInt, scale: i32) -> String {
    use num_traits::{Signed, Zero};
    let sign = if n.is_negative() { "-" } else { "" };
    let digits = n.abs().to_string();
    if scale <= 0 {
        let zeros = if n.is_zero() { 0 } else { (-scale) as usize };
        return format!("{sign}{digits}{}", "0".repeat(zeros));
    }
    let s = scale as usize;
    let padded = format!("{digits:0>width$}", width = s + 1);
    let (whole, fraction) = padded.split_at(padded.len() - s);
    format!("{sign}{whole}.{fraction}")
}

/// `PrimitiveArray`'s decimal precision methods against exact integers:
/// `validate_decimal_precision` and `null_if_overflow_precision` with
/// precision `p`, and `with_precision_and_scale` with `p1` and `s1`
pub fn decimal_precision_prop((a, p, p1, s1): (ArrayRef, u8, u8, i8)) -> R {
    check_in(a.as_ref())?;
    match a.data_type() {
        DataType::Decimal32(..) => decimal_precision::<Decimal32Type>(&a, p, p1, s1),
        DataType::Decimal64(..) => decimal_precision::<Decimal64Type>(&a, p, p1, s1),
        DataType::Decimal128(..) => decimal_precision::<Decimal128Type>(&a, p, p1, s1),
        _ => decimal_precision::<Decimal256Type>(&a, p, p1, s1),
    }
}

fn decimal_precision<T: DecimalType>(a: &ArrayRef, p: u8, p1: u8, s1: i8) -> R {
    let dt = a.data_type().clone();
    let s0 = decimal_scale(&dt).unwrap();
    let v = vals(a.as_ref());
    let limit = pow10(i32::from(p));
    let fits_p = |x: &Val| *x != Val::Null && big(&dt, x).abs() < limit;
    let arr = a.as_primitive::<T>();
    // A known bug is discarded after the rest of the case is checked
    let mut known_bug = None;

    // A scale above the precision is an invalid type, as
    // `validate_decimal_precision_and_scale` documents
    let valid = s0 <= i32::from(p) && v.iter().all(|x| *x == Val::Null || fits_p(x));
    match arr.validate_decimal_precision(p) {
        // Its check of the precision against the scale casts a negative
        // scale to u8. Found by this harness on main at 9ffd63d528
        // (apache/arrow-rs#12095)
        Err(e) if s0 < 0 && e.to_string().contains("is less than scale -") => {
            known_bug = Some(
                "known: validate_decimal_precision rejects every negative scale (apache/arrow-rs#12095)",
            );
        }
        r if r.is_ok() != valid => {
            return Err(fail(format!(
                "validate_decimal_precision({p}) on {dt}: {r:?}, expected {}",
                if valid { "Ok" } else { "an error" }
            )));
        }
        _ => {}
    }

    let out = arr.null_if_overflow_precision(p);
    let kept: Vec<Val> = v
        .iter()
        .map(|x| if fits_p(x) { x.clone() } else { Val::Null })
        .collect();
    check_rows(&out, out.data_type(), &kept)
        .map_err(|e| fail(format!("null_if_overflow_precision({p}) on {dt}: {e:?}")))?;
    if out.data_type() != &dt {
        // It returns the type's default precision and scale, which changes
        // every value's meaning. It does so on every input, so discarding
        // would discard every case (apache/arrow-rs#11300)
        tolerated(
            "known: null_if_overflow_precision resets the precision and scale (apache/arrow-rs#11300)",
        )?;
    }

    // The errors `validate_decimal_precision_and_scale` documents
    let valid =
        p1 != 0 && p1 <= T::MAX_PRECISION && s1 <= T::MAX_SCALE && i32::from(s1) <= i32::from(p1);
    match (arr.clone().with_precision_and_scale(p1, s1), valid) {
        (Ok(b), true) => {
            let want = (T::TYPE_CONSTRUCTOR)(p1, s1);
            if b.data_type() != &want {
                return Err(fail(format!(
                    "with_precision_and_scale({p1}, {s1}) on {dt} returned {}",
                    b.data_type()
                )));
            }
            check_rows(&b, &want, &v).map_err(|e| {
                fail(format!(
                    "with_precision_and_scale({p1}, {s1}) on {dt}: {e:?}"
                ))
            })?;
        }
        (Err(_), false) => {}
        (r, _) => {
            return Err(fail(format!(
                "with_precision_and_scale({p1}, {s1}) on {dt}: {:?}, expected {}",
                r.map(|b| b.data_type().clone()),
                if valid { "Ok" } else { "an error" }
            )));
        }
    }
    known_bug.map_or(Ok(()), |why| Err(known(why)))
}

/// Decimals format as `format_decimal` documents, through
/// `value_as_string`, casts to every string type, and `ArrayFormatter`.
/// For Decimal256, `i256`'s `Display` agrees with exact integers and
/// `FromStr` reads it back
pub fn decimal_format_prop(a: ArrayRef) -> R {
    use arrow_buffer::i256;
    use arrow_cast::display::{ArrayFormatter, FormatOptions};
    check_in(a.as_ref())?;
    let dt = a.data_type().clone();
    let scale = decimal_scale(&dt).unwrap();
    let v = vals(a.as_ref());
    let expected: Vec<Val> = v
        .iter()
        .map(|x| {
            if *x == Val::Null {
                Val::Null
            } else {
                Val::Str(decimal_string(&big(&dt, x), scale))
            }
        })
        .collect();
    let formatter = ArrayFormatter::try_new(a.as_ref(), &FormatOptions::default())
        .map_err(|e| fail(format!("ArrayFormatter: {e}")))?;
    for i in (0..a.len()).filter(|&i| v[i] != Val::Null) {
        let Val::Str(want) = &expected[i] else {
            unreachable!()
        };
        let got = match dt {
            DataType::Decimal32(..) => a.as_primitive::<Decimal32Type>().value_as_string(i),
            DataType::Decimal64(..) => a.as_primitive::<Decimal64Type>().value_as_string(i),
            DataType::Decimal128(..) => a.as_primitive::<Decimal128Type>().value_as_string(i),
            _ => a.as_primitive::<Decimal256Type>().value_as_string(i),
        };
        if got != *want {
            return Err(fail(format!(
                "value_as_string row {i}: {got:?}, expected {want:?}"
            )));
        }
        let shown = formatter.value(i).to_string();
        if shown != *want {
            return Err(fail(format!(
                "ArrayFormatter row {i}: {shown:?}, expected {want:?}"
            )));
        }
        if let DataType::Decimal256(..) = dt {
            let x = a.as_primitive::<Decimal256Type>().value(i);
            let exact = big(&dt, &v[i]).to_string();
            if x.to_string() != exact {
                return Err(fail(format!("i256 Display row {i}: {x}, expected {exact}")));
            }
            match exact.parse::<i256>() {
                Ok(y) if y == x => {}
                r => return Err(fail(format!("i256 FromStr row {i} of {exact}: {r:?}"))),
            }
        }
    }
    for to in [DataType::Utf8, DataType::LargeUtf8, DataType::Utf8View] {
        let out = arrow_cast::cast(&a, &to).map_err(|e| fail(format!("cast to {to}: {e}")))?;
        check_rows(&out, &to, &expected).map_err(|e| fail(format!("cast to {to}: {e:?}")))?;
    }
    Ok(())
}

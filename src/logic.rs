//! Properties for kernels that pick rows or nulls by a mask: `zip`, `nullif`,
//! Kleene logic, and `is_null`

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, BooleanArray, Datum, Scalar};
use arrow_schema::DataType;
use proptest::test_runner::TestCaseError;

use crate::known::known;
use crate::props::R;
use crate::props::{can_hold_null, check_in, fail, guarded, kernel_err};
use crate::val::check_rows;
use crate::val::{Val, vals};

fn bit(m: &BooleanArray, i: usize) -> Option<bool> {
    m.is_valid(i).then(|| m.value(i))
}

/// `zip` takes `truthy` where the mask is true and `falsy` where it is false
/// or null. Either side may be a one-row scalar
pub fn zip_prop(
    (mask, t, f, t_scalar, f_scalar): (BooleanArray, ArrayRef, ArrayRef, bool, bool),
) -> R {
    check_in(&mask)?;
    check_in(t.as_ref())?;
    check_in(f.as_ref())?;
    let (t_scalar, f_scalar) = (t_scalar && !t.is_empty(), f_scalar && !f.is_empty());
    let (vt, vf) = (vals(t.as_ref()), vals(f.as_ref()));
    let expected: Vec<Val> = (0..mask.len())
        .map(|i| match bit(&mask, i) {
            Some(true) => vt[if t_scalar { 0 } else { i }].clone(),
            _ => vf[if f_scalar { 0 } else { i }].clone(),
        })
        .collect();
    let datum = |a: &ArrayRef, scalar: bool| -> Box<dyn Datum> {
        if scalar {
            Box::new(Scalar::new(a.slice(0, 1)))
        } else {
            Box::new(a.clone())
        }
    };
    let (dt, dt_s, ft_s) = (
        t.data_type().clone(),
        datum(&t, t_scalar),
        datum(&f, f_scalar),
    );
    let out = guarded(&dt, || {
        arrow_select::zip::zip(&mask, dt_s.as_ref(), ft_s.as_ref())
    })?
    .map_err(|e| {
        kernel_err(e, &dt, |z| {
            arrow_select::zip::zip(&BooleanArray::from(vec![true, false]), z, z)
        })
    })?;
    check_rows(&out, &dt, &expected)
        .map_err(|e| fail(format!("zip (scalars: {t_scalar}, {f_scalar}): {e:?}")))
}

/// `merge` takes the next `truthy` row where the mask is true and the next
/// `falsy` row where it is false or null. Either side may be a one-row
/// scalar, which repeats
pub fn merge_prop(
    (mask, t, f, t_scalar, f_scalar): (BooleanArray, ArrayRef, ArrayRef, bool, bool),
) -> R {
    use arrow_select::merge::merge;
    check_in(&mask)?;
    check_in(t.as_ref())?;
    check_in(f.as_ref())?;
    let (vt, vf) = (vals(t.as_ref()), vals(f.as_ref()));
    let (mut ti, mut fi) = (0, 0);
    let expected: Vec<Val> = (0..mask.len())
        .map(|i| {
            let (v, next, scalar) = match bit(&mask, i) {
                Some(true) => (&vt, &mut ti, t_scalar),
                _ => (&vf, &mut fi, f_scalar),
            };
            if scalar {
                return v[0].clone();
            }
            *next += 1;
            v[*next - 1].clone()
        })
        .collect();
    let datum = |a: &ArrayRef, scalar: bool| -> Box<dyn Datum> {
        if scalar {
            Box::new(Scalar::new(a.clone()))
        } else {
            Box::new(a.clone())
        }
    };
    let (dt, t_d, f_d) = (t.data_type(), datum(&t, t_scalar), datum(&f, f_scalar));
    let out = guarded(dt, || merge(&mask, t_d.as_ref(), f_d.as_ref()))?.map_err(|e| {
        kernel_err(e, dt, |z| {
            merge(
                &BooleanArray::from(vec![true, false]),
                &z.slice(0, 1),
                &z.slice(1, 1),
            )
        })
    })?;
    check_rows(&out, dt, &expected)
        .map_err(|e| fail(format!("merge (scalars: {t_scalar}, {f_scalar}): {e:?}")))
}

/// `merge_n` takes the next row of array `i` for each index `Some(i)`, and
/// a null for each `None`. Rows past those the indices use are ignored
pub fn merge_n_prop((arrays, indices): (Vec<ArrayRef>, Vec<Option<usize>>)) -> R {
    use arrow_select::merge::merge_n;
    arrays.iter().try_for_each(|a| check_in(a.as_ref()))?;
    let refs: Vec<&dyn Array> = arrays.iter().map(|a| a.as_ref()).collect();
    let Some(first) = arrays.first() else {
        return match merge_n(&refs, &indices) {
            Err(_) => Ok(()),
            Ok(_) => Err(fail("merge_n of no arrays returned a result")),
        };
    };
    let dt = first.data_type();
    let v: Vec<Vec<Val>> = arrays.iter().map(|a| vals(a.as_ref())).collect();
    let mut next = vec![0; arrays.len()];
    let expected: Vec<Val> = indices
        .iter()
        .map(|ix| match ix {
            Some(i) => {
                next[*i] += 1;
                v[*i][next[*i] - 1].clone()
            }
            None => Val::Null,
        })
        .collect();
    let r = guarded(dt, || merge_n(&refs, &indices))?;
    // A null this type cannot hold is an error, as apache/arrow-rs#10992
    // settled for every kernel that introduces nulls
    if !can_hold_null(dt) && indices.contains(&None) {
        let r = match r {
            Err(_) => Ok(()),
            Ok(out) => Err(fail(format!(
                "merge_n put {} nulls in {dt}, which cannot hold one",
                out.logical_null_count()
            ))),
        };
        // Known: `MutableArrayData` extends a run-end encoded array with a
        // null value whatever its values field allows. Unions return an
        // error. Found by this harness on main at 01a991765
        // (apache/arrow-rs#11835)
        if let Err(TestCaseError::Fail(m)) = &r
            && m.to_string().contains("which cannot hold one")
            && matches!(dt, DataType::RunEndEncoded(..))
        {
            return Err(known(
                "known: merge_n puts nulls in fields that cannot hold one (apache/arrow-rs#11835)",
            ));
        }
        return r;
    }
    // Known: `MutableArrayData` extends every child of a struct or
    // fixed-size list with nulls, which fails on a union child with no
    // nullable field, although the parent's validity can hold the null.
    // Found by this harness on main at 01a991765 (apache/arrow-rs#11836)
    if let Err(e) = &r
        && e.to_string()
            .contains("Cannot extend a union with no nullable fields")
    {
        return Err(known(
            "known: extending a struct or fixed-size list with nulls fails on a union child (apache/arrow-rs#11836)",
        ));
    }
    let probe = |z: &ArrayRef| merge_n(&[z.as_ref()], &[Some(0), None]);
    let out = r.map_err(|e| kernel_err(e, dt, probe))?;
    check_rows(&out, dt, &expected).map_err(|e| fail(format!("merge_n: {e:?}")))?;
    // `usize` indices, which cannot be null, take the same path
    if let Some(plain) = indices.iter().copied().collect::<Option<Vec<usize>>>() {
        let out = guarded(dt, || merge_n(&refs, &plain))?
            .map_err(|e| fail(format!("merge_n with usize indices: {e}")))?;
        check_rows(&out, dt, &expected)
            .map_err(|e| fail(format!("merge_n with usize indices: {e:?}")))?;
    }
    Ok(())
}

/// `nullif` sets rows null where the mask is true, and keeps the others
pub fn nullif_prop((a, mask): (ArrayRef, BooleanArray)) -> R {
    check_in(a.as_ref())?;
    check_in(&mask)?;
    let v = vals(a.as_ref());
    let expected: Vec<Val> = (0..a.len())
        .map(|i| {
            if bit(&mask, i) == Some(true) {
                Val::Null
            } else {
                v[i].clone()
            }
        })
        .collect();
    let dt = a.data_type().clone();
    let r = arrow_select::nullif::nullif(a.as_ref(), &mask);
    // A null this type cannot hold is an error, as apache/arrow-rs#10992
    // settled for every kernel that introduces nulls
    if !can_hold_null(&dt) && expected.contains(&Val::Null) {
        return match r {
            Err(_) => Ok(()),
            Ok(out) => Err(fail(format!(
                "nullif put {} nulls in {dt}, which cannot hold one",
                out.logical_null_count()
            ))),
        };
    }
    let out = r.map_err(|e| {
        kernel_err(e, &dt, |z| {
            arrow_select::nullif::nullif(z.as_ref(), &BooleanArray::from(vec![false; z.len()]))
        })
    })?;
    check_rows(&out, &dt, &expected).map_err(|e| fail(format!("nullif: {e:?}")))
}

/// Kleene `and` and `or` against their truth tables, and De Morgan's law
pub fn kleene((a, b): (BooleanArray, BooleanArray)) -> R {
    use arrow_arith::boolean::{and_kleene, not, or_kleene};
    check_in(&a)?;
    check_in(&b)?;
    let table = |f: fn(Option<bool>, Option<bool>) -> Option<bool>| -> Vec<Val> {
        (0..a.len())
            .map(|i| f(bit(&a, i), bit(&b, i)).map_or(Val::Null, Val::Bool))
            .collect()
    };
    let and_t = |x: Option<bool>, y: Option<bool>| match (x, y) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
    };
    let or_t = |x: Option<bool>, y: Option<bool>| match (x, y) {
        (Some(true), _) | (_, Some(true)) => Some(true),
        (Some(false), Some(false)) => Some(false),
        _ => None,
    };
    let and_out: ArrayRef =
        Arc::new(and_kleene(&a, &b).map_err(|e| fail(format!("and_kleene: {e}")))?);
    check_rows(&and_out, &DataType::Boolean, &table(and_t))
        .map_err(|e| fail(format!("and_kleene: {e:?}")))?;
    let or_out: ArrayRef =
        Arc::new(or_kleene(&a, &b).map_err(|e| fail(format!("or_kleene: {e}")))?);
    check_rows(&or_out, &DataType::Boolean, &table(or_t))
        .map_err(|e| fail(format!("or_kleene: {e:?}")))?;
    // not(a and b) == not(a) or not(b)
    let lhs = not(and_out.as_any().downcast_ref::<BooleanArray>().unwrap())
        .map_err(|e| fail(format!("not: {e}")))?;
    let rhs = or_kleene(&not(&a).unwrap(), &not(&b).unwrap())
        .map_err(|e| fail(format!("or_kleene: {e}")))?;
    if vals(&lhs) != vals(&rhs) {
        return Err(fail("not(and_kleene(a, b)) != or_kleene(not(a), not(b))"));
    }
    Ok(())
}

/// `is_null` and `is_not_null` report logical nulls, the rows the oracle
/// reads as null, and have no nulls themselves
pub fn is_null_prop(a: ArrayRef) -> R {
    use arrow_arith::boolean::{is_not_null, is_null};
    check_in(a.as_ref())?;
    let v = vals(a.as_ref());
    let want: Vec<Val> = v.iter().map(|x| Val::Bool(*x == Val::Null)).collect();
    let got: ArrayRef = Arc::new(is_null(a.as_ref()).map_err(|e| fail(format!("is_null: {e}")))?);
    check_rows(&got, &DataType::Boolean, &want).map_err(|e| fail(format!("is_null: {e:?}")))?;
    let not_want: Vec<Val> = v.iter().map(|x| Val::Bool(*x != Val::Null)).collect();
    let got: ArrayRef =
        Arc::new(is_not_null(a.as_ref()).map_err(|e| fail(format!("is_not_null: {e}")))?);
    check_rows(&got, &DataType::Boolean, &not_want)
        .map_err(|e| fail(format!("is_not_null: {e:?}")))?;
    let nulls = v.iter().filter(|x| **x == Val::Null).count();
    if a.logical_null_count() != nulls {
        return Err(fail(format!(
            "logical_null_count {} != {nulls}",
            a.logical_null_count()
        )));
    }
    Ok(())
}

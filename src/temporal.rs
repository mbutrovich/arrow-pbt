//! Properties for temporal arithmetic: dates, timestamps, durations and
//! intervals with `add` and `sub`, against a calendar written from the
//! proleptic Gregorian rules rather than chrono

use arrow_array::{ArrayRef, Datum, Scalar};
use arrow_schema::{DataType, IntervalUnit, TimeUnit};

use crate::known::known;
use crate::props::Operands;
use crate::props::R;
use crate::props::{check_in, fail, interval_fields};
use crate::val::check_rows;
use crate::val::{Val, vals};
use arrow_schema::ArrowError;

/// Two arrays, the operation (0 add, 1 sub, 2 add_wrapping, 3 sub_wrapping),
/// and whether each side is a one-row scalar
pub type TemporalCase = (ArrayRef, ArrayRef, u8, bool, bool);

const DAY_NS: i128 = 86_400_000_000_000;

/// Days since 1970-01-01 of a proleptic Gregorian date (H. Hinnant's
/// `days_from_civil`)
pub fn days_from_civil(y: i128, m: i128, d: i128) -> i128 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The date of a day number (H. Hinnant's `civil_from_days`)
fn civil_from_days(z: i128) -> (i128, i128, i128) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i128::from(m <= 2), m, d)
}

fn days_in_month(y: i128, m: i128) -> i128 {
    let (ny, nm) = if m == 12 { (y + 1, 1) } else { (y, m + 1) };
    days_from_civil(ny, nm, 1) - days_from_civil(y, m, 1)
}

/// `day` plus `months`, with the day of month clamped to the last day of the
/// target month (2020-01-31 plus one month is 2020-02-29)
fn add_months(day: i128, months: i128) -> i128 {
    let (y, m, d) = civil_from_days(day);
    let total = y * 12 + (m - 1) + months;
    let (ny, nm) = (total.div_euclid(12), total.rem_euclid(12) + 1);
    days_from_civil(ny, nm, d.min(days_in_month(ny, nm)))
}

/// Dates this far from 1970 are past what chrono represents, so the result
/// there is left open: an error or the right value both pass
fn in_calendar_range(day: i128) -> bool {
    civil_from_days(day).0.abs() < 262_000
}

fn unit_ns(u: &TimeUnit) -> i128 {
    match u {
        TimeUnit::Second => 1_000_000_000,
        TimeUnit::Millisecond => 1_000_000,
        TimeUnit::Microsecond => 1_000,
        TimeUnit::Nanosecond => 1,
    }
}

/// The offset of a fixed time zone such as "+05:30" or "UTC", in ns
fn offset_ns(tz: Option<&str>) -> i128 {
    match tz {
        None | Some("UTC" | "+00:00") => 0,
        Some(s) => {
            let sign = if s.starts_with('-') { -1 } else { 1 };
            let (h, m) = s[1..].split_once(':').unwrap();
            sign * (h.parse::<i128>().unwrap() * 3600 + m.parse::<i128>().unwrap() * 60)
                * 1_000_000_000
        }
    }
}

fn int(v: &Val) -> i128 {
    match v {
        Val::I32(x) => *x as i128,
        Val::I64(x) => *x as i128,
        Val::Raw(b) => {
            let mut a = [if b[b.len() - 1] & 0x80 != 0 { 0xFF } else { 0 }; 16];
            a[..b.len()].copy_from_slice(b);
            i128::from_le_bytes(a)
        }
        v => panic!("not an integer: {v:?}"),
    }
}

fn raw(v: i128, w: usize) -> Val {
    Val::Raw(v.to_le_bytes()[..w].to_vec())
}

/// An interval's (months, days, ns) and whether it has those parts
fn interval_parts(unit: &IntervalUnit, v: &Val) -> (i128, i128, i128) {
    let Val::Raw(b) = v else { unreachable!() };
    let f = interval_fields(b);
    match unit {
        IntervalUnit::YearMonth => (f[0] as i128, 0, 0),
        IntervalUnit::DayTime => (0, f[0] as i128, f[1] as i128 * 1_000_000),
        IntervalUnit::MonthDayNano => (f[0] as i128, f[1] as i128, f[2] as i128),
    }
}

/// One row's expected result: a value, an overflow that must be an error,
/// or an open case
enum Row {
    Value(Val),
    Overflow,
    Open,
}

fn fits(v: i128, bits: u32) -> bool {
    let lim = 1i128 << (bits - 1);
    -lim <= v && v < lim
}

/// The result of `l op r` for one row, and its type
fn row(lt: &DataType, rt: &DataType, sub: bool, l: &Val, r: &Val) -> Option<(Row, DataType)> {
    use DataType::*;
    let sign = if sub { -1 } else { 1 };
    let checked64 = |v: i128| {
        if fits(v, 64) {
            Row::Value(raw(v, 8))
        } else {
            Row::Overflow
        }
    };
    let out = match (lt, rt) {
        (Timestamp(u, _), Duration(v)) if u == v => (checked64(int(l) + sign * int(r)), lt.clone()),
        (Timestamp(u, _), Timestamp(v, _)) if u == v && sub => {
            (checked64(int(l) - int(r)), Duration(*u))
        }
        (Duration(u), Duration(v)) if u == v => (checked64(int(l) + sign * int(r)), lt.clone()),
        (Date32, Date32) if sub => (
            Row::Value(raw((int(l) - int(r)) * 86_400, 8)),
            Duration(TimeUnit::Second),
        ),
        (Date64, Date64) if sub => (checked64(int(l) - int(r)), Duration(TimeUnit::Millisecond)),
        (Interval(u), Interval(v)) if u == v => {
            let (Val::Raw(a), Val::Raw(b)) = (l, r) else {
                unreachable!()
            };
            let (fa, fb) = (interval_fields(a), interval_fields(b));
            let mut bytes = vec![];
            let mut over = false;
            for k in 0..fa.len() {
                let x = fa[k] as i128 + sign * fb[k] as i128;
                let w = if k == 2 { 8 } else { 4 };
                over |= !fits(x, w as u32 * 8);
                bytes.extend(&x.to_le_bytes()[..w]);
            }
            (
                if over {
                    Row::Overflow
                } else {
                    Row::Value(Val::Raw(bytes))
                },
                lt.clone(),
            )
        }
        (Timestamp(u, tz), Interval(iu)) => {
            let (months, days, ns) = interval_parts(iu, r);
            let local = int(l) * unit_ns(u) + offset_ns(tz.as_deref());
            let (day, tod) = (local.div_euclid(DAY_NS), local.rem_euclid(DAY_NS));
            let after_months = add_months(day, sign * months);
            let moved = after_months + sign * days;
            let instant = moved * DAY_NS + tod - offset_ns(tz.as_deref()) + sign * ns;
            // Each step must stay in range, not only the result
            let open = [day, after_months, moved, instant.div_euclid(DAY_NS)]
                .iter()
                .any(|d| !in_calendar_range(*d));
            let v = instant.div_euclid(unit_ns(u));
            (if open { Row::Open } else { checked64(v) }, lt.clone())
        }
        (Date32 | Date64, Interval(iu)) => {
            let (months, days, ns) = interval_parts(iu, r);
            let day = if *lt == Date32 {
                int(l)
            } else {
                int(l).div_euclid(86_400_000)
            };
            // The part of a day in the interval is dropped toward zero, as
            // chrono's `NaiveDate + Duration` does
            let after_months = add_months(day, sign * months);
            let moved = after_months + sign * days + sign * (ns / DAY_NS);
            let open = [day, after_months, after_months + sign * days, moved]
                .iter()
                .any(|d| !in_calendar_range(*d));
            let row = match lt {
                _ if open => Row::Open,
                Date32 if fits(moved, 32) => Row::Value(raw(moved, 4)),
                Date32 => Row::Overflow,
                _ => checked64(moved * 86_400_000),
            };
            (row, lt.clone())
        }
        _ => return None,
    };
    Some(out)
}

/// An interval array, an `Int64` or `Float64` factor array of the same
/// length, and which is a one-row scalar: `ScalarLeft` is the interval
pub type IntervalScaleCase = (ArrayRef, ArrayRef, Operands);

const TWO_63: f64 = -(i64::MIN as f64);

/// `x` rounded ties-to-even at nanosecond precision, where PostgreSQL's
/// `TSROUND` rounds at its microsecond precision
fn round_ns(x: f64) -> f64 {
    (x * 1e9).round_ties_even() / 1e9
}

/// `(months, days, ns) * f` as DuckDB's `INTERVAL * DOUBLE` computes it, at
/// nanosecond precision, or `None` when a field is out of range. DuckDB
/// takes it from PostgreSQL's `interval_mul`, and apache/arrow-rs#10409
/// adapts DuckDB's:
/// <https://github.com/duckdb/duckdb/blob/21aca0424f1faf78b593b1e6fbfdd4846624c987/src/function/scalar/operator/multiply.cpp#L48-L123>
/// and
/// <https://github.com/postgres/postgres/blob/78758d37306cd89ab060f00cb06f249018d5b8da/src/backend/utils/adt/timestamp.c#L3627-L3744>.
/// Whole months and days are truncated, and their fractions cascade down at
/// 30 days a month and 86,400 seconds a day, never up, in DuckDB's order of
/// operations. A product above `i32::MAX` is out of range,
/// as in DuckDB. apache/arrow-rs#10409 settles the rest: an integral factor
/// multiplies exactly, and rounding is ties-to-even at nanosecond precision.
/// With `fracts`, the fraction of the day product is added to the fraction
/// of the month remainder instead, as arrow-rs adds them
fn interval_mul_f64(
    months: i32,
    days: i32,
    ns: i64,
    f: f64,
    fracts: bool,
) -> Option<(i32, i32, i64)> {
    if !f.is_finite() {
        return None;
    }
    if f.fract() == 0.0 && (-TWO_63..TWO_63).contains(&f) {
        let r = f as i64;
        let int = |v: i32| i32::try_from(i64::from(v).checked_mul(r)?).ok();
        return Some((int(months)?, int(days)?, ns.checked_mul(r)?));
    }
    let m = f64::from(months) * f;
    let d = f64::from(days) * f;
    let int32 = f64::from(i32::MIN)..=f64::from(i32::MAX);
    if !int32.contains(&m) || !int32.contains(&d) {
        return None;
    }
    let (rm, mut rd) = (m.trunc() as i32, d.trunc() as i32);
    let month_days = round_ns((f64::from(months) * f - f64::from(rm)) * 30.0);
    let day_fraction = match fracts {
        true => d.fract() + month_days.fract(),
        false => f64::from(days) * f - f64::from(rd) + month_days - month_days.trunc(),
    };
    let mut secs = round_ns(day_fraction * 86_400.0);
    if secs.abs() >= 86_400.0 {
        let carry = (secs / 86_400.0).trunc();
        rd = rd.checked_add(carry as i32)?;
        secs -= carry * 86_400.0;
    }
    rd = rd.checked_add(month_days.trunc() as i32)?;
    let t = (ns as f64 * f + secs * 1e9).round_ties_even();
    if !(-TWO_63..TWO_63).contains(&t) {
        return None;
    }
    Some((rm, rd, t as i64))
}

/// Known: arrow-rs adds the fraction of the day product to the fraction of
/// the month remainder, where DuckDB adds the month remainder and then takes
/// its whole days away. Floating-point addition is not associative, so a
/// row can differ from DuckDB's in the last place of its nanoseconds. The
/// signature is a row that equals `interval_mul_f64` with `fracts`. Found
/// by this harness on main at 9ffd63d528 (apache/arrow-rs#12004)
const FRACTS: &str = "known: interval mul by Float64 sums the day fractions in another order than DuckDB (apache/arrow-rs#12004)";

/// `want`, with the rows of `out` that equal `fracts` in their place, and
/// whether there were any
fn with_fracts(out: &ArrayRef, want: Vec<Val>, fracts: &[Option<Val>]) -> (Vec<Val>, bool) {
    let got = vals(out.as_ref());
    let mut any = false;
    let want = want
        .into_iter()
        .zip(got)
        .zip(fracts)
        .map(|((w, g), a)| match a {
            Some(a) if g == *a && g != w => {
                any = true;
                g
            }
            _ => w,
        })
        .collect();
    (want, any)
}

/// `mul` of an interval by an `Int64`, either way round, multiplies each
/// field exactly, and an overflow is an error. `mul` of a `MonthDayNano`
/// interval by a `Float64`, either way round, follows `interval_mul_f64`
/// without `fracts`.
/// `div` by a `Float64` multiplies by its reciprocal, as apache/arrow-rs#10409
/// settles, and division by zero is an error. A row is null when either
/// input row is, and a row that fails makes the whole call an error
pub fn interval_scale((iv, factor, operands): IntervalScaleCase) -> R {
    use arrow_arith::numeric::{div, mul};
    check_in(iv.as_ref())?;
    check_in(factor.as_ref())?;
    let dt = iv.data_type().clone();
    let DataType::Interval(unit) = dt.clone() else {
        return Err(fail(format!("interval_scale: not an interval: {dt}")));
    };
    let float = *factor.data_type() == DataType::Float64;
    let (vi, vf) = (vals(iv.as_ref()), vals(factor.as_ref()));
    let left = operands == Operands::ScalarLeft && !iv.is_empty();
    let right = operands == Operands::ScalarRight && !factor.is_empty();
    let (n, m) = (iv.len(), factor.len());
    let at_i = |i: usize| if left { &vi[0] } else { &vi[i] };
    let at_f = |i: usize| if right { &vf[0] } else { &vf[i] };
    let widths: &[u32] = match unit {
        IntervalUnit::YearMonth => &[32],
        IntervalUnit::DayTime => &[32, 32],
        IntervalUnit::MonthDayNano => &[32, 32, 64],
    };
    let encode = |fields: &[i128]| {
        Val::Raw(
            fields
                .iter()
                .zip(widths)
                .flat_map(|(v, w)| v.to_le_bytes()[..*w as usize / 8].to_vec())
                .collect(),
        )
    };
    // A row's result, and for a `Float64` factor the row in arrow-rs's order
    // of the day fractions
    let expect = |i: usize, divide: bool| -> (Row, Option<Val>) {
        let (x, k) = (at_i(i), at_f(i));
        if *x == Val::Null || *k == Val::Null {
            return (Row::Value(Val::Null), None);
        }
        let Val::Raw(b) = x else { unreachable!() };
        let f = interval_fields(b);
        if float {
            let k = k.as_f64();
            let r = |fracts| match divide {
                true if k == 0.0 => None,
                true => interval_mul_f64(f[0] as i32, f[1] as i32, f[2], 1.0 / k, fracts),
                false => interval_mul_f64(f[0] as i32, f[1] as i32, f[2], k, fracts),
            };
            let v = |(m, d, t): (i32, i32, i64)| encode(&[m as i128, d as i128, t as i128]);
            return (
                r(false).map_or(Row::Overflow, |x| Row::Value(v(x))),
                r(true).map(v),
            );
        }
        let r = int(k);
        let p: Vec<i128> = f.iter().map(|&v| v as i128 * r).collect();
        match p.iter().zip(widths).all(|(p, w)| fits(*p, *w)) {
            true => (Row::Value(encode(&p)), None),
            false => (Row::Overflow, None),
        }
    };
    type Op = fn(&dyn Datum, &dyn Datum) -> Result<ArrayRef, ArrowError>;
    // `op` on rows `i..i + len`, or on a scalar's row, with the factor
    // first when `swap`
    let call = |op: Op, swap: bool, i: usize, len: usize| {
        let x: Box<dyn Datum> = match left {
            true => Box::new(Scalar::new(iv.slice(0, 1))),
            false => Box::new(iv.slice(i, len)),
        };
        let k: Box<dyn Datum> = match right {
            true => Box::new(Scalar::new(factor.slice(0, 1))),
            false => Box::new(factor.slice(i, len)),
        };
        match swap {
            true => op(k.as_ref(), x.as_ref()),
            false => op(x.as_ref(), k.as_ref()),
        }
    };
    let mut forms: Vec<(&str, Op, bool, bool)> = vec![
        ("mul", mul, false, false),
        ("mul with the factor first", mul, true, false),
    ];
    if float {
        forms.push(("div", div, false, true));
    }
    let len = if left { m } else { n };
    // Whether a row matched the known bug. The case is discarded only after
    // every other row and form is checked
    let mut fracts_seen = false;
    for (name, op, swap, divide) in forms {
        let ctx = |m: String| fail(format!("{name} of {dt} and {}: {m}", factor.data_type()));
        let (rows, fracts): (Vec<Row>, Vec<Option<Val>>) =
            (0..len).map(|i| expect(i, divide)).unzip();
        let must_fail = rows.iter().any(|r| matches!(r, Row::Overflow));
        match call(op, swap, 0, len) {
            Ok(_) if must_fail => return Err(ctx("expected an error, got a result".into())),
            Err(e) if !must_fail => return Err(ctx(format!("unexpected error {e}"))),
            Ok(out) => {
                let want: Vec<Val> = rows
                    .into_iter()
                    .map(|r| match r {
                        Row::Value(v) => v,
                        _ => unreachable!(),
                    })
                    .collect();
                let (want, off) = with_fracts(&out, want, &fracts);
                check_rows(&out, &dt, &want).map_err(|e| ctx(format!("{e:?}")))?;
                fracts_seen |= off;
            }
            // One failing row fails the call, so every row is checked alone
            Err(_) => {
                for (i, row) in rows.into_iter().enumerate() {
                    let at = |m: String| ctx(format!("row {i} alone: {m}"));
                    match (call(op, swap, i, 1), row) {
                        (Err(_), Row::Overflow) => {}
                        (Ok(_), Row::Overflow) => return Err(at("expected an error".into())),
                        (Err(e), _) => return Err(at(format!("unexpected error {e}"))),
                        (Ok(out), Row::Value(v)) => {
                            let (want, off) = with_fracts(&out, vec![v], &fracts[i..i + 1]);
                            check_rows(&out, &dt, &want).map_err(|e| at(format!("{e:?}")))?;
                            fracts_seen |= off;
                        }
                        (Ok(_), Row::Open) => {}
                    }
                }
            }
        }
    }
    match fracts_seen {
        true => Err(known(FRACTS)),
        false => Ok(()),
    }
}

/// `add`, `sub` and their wrapping forms on temporal types against the
/// calendar above. A row that overflows makes the whole call an error
pub fn temporal_arith((l, r, op, l_scalar, r_scalar): TemporalCase) -> R {
    use arrow_arith::numeric::{add, add_wrapping, sub, sub_wrapping};
    check_in(l.as_ref())?;
    check_in(r.as_ref())?;
    let (l_scalar, r_scalar) = (l_scalar && !l.is_empty(), r_scalar && !r.is_empty());
    let n = if l_scalar && !r_scalar {
        r.len()
    } else if l_scalar {
        1
    } else {
        l.len()
    };
    let (lv, rv) = (vals(l.as_ref()), vals(r.as_ref()));
    let sub_op = op % 2 == 1;
    // Interval or duration on the left of a date or timestamp: `add` swaps
    // the sides
    let swap = !sub_op
        && matches!(l.data_type(), DataType::Interval(_) | DataType::Duration(_))
        && matches!(
            r.data_type(),
            DataType::Date32 | DataType::Date64 | DataType::Timestamp(..)
        );
    let (ft, st) = if swap {
        (r.data_type(), l.data_type())
    } else {
        (l.data_type(), r.data_type())
    };
    let mut out_type = None;
    let mut rows = vec![];
    for i in 0..n {
        let (a, b) = (
            &lv[if l_scalar { 0 } else { i }],
            &rv[if r_scalar { 0 } else { i }],
        );
        let (a, b) = if swap { (b, a) } else { (a, b) };
        let null = *a == Val::Null || *b == Val::Null;
        // A null row still gives the output type, from zero stand-ins
        let zero = |t: &DataType| Val::Raw(vec![0; t.primitive_width().unwrap()]);
        let (a, b) = if null {
            (zero(ft), zero(st))
        } else {
            (a.clone(), b.clone())
        };
        let Some((rw, t)) = row(ft, st, sub_op, &a, &b) else {
            return Err(fail(format!("no oracle for {ft} and {st}")));
        };
        out_type = Some(t);
        rows.push(if null { Row::Value(Val::Null) } else { rw });
    }
    let f = [add, sub, add_wrapping, sub_wrapping][op as usize];
    let datum = |a: &ArrayRef, s: bool| -> Box<dyn Datum> {
        if s {
            Box::new(Scalar::new(a.slice(0, 1)))
        } else {
            Box::new(a.clone())
        }
    };
    let (ld, rd) = (datum(&l, l_scalar), datum(&r, r_scalar));
    let name = ["add", "sub", "add_wrapping", "sub_wrapping"][op as usize];
    let ctx = |m: String| fail(format!("{} {name} {}: {m}", l.data_type(), r.data_type()));
    let result = f(ld.as_ref(), rd.as_ref());
    let t = out_type.clone();
    let check = |result: Result<ArrayRef, ArrowError>, rows: &[&Row], at: &str| -> R {
        let overflow = rows.iter().any(|r| matches!(r, Row::Overflow));
        let open = rows.iter().any(|r| matches!(r, Row::Open));
        match result {
            Err(_) if overflow || open => Ok(()),
            Err(e) => Err(ctx(format!("{at}unexpected error {e}"))),
            Ok(_) if overflow => Err(ctx(format!("{at}no error for a row that overflows"))),
            Ok(got) => {
                let t = t.clone().unwrap_or_else(|| got.data_type().clone());
                // Open rows take whatever the kernel returned
                let g = vals(got.as_ref());
                let expected: Vec<Val> = rows
                    .iter()
                    .enumerate()
                    .map(|(i, r)| match r {
                        Row::Value(v) => v.clone(),
                        _ => g.get(i).cloned().unwrap_or(Val::Null),
                    })
                    .collect();
                check_rows(&got, &t, &expected).map_err(|e| ctx(format!("{at}{e:?}")))
            }
        }
    };
    check(result, &rows.iter().collect::<Vec<_>>(), "")?;
    // One overflowing row makes the whole call an error, so each row is
    // also checked on its own, through one-row slices of the inputs
    for (i, row) in rows.iter().enumerate().take(64) {
        let one = |a: &ArrayRef, s: bool| -> Box<dyn Datum> {
            if s {
                Box::new(Scalar::new(a.slice(0, 1)))
            } else {
                Box::new(a.slice(i, 1))
            }
        };
        let (ld, rd) = (one(&l, l_scalar), one(&r, r_scalar));
        if l_scalar && r_scalar {
            break;
        }
        check(
            f(ld.as_ref(), rd.as_ref()),
            &[row],
            &format!("row {i} alone: "),
        )?;
    }
    Ok(())
}

/// An array, the type to cast it to, and whether the cast is `safe`
pub type TemporalCastCase = (ArrayRef, DataType, bool);

/// One row of a temporal cast: the value, or an overflow (null when safe,
/// an error otherwise), or open
fn cast_row(from: &DataType, to: &DataType, v: &Val) -> Row {
    use DataType::*;
    let x = int(v);
    let fit = |y: i128, bits: u32, as_raw: bool| -> Row {
        if !fits(y, bits) {
            Row::Overflow
        } else if as_raw {
            Row::Value(raw(y, bits as usize / 8))
        } else if bits == 32 {
            Row::Value(Val::I32(y as i32))
        } else {
            Row::Value(Val::I64(y as i64))
        }
    };
    let time_ns = |t: &DataType| match t {
        Time32(u) | Time64(u) => unit_ns(u),
        _ => unreachable!(),
    };
    match (from, to) {
        // Temporal types and their backing integers: the same value
        (Int32, Date32 | Time32(_)) | (Int64, Date64 | Time64(_) | Timestamp(..)) => {
            fit(x, from.primitive_width().unwrap() as u32 * 8, true)
        }
        (Date32 | Time32(_), Int32) | (Date64 | Time64(_) | Timestamp(..), Int64) => {
            fit(x, to.primitive_width().unwrap() as u32 * 8, false)
        }
        (Timestamp(fu, ftz), Timestamp(tu, ttz)) => {
            // Finer units multiply, coarser ones floor, as the cast to a
            // date does since apache/arrow-rs#4211
            let ns = x * unit_ns(fu);
            let scaled = if unit_ns(tu) <= unit_ns(fu) {
                ns / unit_ns(tu)
            } else {
                ns.div_euclid(unit_ns(tu))
            };
            if !fits(scaled, 64) {
                return Row::Overflow;
            }
            match (ftz, ttz) {
                // A time without a zone is read as local time in the new zone
                (None, Some(_)) => {
                    let utc = scaled * unit_ns(tu) - offset_ns(ttz.as_deref());
                    if !in_calendar_range(utc.div_euclid(DAY_NS))
                        || !in_calendar_range(scaled * unit_ns(tu) / DAY_NS)
                    {
                        return Row::Open;
                    }
                    fit(utc.div_euclid(unit_ns(tu)), 64, true)
                }
                _ => fit(scaled, 64, true),
            }
        }
        (Timestamp(fu, tz), Date32 | Date64 | Time32(_) | Time64(_)) => {
            let local = x * unit_ns(fu) + offset_ns(tz.as_deref());
            let day = local.div_euclid(DAY_NS);
            if tz.is_some() && !in_calendar_range(day) {
                return if civil_from_days(day).0.abs() > 262_200 {
                    Row::Overflow
                } else {
                    Row::Open
                };
            }
            match to {
                Date32 => fit(day, 32, true),
                Date64 => fit(day * 86_400_000, 64, true),
                t => fit(
                    local.rem_euclid(DAY_NS) / time_ns(t),
                    t.primitive_width().unwrap() as u32 * 8,
                    true,
                ),
            }
        }
        (Date32, Date64) => fit(x * 86_400_000, 64, true),
        (Date64, Date32) => fit(x.div_euclid(86_400_000), 32, true),
        (Date32 | Date64, Timestamp(tu, None)) => {
            let day = if *from == Date32 {
                x
            } else {
                x.div_euclid(86_400_000)
            };
            fit(day * DAY_NS / unit_ns(tu), 64, true)
        }
        (Time32(_) | Time64(_), Time32(_) | Time64(_)) => {
            let ns = x * time_ns(from);
            fit(
                ns.div_euclid(time_ns(to)),
                to.primitive_width().unwrap() as u32 * 8,
                true,
            )
        }
        _ => panic!("no cast oracle for {from} to {to}"),
    }
}

/// What arrow-rs does where it differs from `cast_row`: coarser timestamp
/// units truncate toward zero (apache/arrow-rs#11602)
fn cast_row_known(from: &DataType, to: &DataType, v: &Val) -> Option<Row> {
    use DataType::*;
    let x = int(v);
    let checked = |y: i128| {
        Some(if fits(y, 64) {
            Row::Value(raw(y, 8))
        } else {
            Row::Overflow
        })
    };
    match (from, to) {
        (Timestamp(fu, ftz), Timestamp(tu, ttz)) if unit_ns(tu) > unit_ns(fu) => {
            let scaled = x * unit_ns(fu) / unit_ns(tu);
            match (ftz, ttz) {
                (None, Some(_)) if fits(scaled, 64) => {
                    let utc = scaled * unit_ns(tu) - offset_ns(ttz.as_deref());
                    if !in_calendar_range(utc.div_euclid(DAY_NS))
                        || !in_calendar_range(scaled * unit_ns(tu) / DAY_NS)
                    {
                        return Some(Row::Open);
                    }
                    checked(utc.div_euclid(unit_ns(tu)))
                }
                _ => checked(scaled),
            }
        }
        _ => None,
    }
}

/// Checks a cast's result against expected rows. `spec` runs the spec
/// checker on the output, which casts that keep out-of-range values skip
fn check_cast(
    result: &Result<ArrayRef, String>,
    rows: &[Row],
    to: &DataType,
    safe: bool,
    spec: bool,
) -> Result<(), String> {
    let overflow = rows.iter().any(|r| matches!(r, Row::Overflow));
    let open = rows.iter().any(|r| matches!(r, Row::Open));
    match result {
        Err(_) if (!safe && overflow) || open => Ok(()),
        Err(e) => Err(format!("unexpected error {e}")),
        Ok(_) if !safe && overflow => Err("no error for a row that does not fit".into()),
        Ok(got) => {
            let g = vals(got.as_ref());
            let expected: Vec<Val> = rows
                .iter()
                .enumerate()
                .map(|(i, r)| match r {
                    Row::Value(x) => x.clone(),
                    Row::Overflow => Val::Null,
                    Row::Open => g.get(i).cloned().unwrap_or(Val::Null),
                })
                .collect();
            if spec {
                return check_rows(got, to, &expected).map_err(|e| format!("{e:?}"));
            }
            got.to_data()
                .validate_full()
                .map_err(|e| format!("invalid output: {e}"))?;
            match (0..g.len().max(expected.len())).find(|&i| g.get(i) != expected.get(i)) {
                _ if got.data_type() != to => Err(format!("type {}", got.data_type())),
                Some(i) => Err(format!(
                    "row {i}: got {:?}, expected {:?}",
                    g.get(i),
                    expected.get(i)
                )),
                None => Ok(()),
            }
        }
    }
}

/// Temporal casts against the rules above
pub fn temporal_cast((a, to, safe): TemporalCastCase) -> R {
    use arrow_cast::cast::{CastOptions, cast_with_options};
    check_in(a.as_ref())?;
    let from = a.data_type().clone();
    let v = vals(a.as_ref());
    let row_of = |x: &Val, known: bool| -> Row {
        match x {
            Val::Null => Row::Value(Val::Null),
            x if known => cast_row_known(&from, &to, x).unwrap_or_else(|| cast_row(&from, &to, x)),
            x => cast_row(&from, &to, x),
        }
    };
    let rows: Vec<Row> = v.iter().map(|x| row_of(x, false)).collect();
    let result = cast_with_options(
        &a,
        &to,
        &CastOptions {
            safe,
            ..Default::default()
        },
    )
    .map_err(|e| e.to_string());
    // Integers cast to Time or Date64 keep their values even outside the
    // spec's range (one day, whole days): arrow-rs leaves that validation
    // optional, as apache/arrow-rs#5288 settled
    let zero_copy = matches!(from, DataType::Int32 | DataType::Int64)
        && matches!(
            to,
            DataType::Time32(_) | DataType::Time64(_) | DataType::Date64
        );
    let Err(m) = check_cast(&result, &rows, &to, safe, !zero_copy) else {
        return Ok(());
    };
    let known_rows: Vec<Row> = v.iter().map(|x| row_of(x, true)).collect();
    if v.iter()
        .any(|x| *x != Val::Null && cast_row_known(&from, &to, x).is_some())
        && check_cast(&result, &known_rows, &to, safe, true).is_ok()
    {
        return Err(crate::known::known(
            "known: coarser timestamp units truncate toward zero (apache/arrow-rs#11602)",
        ));
    }
    Err(fail(format!("{from} to {to} (safe: {safe}): {m}")))
}

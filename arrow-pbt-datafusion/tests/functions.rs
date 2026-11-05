//! The broadcast and row-by-row laws for DataFusion's scalar functions.
//! The broadcast law: calling a function with some arguments as scalars
//! gives the same result as calling it with those scalars broadcast to
//! arrays. The row-by-row law: each row of a call equals the call with every
//! argument as that row's scalar.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, Int64Array};
use arrow_buffer::{BooleanBuffer, NullBuffer};
use arrow_pbt::known::known;
use arrow_pbt::run::{caught, check};
use arrow_pbt::val::Val;
use arrow_pbt_datafusion::{Arg, Call, calls, check_broadcast, check_row_by_row};
use arrow_schema::{DataType, Field, FieldRef};
use datafusion_common::{Result, ScalarValue};
use datafusion_expr::type_coercion::functions::fields_with_udf;
use datafusion_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};
use datafusion_functions::{core, math, string, unicode};
use proptest::test_runner::TestCaseError;

/// Checks both laws for `udf` on calls with the argument types of one of
/// `signatures`. Each signature must be one that DataFusion's coercion leaves
/// unchanged, since the laws call the function directly, after the planner
/// would have coerced its arguments
fn laws(name: &str, udf: &ScalarUDF, signatures: &[Vec<DataType>]) {
    laws_with(name, udf, signatures, |_| Ok(()), |_, e| e);
}

/// `laws` with known-bug filters: `before` checks a call before the laws run
/// it, for a bug that aborts the process, and `after` maps a failure to its
/// known-bug discard
fn laws_with(
    name: &str,
    udf: &ScalarUDF,
    signatures: &[Vec<DataType>],
    before: impl Fn(&Call) -> Result<(), TestCaseError> + Sync,
    after: impl Fn(&Call, TestCaseError) -> TestCaseError + Sync,
) {
    for types in signatures {
        let fields: Vec<FieldRef> = types
            .iter()
            .map(|t| Arc::new(Field::new("arg", t.clone(), true)))
            .collect();
        let coerced: Vec<DataType> = fields_with_udf(&fields, udf)
            .unwrap_or_else(|e| panic!("{name} does not accept {types:?}: {e}"))
            .iter()
            .map(|f| f.data_type().clone())
            .collect();
        assert_eq!(&coerced, types, "{name}: coercion changes {types:?}");
    }
    if let Err(e) = calls(signatures) {
        panic!("{name}: {e}");
    }
    let cases = || calls(signatures).expect("checked above");
    check(name, cases, |call| {
        before(&call)?;
        caught(|| {
            check_broadcast(udf, &call)?;
            check_row_by_row(udf, &call)
        })
        .map_err(|e| after(&call, e))
    });
}

// `ceil` and `floor` turn a decimal scalar into an array of `number_rows`
// rows and read row 0, which panics when there are no rows. Found by this
// harness on DataFusion main at 926901f90 (apache/datafusion#26950)
fn decimal_scalar_no_rows(call: &Call, e: TestCaseError) -> TestCaseError {
    let decimal = matches!(
        call.fields[0].data_type(),
        DataType::Decimal32(..)
            | DataType::Decimal64(..)
            | DataType::Decimal128(..)
            | DataType::Decimal256(..)
    );
    match &e {
        TestCaseError::Fail(m)
            if decimal
                && call.number_rows == 0
                && m.to_string()
                    .contains("index 0 from a PrimitiveArray of length 0") =>
        {
            known(
                "known: ceil and floor panic on a decimal scalar with no rows (apache/datafusion#26950)",
            )
        }
        _ => e,
    }
}

/// Bytes above which a result costs too much to check: the function builds
/// it, and the oracle copies every row
const BUDGET: u128 = 64 << 20;

// `repeat` on `LargeUtf8` caps its result at `i64::MAX` bytes, so a result
// that passes the cap but no allocation can satisfy aborts the process. An
// abort cannot be caught, so `repeat_before` checks the result's size before
// the call: over `i32::MAX` bytes, where `Utf8` and `Utf8View` return an
// error. Found by this harness on DataFusion main at 926901f90
// (apache/datafusion#26948)
fn repeat_before(call: &Call) -> Result<(), TestCaseError> {
    let row = |a: &Arg, i: usize| match a {
        Arg::Array(g) => g.rows[i].clone(),
        Arg::Scalar(g) => g.rows[0].clone(),
    };
    // A call with only scalar arguments evaluates them once, even with no
    // rows
    let all_scalar = call.args.iter().all(|a| matches!(a, Arg::Scalar(_)));
    let n = if all_scalar {
        call.number_rows.max(1)
    } else {
        call.number_rows
    };
    let bytes: Vec<u128> = (0..n)
        .map(|i| match (row(&call.args[0], i), row(&call.args[1], i)) {
            (Val::Str(s), Val::I64(k)) if k > 0 => s.len() as u128 * k as u128,
            _ => 0,
        })
        .collect();
    let total: u128 = bytes.iter().sum();
    let utf8_max = i32::MAX as u128;
    if call.fields[0].data_type() == &DataType::LargeUtf8 && total > utf8_max {
        return Err(known(
            "known: repeat on LargeUtf8 aborts on a result no allocation can satisfy (apache/datafusion#26948)",
        ));
    }
    // Over `i32::MAX` bytes, `Utf8` and `Utf8View` return an error before they
    // allocate, so only results up to that size are costly. The row-by-row
    // law calls each row on its own, so a costly row counts too
    let costly = |b: u128| b > BUDGET && b <= utf8_max;
    if costly(total) || bytes.into_iter().any(costly) {
        return Err(TestCaseError::reject(
            "outside the budget: a result over 64 MiB",
        ));
    }
    Ok(())
}

// With an array of lengths, `lpad` and `rpad` add up the lengths as `usize`
// to estimate their capacity before they validate any length. A negative
// length wraps, and the sum overflows the capacity, which panics. A sum of
// large positive lengths asks for more than any allocation can satisfy,
// which aborts the process. The broadcast law turns a scalar length into an
// array, so a scalar length counts once per row. Found by this harness on
// DataFusion main at 926901f90 (apache/datafusion#26949)
const PAD_KNOWN: &str = "known: lpad and rpad estimate their capacity from lengths they have not validated (apache/datafusion#26949)";

fn pad_lengths(call: &Call) -> Vec<i64> {
    match &call.args[1] {
        Arg::Array(g) => g
            .rows
            .iter()
            .filter_map(|v| match v {
                Val::I64(n) => Some(*n),
                _ => None,
            })
            .collect(),
        Arg::Scalar(g) => match g.rows[0] {
            Val::I64(n) => vec![n; call.number_rows.max(1)],
            _ => vec![],
        },
    }
}

/// An abort cannot be caught, so this checks the sum of the lengths before
/// the call. Above `i32::MAX`, the capacity estimate exceeds what any single
/// valid length allows. Below it, a sum over the budget costs too much to
/// check
fn pad_before(call: &Call) -> Result<(), TestCaseError> {
    let total: u128 = pad_lengths(call)
        .into_iter()
        .map(|n| n.max(0) as u128)
        .sum();
    if total > i32::MAX as u128 {
        return Err(known(PAD_KNOWN));
    }
    // Each character takes up to 4 bytes
    if 4 * total > BUDGET {
        return Err(TestCaseError::reject(
            "outside the budget: a result over 64 MiB",
        ));
    }
    Ok(())
}

/// A negative length makes the capacity overflow, which panics
fn pad_after(call: &Call, e: TestCaseError) -> TestCaseError {
    let negative = pad_lengths(call).into_iter().any(|n| n < 0);
    match &e {
        TestCaseError::Fail(m) if negative && m.to_string().contains("capacity overflow") => {
            known(PAD_KNOWN)
        }
        _ => e,
    }
}

#[test]
fn lpad() {
    let types = [
        per_string(|s| vec![s, Int64]),
        per_string(|s| vec![s.clone(), Int64, s]),
    ]
    .concat();
    laws_with("lpad", &unicode::lpad(), &types, pad_before, pad_after);
}

#[test]
fn rpad() {
    let types = [
        per_string(|s| vec![s, Int64]),
        per_string(|s| vec![s.clone(), Int64, s]),
    ]
    .concat();
    laws_with("rpad", &unicode::rpad(), &types, pad_before, pad_after);
}

#[test]
fn ceil() {
    let types = one_of_each(&[floats(), decimals()].concat());
    laws_with(
        "ceil",
        &math::ceil(),
        &types,
        |_| Ok(()),
        decimal_scalar_no_rows,
    );
}

#[test]
fn floor() {
    let types = one_of_each(&[floats(), decimals()].concat());
    laws_with(
        "floor",
        &math::floor(),
        &types,
        |_| Ok(()),
        decimal_scalar_no_rows,
    );
}

#[test]
fn repeat() {
    let types = per_string(|s| vec![s, Int64]);
    laws_with("repeat", &string::repeat(), &types, repeat_before, |_, e| e);
}

fn one_of_each(types: &[DataType]) -> Vec<Vec<DataType>> {
    types.iter().map(|t| vec![t.clone()]).collect()
}

/// One signature for each string type, from a function of that type
fn per_string(f: impl Fn(DataType) -> Vec<DataType>) -> Vec<Vec<DataType>> {
    strings().into_iter().map(f).collect()
}

fn integers() -> Vec<DataType> {
    use DataType::*;
    vec![Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64]
}

fn floats() -> Vec<DataType> {
    vec![DataType::Float32, DataType::Float64]
}

fn decimals() -> Vec<DataType> {
    use DataType::*;
    vec![
        Decimal32(9, 2),
        Decimal64(18, 4),
        Decimal128(38, 10),
        Decimal256(76, 20),
    ]
}

fn numeric() -> Vec<DataType> {
    [integers(), floats(), decimals()].concat()
}

fn strings() -> Vec<DataType> {
    vec![DataType::Utf8, DataType::LargeUtf8, DataType::Utf8View]
}

/// Types that functions such as `nullif` and `coalesce` compare or choose
/// between
fn comparable() -> Vec<DataType> {
    vec![
        DataType::Boolean,
        DataType::Int32,
        DataType::Float64,
        DataType::Decimal128(20, 3),
        DataType::Date32,
        DataType::Utf8,
        DataType::Utf8View,
        DataType::Binary,
    ]
}

/// `n` arguments of each type in `types`
fn repeated(types: &[DataType], n: usize) -> Vec<Vec<DataType>> {
    types.iter().map(|t| vec![t.clone(); n]).collect()
}

/// Defines a test that checks both laws for each function and its signatures
macro_rules! law_tests {
    ($($name:ident: $udf:expr, $signatures:expr;)*) => {
        $(
            #[test]
            fn $name() {
                laws(stringify!($name), &$udf, &$signatures);
            }
        )*
    };
}

use DataType::{Float64, Int32, Int64};

law_tests! {
    // Math
    abs: math::abs(), one_of_each(&numeric());
    round: math::round(), [
        one_of_each(&[floats(), decimals()].concat()),
        [floats(), decimals()].concat().into_iter().map(|t| vec![t, Int32]).collect(),
    ].concat();
    trunc: math::trunc(), [
        one_of_each(&[floats(), decimals()].concat()),
        [floats(), decimals()].concat().into_iter().map(|t| vec![t, Int64]).collect(),
    ].concat();
    power: math::power(), [vec![Float64, Float64]];
    gcd: math::gcd(), [vec![Int64, Int64]];
    lcm: math::lcm(), [vec![Int64, Int64]];
    factorial: math::factorial(), [vec![Int64]];
    log: math::log(), [one_of_each(&floats()), repeated(&floats(), 2)].concat();
    iszero: math::iszero(), one_of_each(&numeric());
    isnan: math::isnan(), one_of_each(&numeric());
    nanvl: math::nanvl(), repeated(&floats(), 2);
    signum: math::signum(), one_of_each(&floats());
    sqrt: math::sqrt(), one_of_each(&floats());
    exp: math::exp(), one_of_each(&floats());
    ln: math::ln(), one_of_each(&floats());
    sin: math::sin(), one_of_each(&floats());
    cot: math::cot(), one_of_each(&floats());
    atan2: math::atan2(), repeated(&floats(), 2);
    // Strings
    upper: string::upper(), [
        one_of_each(&strings()),
        vec![vec![DataType::Dictionary(Box::new(Int32), Box::new(DataType::Utf8))]],
    ].concat();
    lower: string::lower(), one_of_each(&strings());
    btrim: string::btrim(), [one_of_each(&strings()), repeated(&strings(), 2)].concat();
    ltrim: string::ltrim(), [one_of_each(&strings()), repeated(&strings(), 2)].concat();
    rtrim: string::rtrim(), [one_of_each(&strings()), repeated(&strings(), 2)].concat();
    concat: string::concat(), [repeated(&strings(), 2), repeated(&strings(), 3)].concat();
    concat_ws: string::concat_ws(), repeated(&strings(), 3);
    replace: string::replace(), repeated(&strings(), 3);
    split_part: string::split_part(), per_string(|s| vec![s.clone(), s, Int64]);
    starts_with: string::starts_with(), repeated(&strings(), 2);
    ends_with: string::ends_with(), repeated(&strings(), 2);
    contains: string::contains(), repeated(&strings(), 2);
    levenshtein: string::levenshtein(), repeated(&strings(), 2);
    octet_length: string::octet_length(), one_of_each(&[
        strings(),
        vec![DataType::Binary, DataType::LargeBinary, DataType::BinaryView],
    ].concat());
    bit_length: string::bit_length(), one_of_each(&strings());
    ascii: string::ascii(), one_of_each(&strings());
    chr: string::chr(), [vec![Int64]];
    to_hex: string::to_hex(), one_of_each(&integers());
    // Unicode
    substr: unicode::substr(), [
        per_string(|s| vec![s, Int64]),
        per_string(|s| vec![s, Int64, Int64]),
    ].concat();
    character_length: unicode::character_length(), one_of_each(&strings());
    left: unicode::left(), per_string(|s| vec![s, Int64]);
    right: unicode::right(), per_string(|s| vec![s, Int64]);
    reverse: unicode::reverse(), one_of_each(&strings());
    strpos: unicode::strpos(), repeated(&strings(), 2);
    translate: unicode::translate(), per_string(|s| vec![s, DataType::Utf8, DataType::Utf8]);
    initcap: unicode::initcap(), one_of_each(&strings());
    find_in_set: unicode::find_in_set(), repeated(&strings(), 2);
    substr_index: unicode::substr_index(), per_string(|s| vec![s.clone(), s, Int64]);
    // Conditional
    nullif: core::nullif(), repeated(&comparable(), 2);
    nvl: core::nvl(), repeated(&[vec![DataType::Boolean], integers(), floats(), strings()].concat(), 2);
    nvl2: core::nvl2(), repeated(&comparable(), 3);
    coalesce: core::coalesce(), [repeated(&comparable(), 2), repeated(&comparable(), 3)].concat();
    greatest: core::greatest(), repeated(&comparable(), 2);
    least: core::least(), repeated(&comparable(), 2);
    overlay: core::overlay(), [
        per_string(|s| vec![s.clone(), s, Int64]),
        per_string(|s| vec![s.clone(), s, Int64, Int64]),
    ].concat();
}

/// How `Broken` is broken
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Mode {
    /// A scalar argument gives its value plus one
    ScalarOffByOne,
    /// An array argument gives one row too many
    ExtraRow,
    /// Every call panics
    Panic,
    /// An array argument reads its validity bitmap from bit 0, ignoring the
    /// offset a slice leaves in it
    NullsIgnoreOffset,
}

/// `Int64` identity, broken on purpose
#[derive(Debug, PartialEq, Eq, Hash)]
struct Broken {
    signature: Signature,
    mode: Mode,
}

impl Broken {
    fn udf(mode: Mode) -> ScalarUDF {
        ScalarUDF::new_from_impl(Self {
            signature: Signature::exact(vec![DataType::Int64], Volatility::Immutable),
            mode,
        })
    }
}

impl ScalarUDFImpl for Broken {
    fn name(&self) -> &'static str {
        "broken"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Int64)
    }
    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        assert!(self.mode != Mode::Panic, "broken panics");
        Ok(match &args.args[0] {
            ColumnarValue::Scalar(ScalarValue::Int64(v)) if self.mode == Mode::ScalarOffByOne => {
                ColumnarValue::Scalar(ScalarValue::Int64(v.map(|v| v.wrapping_add(1))))
            }
            ColumnarValue::Scalar(s) => ColumnarValue::Scalar(s.clone()),
            ColumnarValue::Array(a) => {
                let a = a.as_primitive::<Int64Type>();
                let out = match self.mode {
                    Mode::ExtraRow => a.iter().chain([None]).collect(),
                    Mode::NullsIgnoreOffset => {
                        let nulls = a.nulls().map(|n| {
                            NullBuffer::new(BooleanBuffer::new(n.buffer().clone(), 0, n.len()))
                        });
                        Int64Array::new(a.values().clone(), nulls)
                    }
                    _ => a.clone(),
                };
                ColumnarValue::Array(Arc::new(out))
            }
        })
    }
}

type Law = fn(&ScalarUDF, &Call) -> Result<(), TestCaseError>;

/// Each law, through the runner, finds each broken function it can see, and
/// the failure says why
#[test]
fn finds_broken_functions() {
    let broadcast: Law = check_broadcast;
    let row_by_row: Law = check_row_by_row;
    for (name, law, mode, expected) in [
        ("broadcast", broadcast, Mode::ScalarOffByOne, Some("row ")),
        ("broadcast", broadcast, Mode::ExtraRow, Some("rows for")),
        (
            "broadcast",
            broadcast,
            Mode::Panic,
            Some("panic: broken panics"),
        ),
        // Broadcasting passes array arguments through, so both calls read
        // the same wrong nulls
        ("broadcast", broadcast, Mode::NullsIgnoreOffset, None),
        ("row by row", row_by_row, Mode::ScalarOffByOne, Some("row ")),
        ("row by row", row_by_row, Mode::ExtraRow, Some("rows for")),
        (
            "row by row",
            row_by_row,
            Mode::Panic,
            Some("panic: broken panics"),
        ),
        (
            "row by row",
            row_by_row,
            Mode::NullsIgnoreOffset,
            Some("row "),
        ),
    ] {
        let udf = Broken::udf(mode);
        let cases = || calls(&[vec![DataType::Int64]]).unwrap();
        let found = catch_unwind(AssertUnwindSafe(|| {
            check("broken", cases, |call| law(&udf, &call));
        }));
        match (found, expected) {
            (Ok(()), None) => {}
            (Ok(()), Some(_)) => panic!("{name} {mode:?}: not found"),
            (Err(p), expected) => {
                let message = p.downcast_ref::<String>().cloned().unwrap_or_default();
                let Some(expected) = expected else {
                    panic!("{name} {mode:?}: found, but the law cannot see it: {message}");
                };
                assert!(message.contains(expected), "{name} {mode:?}: {message}");
            }
        }
    }
}

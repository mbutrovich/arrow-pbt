//! The broadcast law for a few of DataFusion's scalar functions: calling a
//! function with some arguments as scalars gives the same result as calling
//! it with those scalars broadcast to arrays.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array};
use arrow_pbt::run::check;
use arrow_pbt_datafusion::{calls, check_broadcast};
use arrow_schema::DataType;
use datafusion_common::{Result, ScalarValue};
use datafusion_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};
use datafusion_functions::{core, math, string, unicode};

fn one_of_each(types: &[DataType]) -> Vec<Vec<DataType>> {
    types.iter().map(|t| vec![t.clone()]).collect()
}

fn numeric() -> Vec<DataType> {
    use DataType::*;
    vec![
        Int8,
        Int16,
        Int32,
        Int64,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
        Float32,
        Float64,
        Decimal32(9, 2),
        Decimal64(18, 4),
        Decimal128(38, 10),
        Decimal256(76, 20),
    ]
}

fn strings() -> Vec<DataType> {
    vec![DataType::Utf8, DataType::LargeUtf8, DataType::Utf8View]
}

#[test]
fn abs() {
    let cases = calls(&one_of_each(&numeric())).unwrap();
    check("abs", cases, |call| check_broadcast(&math::abs(), &call));
}

#[test]
fn upper() {
    let dictionary = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
    let cases = calls(&one_of_each(&[strings(), vec![dictionary]].concat())).unwrap();
    check("upper", cases, |call| {
        check_broadcast(&string::upper(), &call)
    });
}

#[test]
fn substr() {
    let signatures: Vec<_> = strings()
        .into_iter()
        .flat_map(|s| {
            [
                vec![s.clone(), DataType::Int64],
                vec![s, DataType::Int64, DataType::Int64],
            ]
        })
        .collect();
    let cases = calls(&signatures).unwrap();
    check("substr", cases, |call| {
        check_broadcast(&unicode::substr(), &call)
    });
}

#[test]
fn nullif() {
    let signatures: Vec<_> = [
        DataType::Boolean,
        DataType::Int32,
        DataType::Float64,
        DataType::Decimal128(20, 3),
        DataType::Date32,
        DataType::Utf8,
        DataType::Utf8View,
        DataType::Binary,
    ]
    .into_iter()
    .map(|t| vec![t.clone(), t])
    .collect();
    let cases = calls(&signatures).unwrap();
    check("nullif", cases, |call| {
        check_broadcast(&core::nullif(), &call)
    });
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
                let mut v: Vec<Option<i64>> = a
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .iter()
                    .collect();
                if self.mode == Mode::ExtraRow {
                    v.push(None);
                }
                ColumnarValue::Array(Arc::new(Int64Array::from(v)) as ArrayRef)
            }
        })
    }
}

/// The harness and the runner find each broken function, and the failure
/// says why
#[test]
fn finds_broken_functions() {
    for (mode, expected) in [
        (Mode::ScalarOffByOne, "row "),
        (Mode::ExtraRow, "rows for"),
        (Mode::Panic, "panic: broken panics"),
    ] {
        let udf = Broken::udf(mode);
        let cases = calls(&[vec![DataType::Int64]]).unwrap();
        let found = catch_unwind(AssertUnwindSafe(|| {
            check("broken", cases, |call| check_broadcast(&udf, &call));
        }));
        let Err(p) = found else {
            panic!("{mode:?}: not found");
        };
        let message = p.downcast_ref::<String>().cloned().unwrap_or_default();
        assert!(message.contains(expected), "{mode:?}: {message}");
    }
}

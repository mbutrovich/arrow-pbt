//! Property-based tests for DataFusion, built on arrow-pbt's generator and
//! oracle. See `docs/design.md` in the repository, under "Downstream
//! adapters".

use std::sync::Arc;

use arrow_array::{Array, ArrayRef};
use arrow_pbt::generator::{GeneratedArray, GeneratorType, ValueDomain, array, len};
use arrow_pbt::val::{check_rows, vals};
use arrow_schema::{DataType, Field, FieldRef};
use datafusion_common::config::ConfigOptions;
use datafusion_common::{DataFusionError, ScalarValue};
use datafusion_expr::{ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, Volatility};
use proptest::prelude::*;
use proptest::sample::select;
use proptest::test_runner::TestCaseError;

mod expr;

pub use expr::{
    Build, ExprCase, Operand, check_expr_broadcast, check_expr_row_by_row, check_expr_split, exprs,
};

/// One argument of a scalar function call
#[derive(Clone, Debug)]
pub enum Arg {
    /// An array of the call's number of rows
    Array(GeneratedArray),
    /// A scalar, from the one row of this array
    Scalar(GeneratedArray),
}

/// A scalar function call: each argument's field, the arguments, and the
/// number of rows
#[derive(Clone, Debug)]
pub struct Call {
    pub fields: Vec<FieldRef>,
    pub args: Vec<Arg>,
    pub number_rows: usize,
}

/// Calls with the argument types of one of `signatures`. Each argument's
/// field is nullable three times in four, and each argument is an array of
/// the call's rows or a scalar. Values come from `ValueDomain::Logical`,
/// since DataFusion treats a decimal past its declared precision as invalid
/// (apache/datafusion#26774)
///
/// # Errors
///
/// No signature, or a type the generator cannot build
///
/// # Example
///
/// ```
/// use arrow_pbt_datafusion::calls;
/// use arrow_schema::DataType;
/// use proptest::strategy::{Strategy, ValueTree};
/// use proptest::test_runner::TestRunner;
/// let mut runner = TestRunner::deterministic();
/// let strategy = calls(&[vec![DataType::Utf8, DataType::Int64]]).unwrap();
/// let call = strategy.new_tree(&mut runner).unwrap().current();
/// assert_eq!(call.args.len(), 2);
/// assert!(calls(&[vec![DataType::Float16]]).is_err());
/// ```
pub fn calls(signatures: &[Vec<DataType>]) -> Result<BoxedStrategy<Call>, String> {
    if signatures.is_empty() {
        return Err("no signatures".to_string());
    }
    // Convert every type now, so that an unsupported one is an error here
    // rather than a panic during the run
    let signatures: Vec<Vec<ArgType>> = signatures
        .iter()
        .map(|types| types.iter().map(ArgType::new).collect())
        .collect::<Result<_, _>>()?;
    Ok((select(signatures), len())
        .prop_flat_map(|(types, n)| {
            let args: Vec<_> = types.into_iter().map(|t| t.arg(n)).collect();
            (args, Just(n))
        })
        .prop_map(|(args, number_rows)| {
            let (fields, args) = args.into_iter().unzip();
            Call {
                fields,
                args,
                number_rows,
            }
        })
        .boxed())
}

/// An argument type, converted for a nullable field and, where the type
/// allows one, a non-nullable field
#[derive(Clone, Debug)]
struct ArgType {
    nullable: (FieldRef, GeneratorType),
    non_nullable: Option<(FieldRef, GeneratorType)>,
}

impl ArgType {
    fn new(dt: &DataType) -> Result<Self, String> {
        let convert = |nullable: bool| {
            let f = Arc::new(Field::new("arg", dt.clone(), nullable));
            GeneratorType::try_from(f.as_ref()).map(|t| (f, t))
        };
        Ok(Self {
            nullable: convert(true)?,
            // Every row of a `Null` field is null
            non_nullable: convert(false).ok(),
        })
    }

    /// The field, nullable three times in four when the type allows both
    fn field(self) -> BoxedStrategy<(FieldRef, GeneratorType)> {
        match self.non_nullable {
            Some(nn) => prop_oneof![3 => Just(self.nullable), 1 => Just(nn)].boxed(),
            None => Just(self.nullable).boxed(),
        }
    }

    fn arg(self, n: usize) -> BoxedStrategy<(FieldRef, Arg)> {
        self.field()
            .prop_flat_map(move |(f, t)| {
                let rows = array(&t, n, ValueDomain::Logical).prop_map(Arg::Array);
                let scalar = array(&t, 1, ValueDomain::Logical).prop_map(Arg::Scalar);
                (Just(f), prop_oneof![rows, scalar])
            })
            .boxed()
    }
}

/// Checks the broadcast law for `udf` on `call`: the call as drawn and the
/// call with every scalar argument broadcast to an array both fail, or both
/// return valid output of the promised type and nullability with the same
/// values. Two kinds of case are outside the law and discarded: the calls
/// promise different return fields, or there are no rows and only the call
/// as drawn fails. With no rows, broadcasting drops the scalars' values,
/// which DataFusion still evaluates once
///
/// # Errors
///
/// A failure that says how the calls differ, or a result with the wrong
/// number of rows. A panic in the function propagates, and
/// `arrow_pbt::run::check` reports it as a failure
///
/// # Example
///
/// ```
/// use arrow_pbt::run::check;
/// use arrow_pbt_datafusion::{calls, check_broadcast};
/// use arrow_schema::DataType;
/// use datafusion_functions::math::abs;
/// let strategy = || calls(&[vec![DataType::Int32]]).unwrap();
/// check("abs", strategy, |call| check_broadcast(&abs(), &call));
/// ```
pub fn check_broadcast(udf: &ScalarUDF, call: &Call) -> Result<(), TestCaseError> {
    let n = call.number_rows;
    let as_drawn = args(call)?;
    let broadcast = as_drawn
        .iter()
        .map(|v| v.to_array(n).map(ColumnarValue::Array))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| fail(format!("ColumnarValue::to_array: {e}")))?;
    let config = Arc::new(ConfigOptions::default());
    let drawn = invoke(udf, &call.fields, as_drawn, n, &config)?;
    let arrays = invoke(udf, &call.fields, broadcast, n, &config)?;
    if promises_differ(&drawn, &arrays) {
        return Err(outside_promise());
    }
    match (drawn.result, arrays.result) {
        (Err(_), Err(_)) => Ok(()),
        (Ok(_), Err(e)) => Err(fail(format!(
            "only the call with scalars broadcast fails: {e}"
        ))),
        (Err(_), Ok(_)) if n == 0 => Err(TestCaseError::reject(
            "outside the law: with no rows, only the call as drawn sees the scalars",
        )),
        (Err(e), Ok(_)) => Err(fail(format!("only the call as drawn fails: {e}"))),
        (Ok((f, out)), Ok((_, expected))) => {
            if !f.is_nullable() && expected.logical_null_count() > 0 {
                return Err(fail(format!(
                    "{} promised a non-nullable result and returned nulls",
                    udf.name()
                )));
            }
            check_rows(&expected, f.data_type(), &vals(&expected))?;
            check_rows(&out, f.data_type(), &vals(&expected))
        }
    }
}

/// Checks the row-by-row law for `udf` on `call`: each row of the call as
/// drawn equals the call with every argument as that row's scalar, and the
/// call as drawn fails only when some row fails on its own. A call with no
/// array argument, or with no rows, has no rows to call on their own, so it
/// passes. A row whose call promises a different return field is outside the
/// law and discards the case
///
/// # Errors
///
/// A failure that names the row that differs, a volatile function, whose
/// rows may differ from call to call, or a result with the wrong number of
/// rows. A panic in the function propagates, and `arrow_pbt::run::check`
/// reports it as a failure
///
/// # Example
///
/// ```
/// use arrow_pbt::run::check;
/// use arrow_pbt_datafusion::{calls, check_row_by_row};
/// use arrow_schema::DataType;
/// use datafusion_functions::math::abs;
/// let strategy = || calls(&[vec![DataType::Int32]]).unwrap();
/// check("abs", strategy, |call| check_row_by_row(&abs(), &call));
/// ```
pub fn check_row_by_row(udf: &ScalarUDF, call: &Call) -> Result<(), TestCaseError> {
    if udf.signature().volatility == Volatility::Volatile {
        return Err(fail(format!(
            "{} is volatile, so the law does not hold",
            udf.name()
        )));
    }
    let n = call.number_rows;
    if n == 0 || call.args.iter().all(|a| matches!(a, Arg::Scalar(_))) {
        return Ok(());
    }
    let as_drawn = args(call)?;
    let config = Arc::new(ConfigOptions::default());
    let row = |i: usize| {
        let scalars = as_drawn
            .iter()
            .map(|v| match v {
                ColumnarValue::Array(a) => ScalarValue::try_from_array(a, i),
                ColumnarValue::Scalar(s) => Ok(s.clone()),
            })
            .map(|s| s.map(ColumnarValue::Scalar))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| fail(format!("ScalarValue::try_from_array: {e}")))?;
        invoke(udf, &call.fields, scalars, 1, &config)
    };
    let drawn = invoke(udf, &call.fields, as_drawn.clone(), n, &config)?;
    let mut rows = (0..n).map(|i| {
        let r = row(i)?;
        if promises_differ(&drawn, &r) {
            return Err(outside_promise());
        }
        Ok((i, r.result))
    });
    match &drawn.result {
        Err(e) => {
            for r in rows {
                if r?.1.is_err() {
                    return Ok(());
                }
            }
            Err(fail(format!(
                "the call as drawn fails, and every row succeeds on its own: {e}"
            )))
        }
        Ok((f, out)) => {
            let expected = vals(out);
            check_rows(out, f.data_type(), &expected)?;
            for r in &mut rows {
                let (i, r) = r?;
                let (_, r) = r.map_err(|e| fail(format!("row {i} fails on its own: {e}")))?;
                check_rows(&r, f.data_type(), &expected[i..=i])
                    .map_err(|e| fail(format!("row {i} on its own differs: {e}")))?;
            }
            Ok(())
        }
    }
}

/// The arguments of `call` as `ColumnarValue`s
fn args(call: &Call) -> Result<Vec<ColumnarValue>, TestCaseError> {
    call.args
        .iter()
        .map(|a| match a {
            Arg::Array(g) => Ok(ColumnarValue::Array(g.build())),
            Arg::Scalar(g) => ScalarValue::try_from_array(&g.build(), 0)
                .map(ColumnarValue::Scalar)
                .map_err(|e| fail(format!("ScalarValue::try_from_array: {e}"))),
        })
        .collect()
}

/// What one call of a function did
struct Invocation {
    /// The return field the function promised, if it promised one
    field: Option<FieldRef>,
    /// That field and the result as rows, or the error DataFusion returned
    result: Result<(FieldRef, ArrayRef), DataFusionError>,
}

/// Whether two calls promise different return fields, as a function whose
/// return type depends on a scalar argument's value does. The laws compare
/// calls only when they promise the same field
fn promises_differ(a: &Invocation, b: &Invocation) -> bool {
    matches!((&a.field, &b.field), (Some(f), Some(g)) if f != g)
}

fn outside_promise() -> TestCaseError {
    TestCaseError::reject("outside the law: the return field depends on scalar arguments")
}

/// Calls `udf` on `n` rows. A result of the wrong length is a failure
fn invoke(
    udf: &ScalarUDF,
    fields: &[FieldRef],
    args: Vec<ColumnarValue>,
    n: usize,
    config: &Arc<ConfigOptions>,
) -> Result<Invocation, TestCaseError> {
    let scalars: Vec<Option<&ScalarValue>> = args
        .iter()
        .map(|a| match a {
            ColumnarValue::Scalar(s) => Some(s),
            ColumnarValue::Array(_) => None,
        })
        .collect();
    let return_field = match udf.return_field_from_args(ReturnFieldArgs {
        arg_fields: fields,
        scalar_arguments: &scalars,
    }) {
        Ok(f) => f,
        Err(e) => {
            return Ok(Invocation {
                field: None,
                result: Err(e),
            });
        }
    };
    let all_scalar = !args.is_empty() && scalars.iter().all(Option::is_some);
    let result = match udf.invoke_with_args(ScalarFunctionArgs {
        args: args.clone(),
        arg_fields: fields.to_vec(),
        number_rows: n,
        return_field: Arc::clone(&return_field),
        config_options: Arc::clone(config),
    }) {
        Ok(r) => r,
        Err(e) => {
            return Ok(Invocation {
                field: Some(return_field),
                result: Err(e),
            });
        }
    };
    let out = match result {
        ColumnarValue::Scalar(s) => s.to_array_of_size(n),
        ColumnarValue::Array(a) if a.len() == n => Ok(a),
        // As `ScalarFunctionExpr::evaluate` reads it, one row from only
        // scalar arguments is a scalar
        // (datafusion/physical-expr/src/scalar_function.rs)
        ColumnarValue::Array(a) if a.len() == 1 && all_scalar => {
            ScalarValue::try_from_array(&a, 0).and_then(|s| s.to_array_of_size(n))
        }
        ColumnarValue::Array(a) => {
            return Err(fail(format!(
                "{} returned {} rows for {n}",
                udf.name(),
                a.len()
            )));
        }
    };
    Ok(Invocation {
        field: Some(Arc::clone(&return_field)),
        result: out.map(|a| (return_field, a)),
    })
}

fn fail(msg: String) -> TestCaseError {
    TestCaseError::fail(msg)
}

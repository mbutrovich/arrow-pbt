//! Property-based tests for DataFusion, built on arrow-pbt's generator and
//! oracle. See `docs/design.md` in the repository, under "Downstream
//! adapters".

use std::sync::Arc;

use arrow_array::{Array, ArrayRef};
use arrow_pbt::generator::{GeneratedArray, GeneratorType, array, len};
use arrow_pbt::val::{check_rows, vals};
use arrow_schema::{DataType, Field, FieldRef};
use datafusion_common::config::ConfigOptions;
use datafusion_common::{DataFusionError, ScalarValue};
use datafusion_expr::{ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF};
use proptest::prelude::*;
use proptest::sample::select;
use proptest::test_runner::TestCaseError;

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
/// the call's rows or a scalar
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

    fn arg(self, n: usize) -> BoxedStrategy<(FieldRef, Arg)> {
        let field = match self.non_nullable {
            Some(nn) => prop_oneof![3 => Just(self.nullable), 1 => Just(nn)].boxed(),
            None => Just(self.nullable).boxed(),
        };
        field
            .prop_flat_map(move |(f, t)| {
                let rows = array(&t, n).prop_map(Arg::Array);
                let scalar = array(&t, 1).prop_map(Arg::Scalar);
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
/// let strategy = calls(&[vec![DataType::Int32]]).unwrap();
/// check("abs", strategy, |call| check_broadcast(&abs(), &call));
/// ```
pub fn check_broadcast(udf: &ScalarUDF, call: &Call) -> Result<(), TestCaseError> {
    let n = call.number_rows;
    let mut as_drawn = Vec::with_capacity(call.args.len());
    for a in &call.args {
        as_drawn.push(match a {
            Arg::Array(g) => ColumnarValue::Array(g.build()),
            Arg::Scalar(g) => ColumnarValue::Scalar(
                ScalarValue::try_from_array(&g.build(), 0)
                    .map_err(|e| fail(format!("ScalarValue::try_from_array: {e}")))?,
            ),
        });
    }
    let broadcast = as_drawn
        .iter()
        .map(|v| v.to_array(n).map(ColumnarValue::Array))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| fail(format!("ColumnarValue::to_array: {e}")))?;
    let config = Arc::new(ConfigOptions::default());
    let drawn = invoke(udf, &call.fields, as_drawn, n, &config)?;
    let arrays = invoke(udf, &call.fields, broadcast, n, &config)?;
    match (drawn, arrays) {
        (Err(_), Err(_)) => Ok(()),
        (Ok(_), Err(e)) => Err(fail(format!(
            "only the call with scalars broadcast fails: {e}"
        ))),
        (Err(_), Ok(_)) if n == 0 => Err(TestCaseError::reject(
            "outside the law: with no rows, only the call as drawn sees the scalars",
        )),
        (Err(e), Ok(_)) => Err(fail(format!("only the call as drawn fails: {e}"))),
        (Ok((f, out)), Ok((expected_field, expected))) => {
            if f != expected_field {
                return Err(TestCaseError::reject(
                    "outside the law: the return field depends on scalar arguments",
                ));
            }
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

/// Calls `udf` and returns its return field and its result as `n` rows, or
/// the error DataFusion returned. A result of the wrong length is a failure
fn invoke(
    udf: &ScalarUDF,
    fields: &[FieldRef],
    args: Vec<ColumnarValue>,
    n: usize,
    config: &Arc<ConfigOptions>,
) -> Result<Result<(FieldRef, ArrayRef), DataFusionError>, TestCaseError> {
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
        Err(e) => return Ok(Err(e)),
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
        Err(e) => return Ok(Err(e)),
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
    Ok(out.map(|a| (return_field, a)))
}

fn fail(msg: String) -> TestCaseError {
    TestCaseError::fail(msg)
}

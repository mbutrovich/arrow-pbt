//! Laws for DataFusion's physical expressions, on generated record batches

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, RecordBatch, RecordBatchOptions};
use arrow_pbt::generator::{GeneratedArray, GeneratedBatch, ValueDomain, array, batches};
use arrow_pbt::val::{check_rows, vals};
use arrow_schema::{DataType, FieldRef, Schema, SchemaRef};
use arrow_select::concat::concat_batches;
use datafusion_common::{DataFusionError, ScalarValue};
use datafusion_expr::ColumnarValue;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::{Column, Literal};
use proptest::prelude::*;
use proptest::sample::select;
use proptest::test_runner::TestCaseError;

use crate::{ArgType, fail};

/// Builds an expression from its operands, which come in the order of the
/// case's fields, and the schema of the batch it will see. The row-by-row
/// law passes a schema with no fields, since every operand is then a literal
pub type Build =
    dyn Fn(Vec<Arc<dyn PhysicalExpr>>, &Schema) -> datafusion_common::Result<Arc<dyn PhysicalExpr>>;

/// One operand of an expression
#[derive(Clone, Debug)]
pub enum Operand {
    /// The operand's column of each batch
    Column,
    /// A literal, from the one row of this array
    Literal(GeneratedArray),
}

/// An expression's operands, one for each field of `schema`, and the batches
/// to evaluate it on. A batch holds a column for each column operand, in the
/// order of the operands, so its schema is `schema` without the fields of the
/// literal operands
#[derive(Clone, Debug)]
pub struct ExprCase {
    pub schema: SchemaRef,
    pub operands: Vec<Operand>,
    pub batches: Vec<GeneratedBatch>,
}

/// Which operands of a case are literals
#[derive(Clone, Copy, Debug)]
enum Literals {
    None,
    All,
    /// All but the first operand, which is often the input, such as the
    /// value an `IN` list tests
    AllButFirst,
    /// Each operand one time in two
    Some,
}

/// Cases with operands of the types of one of `signatures`, each operand a
/// column or a literal. A case's operands are all columns, all literals, all
/// literals but the first, or a mix, because expressions choose how to
/// evaluate from which operands are literals. Each field is nullable three
/// times in four, and the batches are a stream as `batches` draws it. Values
/// come from `ValueDomain::Logical`, since DataFusion treats a decimal past
/// its declared precision as invalid (apache/datafusion#26774)
///
/// # Errors
///
/// No signature, or a type the generator cannot build
///
/// # Example
///
/// ```
/// use arrow_pbt_datafusion::{Operand, exprs};
/// use arrow_schema::DataType;
/// use proptest::strategy::{Strategy, ValueTree};
/// use proptest::test_runner::TestRunner;
/// let mut runner = TestRunner::deterministic();
/// let strategy = exprs(&[vec![DataType::Boolean, DataType::Utf8]]).unwrap();
/// let case = strategy.new_tree(&mut runner).unwrap().current();
/// assert_eq!(case.operands.len(), 2);
/// let columns = case.operands.iter().filter(|o| matches!(o, Operand::Column)).count();
/// assert!(case.batches.iter().all(|b| b.schema.fields().len() == columns));
/// assert!(exprs(&[vec![DataType::Float16]]).is_err());
/// ```
pub fn exprs(signatures: &[Vec<DataType>]) -> Result<BoxedStrategy<ExprCase>, String> {
    if signatures.is_empty() {
        return Err("no signatures".to_string());
    }
    let signatures: Vec<Vec<ArgType>> = signatures
        .iter()
        .map(|types| types.iter().map(ArgType::new).collect())
        .collect::<Result<_, _>>()?;
    let literals = prop_oneof![
        1 => Just(Literals::None),
        1 => Just(Literals::All),
        1 => Just(Literals::AllButFirst),
        2 => Just(Literals::Some),
    ];
    Ok((select(signatures), literals)
        .prop_flat_map(|(types, literals)| {
            let fields: Vec<_> = types.into_iter().map(ArgType::field).collect();
            (fields, Just(literals))
        })
        .prop_flat_map(|(fields, literals)| {
            let schema = Arc::new(Schema::new(
                fields
                    .iter()
                    .enumerate()
                    .map(|(i, (f, _))| f.as_ref().clone().with_name(format!("c{i}")))
                    .collect::<Vec<_>>(),
            ));
            let operands: Vec<_> = fields
                .iter()
                .enumerate()
                .map(|(i, (_, t))| {
                    let literal = array(t, 1, ValueDomain::Logical).prop_map(Operand::Literal);
                    match literals {
                        Literals::None => Just(Operand::Column).boxed(),
                        Literals::AllButFirst if i == 0 => Just(Operand::Column).boxed(),
                        Literals::All | Literals::AllButFirst => literal.boxed(),
                        Literals::Some => prop_oneof![Just(Operand::Column), literal].boxed(),
                    }
                })
                .collect();
            (Just(schema), operands)
        })
        .prop_flat_map(|(schema, operands): (SchemaRef, Vec<Operand>)| {
            let columns: Vec<FieldRef> = schema
                .fields()
                .iter()
                .zip(&operands)
                .filter(|(_, o)| matches!(o, Operand::Column))
                .map(|(f, _)| Arc::clone(f))
                .collect();
            // The fields' types converted above, so they convert again
            let batches = batches(&Arc::new(Schema::new(columns)), ValueDomain::Logical)
                .expect("converted types");
            (Just(schema), Just(operands), batches)
        })
        .prop_map(|(schema, operands, batches)| ExprCase {
            schema,
            operands,
            batches,
        })
        .boxed())
}

/// Checks the broadcast law for the expression `build` makes on each batch
/// of `case`: the expression as drawn and the expression with every literal
/// replaced by a column holding its value both fail, or both return valid
/// output of the type and nullability they promise, with the same values. A
/// batch with no rows where only the expression as drawn fails is outside the
/// law and discards the case, since a column of no rows drops the literal's
/// value
///
/// # Errors
///
/// A failure that says how the two differ, or a result with the wrong number
/// of rows. A panic propagates, and `arrow_pbt::run::check` reports it as a
/// failure
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use arrow_pbt::run::check;
/// use arrow_pbt_datafusion::{check_expr_broadcast, exprs};
/// use arrow_schema::{DataType, Schema};
/// use datafusion_physical_expr::PhysicalExpr;
/// use datafusion_physical_expr::expressions::not;
/// let build = |ops: Vec<Arc<dyn PhysicalExpr>>, _: &Schema| not(Arc::clone(&ops[0]));
/// let strategy = exprs(&[vec![DataType::Boolean]]).unwrap();
/// check("not", strategy, |case| check_expr_broadcast(&build, &case));
/// ```
pub fn check_expr_broadcast(build: &Build, case: &ExprCase) -> Result<(), TestCaseError> {
    let literals = literals(case)?;
    for g in &case.batches {
        let batch = g.build();
        let n = batch.num_rows();
        let drawn = evaluate(build, operands(case, &literals, &batch, None)?, &batch)?;
        let mut drawn_columns = batch.columns().iter();
        let columns = literals
            .iter()
            .map(|l| match l {
                Some(l) => l
                    .to_array_of_size(n)
                    .map_err(|e| fail(format!("ScalarValue::to_array_of_size: {e}"))),
                None => Ok(Arc::clone(
                    drawn_columns.next().expect("a column per operand"),
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let options = RecordBatchOptions::new().with_row_count(Some(n));
        let broadcast =
            RecordBatch::try_new_with_options(Arc::clone(&case.schema), columns, &options)
                .map_err(|e| fail(format!("a batch of the literals broadcast: {e}")))?;
        let columns = (0..case.operands.len())
            .map(|i| column(case, i, i))
            .collect();
        let arrays = evaluate(build, columns, &broadcast)?;
        match (drawn, arrays) {
            (Err(_), Err(_)) => {}
            (Ok(_), Err(e)) => {
                return Err(fail(format!(
                    "only the expression with literals broadcast fails: {e}"
                )));
            }
            (Err(_), Ok(_)) if n == 0 => {
                return Err(TestCaseError::reject(
                    "outside the law: with no rows, only the expression as drawn sees the literals",
                ));
            }
            (Err(e), Ok(_)) => {
                return Err(fail(format!("only the expression as drawn fails: {e}")));
            }
            (Ok((f, out)), Ok((g, expected))) => {
                if f.data_type() != g.data_type() {
                    return Err(fail(format!(
                        "the expression as drawn returns {} and with literals broadcast {}",
                        f.data_type(),
                        g.data_type()
                    )));
                }
                check_rows(&out, f.data_type(), &vals(&expected)).map_err(|e| {
                    fail(format!(
                        "the expression with literals broadcast differs: {e}"
                    ))
                })?;
            }
        }
    }
    Ok(())
}

/// Checks the row-by-row law for the expression `build` makes on each batch
/// of `case`: each row of the expression as drawn equals the expression with
/// every operand as that row's literal, and the expression as drawn fails
/// only when some row fails on its own. Every operand of a row is a literal,
/// so each row is evaluated on a batch of one row with no columns. A case
/// whose operands are all literals has nothing to evaluate row by row, and
/// passes
///
/// # Errors
///
/// A failure that names the row that differs, or a result with the wrong
/// number of rows. A panic propagates, and `arrow_pbt::run::check` reports
/// it as a failure
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use arrow_pbt::run::check;
/// use arrow_pbt_datafusion::{check_expr_row_by_row, exprs};
/// use arrow_schema::{DataType, Schema};
/// use datafusion_physical_expr::PhysicalExpr;
/// use datafusion_physical_expr::expressions::not;
/// let build = |ops: Vec<Arc<dyn PhysicalExpr>>, _: &Schema| not(Arc::clone(&ops[0]));
/// let strategy = exprs(&[vec![DataType::Boolean]]).unwrap();
/// check("not", strategy, |case| check_expr_row_by_row(&build, &case));
/// ```
pub fn check_expr_row_by_row(build: &Build, case: &ExprCase) -> Result<(), TestCaseError> {
    if case
        .operands
        .iter()
        .all(|o| matches!(o, Operand::Literal(_)))
    {
        return Ok(());
    }
    let literals = literals(case)?;
    let one = RecordBatch::try_new_with_options(
        Arc::new(Schema::empty()),
        vec![],
        &RecordBatchOptions::new().with_row_count(Some(1)),
    )
    .map_err(|e| fail(format!("a batch of one row: {e}")))?;
    for g in &case.batches {
        let batch = g.build();
        let n = batch.num_rows();
        if n == 0 {
            continue;
        }
        let drawn = evaluate(build, operands(case, &literals, &batch, None)?, &batch)?;
        let row = |i: usize| evaluate(build, operands(case, &literals, &batch, Some(i))?, &one);
        match drawn {
            Err(e) => {
                let mut any = false;
                for i in 0..n {
                    any |= row(i)?.is_err();
                    if any {
                        break;
                    }
                }
                if !any {
                    return Err(fail(format!(
                        "the expression as drawn fails, and every row succeeds on its own: {e}"
                    )));
                }
            }
            Ok((f, out)) => {
                let expected = vals(&out);
                for i in 0..n {
                    let (_, r) =
                        row(i)?.map_err(|e| fail(format!("row {i} fails on its own: {e}")))?;
                    check_rows(&r, f.data_type(), &expected[i..=i])
                        .map_err(|e| fail(format!("row {i} on its own differs: {e}")))?;
                }
            }
        }
    }
    Ok(())
}

/// Checks the split law for the expression `build` makes on the batches of
/// `case`: evaluating the batches one at a time gives the rows of evaluating
/// them concatenated into one batch, and the concatenated batch fails only
/// when some batch fails on its own. It holds for expressions that compute
/// each row from that row alone, which excludes volatile ones
///
/// # Errors
///
/// A failure that names the batch that differs, or a result with the wrong
/// number of rows. A panic propagates, and `arrow_pbt::run::check` reports
/// it as a failure
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use arrow_pbt::run::check;
/// use arrow_pbt_datafusion::{check_expr_split, exprs};
/// use arrow_schema::{DataType, Schema};
/// use datafusion_physical_expr::PhysicalExpr;
/// use datafusion_physical_expr::expressions::not;
/// let build = |ops: Vec<Arc<dyn PhysicalExpr>>, _: &Schema| not(Arc::clone(&ops[0]));
/// let strategy = exprs(&[vec![DataType::Boolean]]).unwrap();
/// check("not", strategy, |case| check_expr_split(&build, &case));
/// ```
pub fn check_expr_split(build: &Build, case: &ExprCase) -> Result<(), TestCaseError> {
    if case.batches.is_empty() {
        return Ok(());
    }
    let parts: Vec<RecordBatch> = case.batches.iter().map(GeneratedBatch::build).collect();
    let whole = concat_batches(&parts[0].schema(), &parts)
        .map_err(|e| fail(format!("concat_batches: {e}")))?;
    let literals = literals(case)?;
    let all = evaluate(build, operands(case, &literals, &whole, None)?, &whole)?;
    let each = parts
        .iter()
        .map(|b| evaluate(build, operands(case, &literals, b, None)?, b))
        .collect::<Result<Vec<_>, _>>()?;
    match all {
        Err(e) => {
            if each.iter().all(Result::is_ok) {
                return Err(fail(format!(
                    "the concatenated batches fail, and every batch succeeds on its own: {e}"
                )));
            }
        }
        Ok((f, out)) => {
            let expected = vals(&out);
            let mut start = 0;
            for (j, r) in each.into_iter().enumerate() {
                let (_, a) = r.map_err(|e| fail(format!("batch {j} fails on its own: {e}")))?;
                check_rows(&a, f.data_type(), &expected[start..start + a.len()])
                    .map_err(|e| fail(format!("batch {j} on its own differs: {e}")))?;
                start += a.len();
            }
        }
    }
    Ok(())
}

/// The operand expressions of `case`. With `row`, each column operand is
/// instead a literal of that row of `batch`
fn operands(
    case: &ExprCase,
    literals: &[Option<ScalarValue>],
    batch: &RecordBatch,
    row: Option<usize>,
) -> Result<Vec<Arc<dyn PhysicalExpr>>, TestCaseError> {
    // The batch's column of each column operand
    let mut j = 0;
    literals
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let Some(s) = l else {
                j += 1;
                return match row {
                    None => Ok(column(case, i, j - 1)),
                    Some(r) => ScalarValue::try_from_array(batch.column(j - 1), r)
                        .map(|s| Arc::new(Literal::new(s)) as Arc<dyn PhysicalExpr>)
                        .map_err(|e| fail(format!("ScalarValue::try_from_array: {e}"))),
                };
            };
            Ok(Arc::new(Literal::new(s.clone())) as Arc<dyn PhysicalExpr>)
        })
        .collect()
}

/// The value of each literal operand of `case`, built once for all the
/// expressions a law evaluates
fn literals(case: &ExprCase) -> Result<Vec<Option<ScalarValue>>, TestCaseError> {
    case.operands
        .iter()
        .map(|op| match op {
            Operand::Column => Ok(None),
            Operand::Literal(l) => ScalarValue::try_from_array(&l.build(), 0)
                .map(Some)
                .map_err(|e| fail(format!("ScalarValue::try_from_array: {e}"))),
        })
        .collect()
}

/// Operand `i` as column `j` of a batch
fn column(case: &ExprCase, i: usize, j: usize) -> Arc<dyn PhysicalExpr> {
    Arc::new(Column::new(case.schema.field(i).name(), j))
}

/// Builds the expression and evaluates it on `batch`, returning the field it
/// promises and its rows, or DataFusion's error. Output of the wrong length,
/// of another type than it promises, or with nulls it promises not to hold,
/// is a failure
fn evaluate(
    build: &Build,
    operands: Vec<Arc<dyn PhysicalExpr>>,
    batch: &RecordBatch,
) -> Result<Result<(FieldRef, ArrayRef), DataFusionError>, TestCaseError> {
    let schema = batch.schema();
    let n = batch.num_rows();
    let evaluated = build(operands, &schema).and_then(|e| {
        let f = e.return_field(&schema)?;
        let v = e.evaluate(batch)?;
        Ok((e, f, v))
    });
    let (e, f, v) = match evaluated {
        Ok(r) => r,
        Err(e) => return Ok(Err(e)),
    };
    let out = match v {
        ColumnarValue::Scalar(s) => match s.to_array_of_size(n) {
            Ok(a) => a,
            Err(e) => return Ok(Err(e)),
        },
        ColumnarValue::Array(a) if a.len() == n => a,
        ColumnarValue::Array(a) => {
            return Err(fail(format!("{e} returned {} rows for {n}", a.len())));
        }
    };
    if !f.is_nullable() && out.logical_null_count() > 0 {
        return Err(fail(format!(
            "{e} promised a non-nullable result and returned nulls"
        )));
    }
    check_rows(&out, f.data_type(), &vals(&out))?;
    Ok(Ok((f, out)))
}

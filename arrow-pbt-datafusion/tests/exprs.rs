//! The broadcast, row-by-row and split laws for DataFusion's physical
//! expressions. The broadcast law: an expression with literal operands gives
//! the same result as with those literals broadcast to columns. The
//! row-by-row law: each row equals the expression with every operand as that
//! row's literal. The split law: evaluating batches one at a time gives the
//! same rows as evaluating them concatenated.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch};
use arrow_buffer::{BooleanBuffer, NullBuffer};
use arrow_pbt::run::check;
use arrow_pbt_datafusion::{
    Build, ExprCase, check_expr_broadcast, check_expr_row_by_row, check_expr_split, exprs,
};
use arrow_schema::{DataType, Field, FieldRef, Fields, Schema, TimeUnit};
use datafusion_common::Result;
use datafusion_expr::ColumnarValue;
use datafusion_expr::type_coercion::binary::comparison_coercion_with_session_timezone;
use datafusion_expr::type_coercion::other::get_coerce_type_for_case_expression;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::{Literal, case, in_list};
use proptest::test_runner::TestCaseError;

/// Checks the three laws for the expressions `build` makes, on operands of
/// the types of one of `signatures`
fn laws(name: &str, build: &Build, signatures: &[Vec<DataType>]) {
    let cases = exprs(signatures).unwrap_or_else(|e| panic!("{name}: {e}"));
    check(name, cases, |case: ExprCase| {
        check_expr_broadcast(build, &case)?;
        check_expr_row_by_row(build, &case)?;
        check_expr_split(build, &case)
    });
}

/// `CASE [base] WHEN w THEN t ... [ELSE e] END`, from operands that are the
/// base when `with_base`, then each WHEN followed by its THEN, then the ELSE
/// when one operand is left over
fn case_of(with_base: bool) -> Box<Build> {
    Box::new(move |ops: Vec<Arc<dyn PhysicalExpr>>, _: &Schema| {
        let mut ops = ops.into_iter();
        let base = if with_base { ops.next() } else { None };
        let mut rest: Vec<_> = ops.collect();
        let else_expr = (rest.len() % 2 == 1).then(|| rest.pop()).flatten();
        let when_thens = rest
            .chunks(2)
            .map(|p| (Arc::clone(&p[0]), Arc::clone(&p[1])))
            .collect();
        case(base, when_thens, else_expr)
    })
}

/// Operand types for `case_of`: the base and each WHEN of type `when`, each
/// THEN and the ELSE of one of `thens`, for one to three branches, with and
/// without an ELSE. The analyzer coerces CASE operands
/// (`coerce_case_expression` in datafusion-optimizer) and the laws build the
/// expression directly, so each set must be one that coercion leaves
/// unchanged
fn case_signatures(base: bool, when: &DataType, thens: &[DataType]) -> Vec<Vec<DataType>> {
    if base {
        assert_eq!(
            comparison_coercion_with_session_timezone(when, when, None).as_ref(),
            Some(when),
            "CASE {when} WHEN {when} coerces"
        );
    } else {
        assert_eq!(
            when,
            &DataType::Boolean,
            "a WHEN without a base is a condition"
        );
    }
    let mut out = vec![];
    for then in thens {
        for k in 1..=3 {
            for has_else in [false, true] {
                let then_types = vec![then.clone(); k];
                assert_eq!(
                    get_coerce_type_for_case_expression(&then_types, has_else.then_some(then))
                        .as_ref(),
                    Some(then),
                    "THEN {then} coerces"
                );
                let mut s: Vec<DataType> = base.then(|| when.clone()).into_iter().collect();
                for _ in 0..k {
                    s.push(when.clone());
                    s.push(then.clone());
                }
                if has_else {
                    s.push(then.clone());
                }
                out.push(s);
            }
        }
    }
    out
}

/// Result types for CASE: flat types of each physical kind, and nested ones
fn then_types() -> Vec<DataType> {
    let item = Arc::new(Field::new("item", DataType::Int32, true));
    vec![
        DataType::Boolean,
        DataType::Int32,
        DataType::Int64,
        DataType::Float64,
        DataType::Decimal128(10, 2),
        DataType::Utf8,
        DataType::LargeUtf8,
        DataType::Utf8View,
        DataType::Binary,
        DataType::Date32,
        DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        DataType::List(item),
        DataType::Struct(Fields::from(vec![Field::new("a", DataType::Int32, true)])),
    ]
}

#[test]
fn case_when() {
    laws(
        "case_when",
        &case_of(false),
        &case_signatures(false, &DataType::Boolean, &then_types()),
    );
}

/// Defines a test of `CASE base WHEN ...` for each base type
macro_rules! case_base_tests {
    ($($name:ident: $base:expr;)*) => {
        $(
            #[test]
            fn $name() {
                let thens = [DataType::Int32, DataType::Utf8, DataType::Float64, DataType::Boolean];
                laws(stringify!($name), &case_of(true), &case_signatures(true, &$base, &thens));
            }
        )*
    };
}

case_base_tests! {
    case_base_boolean: DataType::Boolean;
    case_base_int32: DataType::Int32;
    case_base_int64: DataType::Int64;
    case_base_float64: DataType::Float64;
    case_base_decimal: DataType::Decimal128(10, 2);
    case_base_utf8: DataType::Utf8;
    case_base_utf8_view: DataType::Utf8View;
    case_base_date32: DataType::Date32;
}

/// `needle [NOT] IN (items...)`, from operands that are the needle, then
/// the items
fn in_of(negated: bool) -> Box<Build> {
    Box::new(move |ops: Vec<Arc<dyn PhysicalExpr>>, schema: &Schema| {
        let mut ops = ops.into_iter();
        let needle = ops.next().expect("a needle");
        in_list(needle, ops.collect(), &negated, schema)
    })
}

/// Operand types for `in_of`: a needle and its items, all of type `t`. The
/// analyzer coerces the needle and items to one type
/// (`TypeCoercionRewriter` in datafusion-optimizer), so `t` must be one that
/// coercion leaves unchanged. The list lengths are the generator's own
/// length boundaries (byte, word and vector lane), not DataFusion's
fn in_signatures(t: &DataType) -> Vec<Vec<DataType>> {
    assert_eq!(
        comparison_coercion_with_session_timezone(t, t, None).as_ref(),
        Some(t),
        "{t} IN ({t}) coerces"
    );
    [1..=9, 15..=17, 31..=33, 63..=65]
        .into_iter()
        .flatten()
        .map(|k| vec![t.clone(); k + 1])
        .collect()
}

/// Defines a test of `IN` and `NOT IN` for each type
macro_rules! in_tests {
    ($($name:ident: $t:expr;)*) => {
        $(
            #[test]
            fn $name() {
                let signatures = in_signatures(&$t);
                laws(concat!(stringify!($name), " in"), &in_of(false), &signatures);
                laws(concat!(stringify!($name), " not in"), &in_of(true), &signatures);
            }
        )*
    };
}

in_tests! {
    in_boolean: DataType::Boolean;
    in_int8: DataType::Int8;
    in_uint8: DataType::UInt8;
    in_int16: DataType::Int16;
    in_uint16: DataType::UInt16;
    in_int32: DataType::Int32;
    in_uint32: DataType::UInt32;
    in_int64: DataType::Int64;
    in_uint64: DataType::UInt64;
    in_float32: DataType::Float32;
    in_float64: DataType::Float64;
    in_decimal128: DataType::Decimal128(10, 2);
    in_decimal256: DataType::Decimal256(40, 5);
    in_date32: DataType::Date32;
    in_timestamp: DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()));
    in_utf8: DataType::Utf8;
    in_large_utf8: DataType::LargeUtf8;
    in_utf8_view: DataType::Utf8View;
    in_binary: DataType::Binary;
    in_binary_view: DataType::BinaryView;
    in_fixed_size_binary: DataType::FixedSizeBinary(4);
    in_dictionary_utf8: DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
    in_dictionary_int64: DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Int64));
    in_list_of_int32: DataType::List(Arc::new(Field::new("item", DataType::Int32, true)));
    in_struct: DataType::Struct(Fields::from(vec![Field::new("a", DataType::Int32, true)]));
}

/// How `Broken` is broken
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Mode {
    /// Adds one when its operand is a literal
    LiteralOffByOne,
    /// Returns one row too many
    ExtraRow,
    /// Panics
    Panic,
    /// Reads its operand's validity bitmap from the start of the buffer,
    /// ignoring the bitmap's offset
    NullsIgnoreOffset,
    /// Adds one when the batch has 64 rows or more, as a fast path that
    /// branches on the batch would
    BatchSize,
}

/// An `Int64` expression of one `Int64` operand, broken as `mode` says
#[derive(Debug, Eq)]
struct Broken {
    operand: Arc<dyn PhysicalExpr>,
    mode: Mode,
}

// Written out because deriving them on an `Arc<dyn PhysicalExpr>` field does
// not compile (rust-lang/rust#78808), as in DataFusion's own expressions
impl PartialEq for Broken {
    fn eq(&self, other: &Self) -> bool {
        self.operand.eq(&other.operand) && self.mode == other.mode
    }
}

impl Hash for Broken {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.operand.hash(state);
        self.mode.hash(state);
    }
}

impl fmt::Display for Broken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "broken({})", self.operand)
    }
}

impl PhysicalExpr for Broken {
    fn return_field(&self, _schema: &Schema) -> Result<FieldRef> {
        Ok(Arc::new(Field::new("broken", DataType::Int64, true)))
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let n = batch.num_rows();
        let a = self.operand.evaluate(batch)?.into_array(n)?;
        let a = a.as_primitive::<Int64Type>();
        let plus_one = |a: &Int64Array| -> ArrayRef {
            Arc::new(a.unary::<_, Int64Type>(|v| v.wrapping_add(1)))
        };
        let out: ArrayRef = match self.mode {
            Mode::LiteralOffByOne if self.operand.is::<Literal>() => plus_one(a),
            Mode::BatchSize if n >= 64 => plus_one(a),
            Mode::ExtraRow => Arc::new(a.iter().chain([Some(0)]).collect::<Int64Array>()),
            Mode::Panic => panic!("broken panics"),
            Mode::NullsIgnoreOffset => {
                let nulls = a
                    .nulls()
                    .map(|b| NullBuffer::new(BooleanBuffer::new(b.buffer().clone(), 0, a.len())));
                Arc::new(Int64Array::new(a.values().clone(), nulls))
            }
            _ => Arc::new(a.clone()),
        };
        Ok(ColumnarValue::Array(out))
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.operand]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        Ok(Arc::new(Broken {
            operand: Arc::clone(&children[0]),
            mode: self.mode,
        }))
    }

    fn fmt_sql(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

type Law = fn(&Build, &ExprCase) -> Result<(), TestCaseError>;

/// Each law finds the breakages it can see in an expression broken on
/// purpose, and the failure says why. It finds none of the others
#[test]
fn finds_broken_expressions() {
    let broadcast: Law = check_expr_broadcast;
    let row_by_row: Law = check_expr_row_by_row;
    let split: Law = check_expr_split;
    let differs = Some("differs");
    let rows = Some("rows for");
    let panics = Some("panic: broken panics");
    for (name, law, mode, expected) in [
        ("broadcast", broadcast, Mode::LiteralOffByOne, differs),
        ("broadcast", broadcast, Mode::ExtraRow, rows),
        ("broadcast", broadcast, Mode::Panic, panics),
        // Both expressions read the same column, with the same wrong nulls,
        // from the same batch
        ("broadcast", broadcast, Mode::NullsIgnoreOffset, None),
        ("broadcast", broadcast, Mode::BatchSize, None),
        ("row by row", row_by_row, Mode::LiteralOffByOne, differs),
        ("row by row", row_by_row, Mode::ExtraRow, rows),
        ("row by row", row_by_row, Mode::Panic, panics),
        ("row by row", row_by_row, Mode::NullsIgnoreOffset, differs),
        ("row by row", row_by_row, Mode::BatchSize, differs),
        // A literal is a literal in every batch
        ("split", split, Mode::LiteralOffByOne, None),
        ("split", split, Mode::ExtraRow, rows),
        ("split", split, Mode::Panic, panics),
        // Concatenating copies the nulls to a bitmap with no offset
        ("split", split, Mode::NullsIgnoreOffset, differs),
        ("split", split, Mode::BatchSize, differs),
    ] {
        let build =
            move |ops: Vec<Arc<dyn PhysicalExpr>>, _: &Schema| -> Result<Arc<dyn PhysicalExpr>> {
                Ok(Arc::new(Broken {
                    operand: Arc::clone(&ops[0]),
                    mode,
                }))
            };
        let cases = exprs(&[vec![DataType::Int64]]).unwrap();
        let found = catch_unwind(AssertUnwindSafe(|| {
            check("broken", cases, |case: ExprCase| law(&build, &case));
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

/// A schema whose type the generator cannot build is an error
#[test]
fn exprs_of_an_unsupported_type() {
    assert!(exprs(&[vec![DataType::Float16]]).is_err());
    assert!(exprs(&[]).is_err());
}

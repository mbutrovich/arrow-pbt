//! Laws for DataFusion's joins, on generated streams of record batches

use std::sync::Arc;

use arrow_array::{Array, RecordBatch};
use arrow_pbt::generator::{GeneratedBatch, ValueDomain, pooled_batch, schemas, value_pool};
use arrow_pbt::val::{Val, check_rows, vals};
use arrow_schema::{DataType, Field, FieldRef, Schema, SchemaRef, SortOptions};
use arrow_select::concat::concat_batches;
use datafusion_common::{DataFusionError, JoinSide, JoinType, NullEquality, plan_err};
use datafusion_execution::{SendableRecordBatchStream, TaskContext, config::SessionConfig};
use datafusion_expr::Operator;
use datafusion_physical_expr::expressions::{BinaryExpr, Column};
use datafusion_physical_expr::{LexOrdering, PhysicalExpr, PhysicalSortExpr};
use datafusion_physical_plan::joins::utils::{ColumnIndex, JoinFilter};
use datafusion_physical_plan::joins::{
    HashJoinExec, NestedLoopJoinExec, PartitionMode, SortMergeJoinExec,
};
use datafusion_physical_plan::sorts::sort::sort_batch;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::streaming::{PartitionStream, StreamingTableExec};
use datafusion_physical_plan::{ExecutionPlan, collect};
use proptest::collection::vec;
use proptest::prelude::*;
use proptest::sample::select;
use proptest::test_runner::TestCaseError;

use crate::{ArgType, fail};

/// Builds one way to compute a case's join from its two inputs, the left
/// one first
pub type JoinPlan = dyn Fn(
        &JoinCase,
        Arc<dyn ExecutionPlan>,
        Arc<dyn ExecutionPlan>,
    ) -> datafusion_common::Result<Arc<dyn ExecutionPlan>>
    + Sync;

/// A join of two streams of batches. Each side's schema has the key
/// columns first, in the same order and with the same fields on both sides,
/// then columns of any types, then an `Int32` column `t`. The residual
/// filter, when there is one, is `left.t < right.t`
#[derive(Clone, Debug)]
pub struct JoinCase {
    pub left_schema: SchemaRef,
    pub right_schema: SchemaRef,
    pub left: Vec<GeneratedBatch>,
    pub right: Vec<GeneratedBatch>,
    /// The number of key columns
    pub keys: usize,
    pub join_type: JoinType,
    pub null_equality: NullEquality,
    pub filter: bool,
    /// How a hash join collects its build side. Each side is one partition
    pub partition_mode: PartitionMode,
    pub batch_size: usize,
    /// The sort options of each key when both inputs are sorted on their
    /// keys (see [`sorted_source`]), or `None` when the inputs keep the order
    /// they were drawn in
    pub sort_options: Option<Vec<SortOptions>>,
}

/// Joins on one or two keys of the types in `key_types`, of every join
/// type, with both null equalities, with and without a residual filter, and
/// a drawn `batch_size`. Each key field is nullable three times in four, and
/// both sides draw its rows from one [`value_pool`], so rows match and one
/// probe row can match more build rows than `batch_size`. Each side is zero
/// to four batches of up to 129 rows, which bounds the output, since it
/// grows with the product of the two sides' rows for each key. Values come
/// from `ValueDomain::Logical`, since DataFusion treats a decimal past its
/// declared precision as invalid (apache/datafusion#26774)
///
/// # Errors
///
/// No key types, or a type the generator cannot build
///
/// # Example
///
/// ```
/// use arrow_pbt_datafusion::joins;
/// use arrow_schema::DataType;
/// use proptest::strategy::{Strategy, ValueTree};
/// use proptest::test_runner::TestRunner;
/// let mut runner = TestRunner::deterministic();
/// let case = joins(&[DataType::Int32]).unwrap().new_tree(&mut runner).unwrap().current();
/// assert!((1..=2).contains(&case.keys));
/// assert_eq!(case.left_schema.field(0), case.right_schema.field(0));
/// assert!(joins(&[DataType::Float16]).is_err());
/// ```
pub fn joins(key_types: &[DataType]) -> Result<BoxedStrategy<JoinCase>, String> {
    if key_types.is_empty() {
        return Err("no key types".to_string());
    }
    let key_types: Vec<ArgType> = key_types
        .iter()
        .map(ArgType::new)
        .collect::<Result<_, _>>()?;
    let key = select(key_types)
        .prop_flat_map(ArgType::field)
        .prop_flat_map(|(f, t)| (Just(f), value_pool(&t)));
    let join_type = select(vec![
        JoinType::Inner,
        JoinType::Left,
        JoinType::Right,
        JoinType::Full,
        JoinType::LeftSemi,
        JoinType::RightSemi,
        JoinType::LeftAnti,
        JoinType::RightAnti,
        JoinType::LeftMark,
        JoinType::RightMark,
    ]);
    let null_equality = select(vec![
        NullEquality::NullEqualsNothing,
        NullEquality::NullEqualsNull,
    ]);
    let partition_mode = select(vec![PartitionMode::CollectLeft, PartitionMode::Partitioned]);
    let batch_size = prop_oneof![1..=8usize, select(vec![16usize, 63, 64, 65, 1024, 8192])];
    Ok((vec(key, 1..=2), schemas(), schemas())
        .prop_flat_map(|(keys, left, right)| {
            let n = keys.len();
            let key_fields: Vec<FieldRef> = keys
                .iter()
                .enumerate()
                .map(|(i, (f, _))| Arc::new(f.as_ref().clone().with_name(format!("k{i}"))))
                .collect();
            let pools: Vec<_> = keys.into_iter().map(|(_, p)| p).collect();
            let left = side(&key_fields, &left);
            let right = side(&key_fields, &right);
            (
                Just(Arc::clone(&left)),
                Just(Arc::clone(&right)),
                stream(&left, &pools),
                stream(&right, &pools),
                Just(n),
            )
        })
        .prop_flat_map(move |(left_schema, right_schema, left, right, keys)| {
            (
                Just((left_schema, right_schema, left, right, keys)),
                join_type.clone(),
                null_equality.clone(),
                any::<bool>(),
                partition_mode.clone(),
                batch_size.clone(),
            )
        })
        .prop_map(
            |(
                (left_schema, right_schema, left, right, keys),
                join_type,
                null_equality,
                filter,
                partition_mode,
                batch_size,
            )| JoinCase {
                left_schema,
                right_schema,
                left,
                right,
                keys,
                join_type,
                null_equality,
                filter,
                partition_mode,
                batch_size,
                sort_options: None,
            },
        )
        .boxed())
}

/// The cases of [`joins`] with both inputs sorted on their keys, as a
/// sort-merge join needs. `descending` and `nulls_first` are drawn for each
/// key, since the join compares keys in the order they give
///
/// # Errors
///
/// As [`joins`]
///
/// # Example
///
/// ```
/// use arrow_pbt_datafusion::sort_merge_joins;
/// use arrow_schema::DataType;
/// use proptest::strategy::{Strategy, ValueTree};
/// use proptest::test_runner::TestRunner;
/// let mut runner = TestRunner::deterministic();
/// let case = sort_merge_joins(&[DataType::Int32]).unwrap().new_tree(&mut runner).unwrap().current();
/// assert_eq!(case.sort_options.unwrap().len(), case.keys);
/// assert!(sort_merge_joins(&[DataType::Float16]).is_err());
/// ```
pub fn sort_merge_joins(key_types: &[DataType]) -> Result<BoxedStrategy<JoinCase>, String> {
    let options = (any::<bool>(), any::<bool>())
        .prop_map(|(descending, nulls_first)| SortOptions::new(descending, nulls_first));
    Ok(joins(key_types)?
        .prop_flat_map(move |case| {
            let keys = case.keys;
            (Just(case), vec(options.clone(), keys))
        })
        .prop_map(|(case, options)| JoinCase {
            sort_options: Some(options),
            ..case
        })
        .boxed())
}

/// One side's schema: the key fields, then the fields of `others` renamed
/// from `p0`, then `t`
fn side(keys: &[FieldRef], others: &Schema) -> SchemaRef {
    let others = others
        .fields()
        .iter()
        .enumerate()
        .map(|(i, f)| Arc::new(f.as_ref().clone().with_name(format!("p{i}"))));
    let t = Arc::new(Field::new("t", DataType::Int32, true));
    Arc::new(Schema::new(
        keys.iter()
            .cloned()
            .chain(others)
            .chain([t])
            .collect::<Vec<_>>(),
    ))
}

/// Zero to four batches of `schema`, none one time in ten, with the key
/// columns drawn from `pools`
fn stream(schema: &SchemaRef, pools: &[Vec<Val>]) -> BoxedStrategy<Vec<GeneratedBatch>> {
    let schema = Arc::clone(schema);
    let pools: Vec<Option<Vec<Val>>> = (0..schema.fields().len())
        .map(|i| pools.get(i).cloned())
        .collect();
    let lens = prop_oneof![
        4 => 0..=8usize,
        3 => 0..=64usize,
        1 => select(vec![127usize, 128, 129]),
    ];
    let one = lens
        .prop_flat_map(move |n| {
            // Every field converted when the case was drawn, there is a pool
            // or `None` for each, a key pool holds a null only when its field
            // is nullable, and `value_pool` is never empty
            pooled_batch(&schema, &pools, n, ValueDomain::Logical).expect("valid pools")
        })
        .boxed();
    prop_oneof![1 => Just(0usize), 9 => 1..=4usize]
        .prop_flat_map(move |k| vec(one.clone(), k))
        .boxed()
}

/// A hash join of `case`, on its keys, with its residual filter, null
/// equality, and partition mode
///
/// # Example
///
/// ```
/// use arrow_pbt_datafusion::{hash_join, joins, source};
/// use arrow_schema::DataType;
/// use datafusion_physical_plan::ExecutionPlan;
/// use proptest::strategy::{Strategy, ValueTree};
/// use proptest::test_runner::TestRunner;
/// let mut runner = TestRunner::deterministic();
/// let case = joins(&[DataType::Utf8]).unwrap().new_tree(&mut runner).unwrap().current();
/// let left = source(&case.left_schema, &case.left).unwrap();
/// let right = source(&case.right_schema, &case.right).unwrap();
/// assert_eq!(hash_join(&case, left, right).unwrap().name(), "HashJoinExec");
/// ```
pub fn hash_join(
    case: &JoinCase,
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
) -> datafusion_common::Result<Arc<dyn ExecutionPlan>> {
    let on = (0..case.keys)
        .map(|i| (column(&case.left_schema, i), column(&case.right_schema, i)))
        .collect();
    Ok(Arc::new(HashJoinExec::try_new(
        left,
        right,
        on,
        join_filter(case, false),
        &case.join_type,
        None,
        case.partition_mode,
        case.null_equality,
        false,
    )?))
}

/// A nested loop join of `case`, whose filter is the residual filter and an
/// equality for each key: `=` when nulls equal nothing, and
/// `IS NOT DISTINCT FROM` when they equal each other
///
/// # Example
///
/// ```
/// use arrow_pbt_datafusion::{joins, nested_loop_join, source};
/// use arrow_schema::DataType;
/// use datafusion_physical_plan::ExecutionPlan;
/// use proptest::strategy::{Strategy, ValueTree};
/// use proptest::test_runner::TestRunner;
/// let mut runner = TestRunner::deterministic();
/// let case = joins(&[DataType::Int64]).unwrap().new_tree(&mut runner).unwrap().current();
/// let left = source(&case.left_schema, &case.left).unwrap();
/// let right = source(&case.right_schema, &case.right).unwrap();
/// let join = nested_loop_join(&case, left, right).unwrap();
/// assert_eq!(join.name(), "NestedLoopJoinExec");
/// ```
pub fn nested_loop_join(
    case: &JoinCase,
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
) -> datafusion_common::Result<Arc<dyn ExecutionPlan>> {
    Ok(Arc::new(NestedLoopJoinExec::try_new(
        left,
        right,
        join_filter(case, true),
        &case.join_type,
        None,
    )?))
}

/// A sort-merge join of `case`, on its keys, with its residual filter, null
/// equality, and sort options
///
/// # Errors
///
/// A case without sort options, whose inputs are not sorted, or one that
/// `SortMergeJoinExec::try_new` rejects
///
/// # Example
///
/// ```
/// use arrow_pbt_datafusion::{sort_merge_join, sort_merge_joins, sorted_source};
/// use arrow_schema::DataType;
/// use datafusion_physical_plan::ExecutionPlan;
/// use proptest::strategy::{Strategy, ValueTree};
/// use proptest::test_runner::TestRunner;
/// let mut runner = TestRunner::deterministic();
/// let case = sort_merge_joins(&[DataType::Int64]).unwrap().new_tree(&mut runner).unwrap().current();
/// let options = case.sort_options.clone().unwrap();
/// let left = sorted_source(&case.left_schema, &case.left, &options).unwrap();
/// let right = sorted_source(&case.right_schema, &case.right, &options).unwrap();
/// let join = sort_merge_join(&case, left.clone(), right.clone()).unwrap();
/// assert_eq!(join.name(), "SortMergeJoinExec");
/// let unsorted = arrow_pbt_datafusion::JoinCase { sort_options: None, ..case.clone() };
/// assert!(sort_merge_join(&unsorted, left, right).is_err());
/// ```
pub fn sort_merge_join(
    case: &JoinCase,
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
) -> datafusion_common::Result<Arc<dyn ExecutionPlan>> {
    let Some(sort_options) = &case.sort_options else {
        return plan_err!("a sort-merge join needs a case whose inputs are sorted");
    };
    let on = (0..case.keys)
        .map(|i| (column(&case.left_schema, i), column(&case.right_schema, i)))
        .collect();
    Ok(Arc::new(SortMergeJoinExec::try_new(
        left,
        right,
        on,
        join_filter(case, false),
        case.join_type,
        sort_options.clone(),
        case.null_equality,
    )?))
}

fn column(schema: &Schema, i: usize) -> Arc<dyn PhysicalExpr> {
    Arc::new(Column::new(schema.field(i).name(), i))
}

fn binary(
    l: Arc<dyn PhysicalExpr>,
    op: Operator,
    r: Arc<dyn PhysicalExpr>,
) -> Arc<dyn PhysicalExpr> {
    Arc::new(BinaryExpr::new(l, op, r))
}

/// The case's residual filter, and with `keys` an equality for each key
/// too, or `None` when that leaves no condition
fn join_filter(case: &JoinCase, keys: bool) -> Option<JoinFilter> {
    let (ls, rs) = (&case.left_schema, &case.right_schema);
    // Pairs of a left and a right column, with the operator between them
    let mut pairs = vec![];
    if case.filter {
        pairs.push((ls.fields().len() - 1, Operator::Lt, rs.fields().len() - 1));
    }
    if keys {
        let eq = match case.null_equality {
            NullEquality::NullEqualsNothing => Operator::Eq,
            NullEquality::NullEqualsNull => Operator::IsNotDistinctFrom,
        };
        pairs.extend((0..case.keys).map(|i| (i, eq, i)));
    }
    let mut fields = vec![];
    let mut indices = vec![];
    let mut operand = |side, schema: &Schema, index: usize| {
        let f = schema.field(index);
        let at = fields.len();
        fields.push(f.clone().with_nullable(true));
        indices.push(ColumnIndex { index, side });
        Arc::new(Column::new(f.name(), at)) as Arc<dyn PhysicalExpr>
    };
    let expression = pairs
        .into_iter()
        .map(|(l, op, r)| {
            let l = operand(JoinSide::Left, ls, l);
            binary(l, op, operand(JoinSide::Right, rs, r))
        })
        .collect::<Vec<_>>()
        .into_iter()
        .reduce(|a, b| binary(a, Operator::And, b))?;
    Some(JoinFilter::new(
        expression,
        indices,
        Arc::new(Schema::new(fields)),
    ))
}

/// One partition that yields a case's batches
#[derive(Debug)]
struct Batches {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
}

impl PartitionStream for Batches {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn execute(&self, _: Arc<TaskContext>) -> SendableRecordBatchStream {
        let batches = self.batches.clone().into_iter().map(Ok);
        Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            futures::stream::iter(batches),
        ))
    }
}

/// An input of one partition that yields `batches`
///
/// # Example
///
/// ```
/// use arrow_pbt_datafusion::{joins, source};
/// use arrow_schema::DataType;
/// use datafusion_physical_plan::ExecutionPlan;
/// use proptest::strategy::{Strategy, ValueTree};
/// use proptest::test_runner::TestRunner;
/// let mut runner = TestRunner::deterministic();
/// let case = joins(&[DataType::Int32]).unwrap().new_tree(&mut runner).unwrap().current();
/// let left = source(&case.left_schema, &case.left).unwrap();
/// assert_eq!(left.schema(), case.left_schema);
/// ```
pub fn source(
    schema: &SchemaRef,
    batches: &[GeneratedBatch],
) -> datafusion_common::Result<Arc<dyn ExecutionPlan>> {
    table(schema, batches.iter().map(GeneratedBatch::build).collect())
}

/// An input of one partition that yields the rows of `batches` sorted on
/// their first `options.len()` columns, the first by `options[0]` and so on,
/// by DataFusion's `sort_batch`. Each batch keeps its number of rows, so the
/// sorted rows split where the drawn ones did
///
/// # Errors
///
/// No options, more options than columns, or a failure to concatenate or
/// sort the rows
///
/// # Example
///
/// ```
/// use arrow_pbt_datafusion::{joins, sorted_source};
/// use arrow_schema::{DataType, SortOptions};
/// use proptest::strategy::{Strategy, ValueTree};
/// use proptest::test_runner::TestRunner;
/// let mut runner = TestRunner::deterministic();
/// let case = joins(&[DataType::Int32]).unwrap().new_tree(&mut runner).unwrap().current();
/// let options = vec![SortOptions::default(); case.keys];
/// let left = sorted_source(&case.left_schema, &case.left, &options).unwrap();
/// assert_eq!(left.schema(), case.left_schema);
/// assert!(sorted_source(&case.left_schema, &case.left, &[]).is_err());
/// ```
pub fn sorted_source(
    schema: &SchemaRef,
    batches: &[GeneratedBatch],
    options: &[SortOptions],
) -> datafusion_common::Result<Arc<dyn ExecutionPlan>> {
    if options.len() > schema.fields().len() {
        return plan_err!(
            "{} sort keys for {} columns",
            options.len(),
            schema.fields().len()
        );
    }
    let Some(ordering) = LexOrdering::new(
        options
            .iter()
            .enumerate()
            .map(|(i, &o)| PhysicalSortExpr::new(column(schema, i), o)),
    ) else {
        return plan_err!("a sorted input needs at least one sort key");
    };
    let built: Vec<RecordBatch> = batches.iter().map(GeneratedBatch::build).collect();
    let sorted = sort_batch(&concat_batches(schema, &built)?, &ordering, None)?;
    let mut offset = 0;
    let split = built
        .iter()
        .map(|b| {
            let slice = sorted.slice(offset, b.num_rows());
            offset += b.num_rows();
            slice
        })
        .collect();
    table(schema, split)
}

/// A one-partition `StreamingTableExec` that yields `batches`
fn table(
    schema: &SchemaRef,
    batches: Vec<RecordBatch>,
) -> datafusion_common::Result<Arc<dyn ExecutionPlan>> {
    let partition = Batches {
        schema: Arc::clone(schema),
        batches,
    };
    Ok(Arc::new(StreamingTableExec::try_new(
        Arc::clone(schema),
        vec![Arc::new(partition)],
        None,
        vec![],
        false,
        None,
    )?))
}

/// Checks the agreement law: the joins `a` and `b` of `case` both fail, or
/// both return valid output of the same types with the same rows, in any
/// order. They run with the case's `batch_size` and the default config
/// otherwise
///
/// # Errors
///
/// A failure that says how the two differ. A panic propagates, and
/// `arrow_pbt::run::check` reports it as a failure
///
/// # Example
///
/// ```
/// use arrow_pbt_datafusion::{check_join_agreement, hash_join, joins, nested_loop_join};
/// use arrow_schema::DataType;
/// use proptest::strategy::{Strategy, ValueTree};
/// use proptest::test_runner::TestRunner;
/// let mut runner = TestRunner::deterministic();
/// let case = joins(&[DataType::Int32]).unwrap().new_tree(&mut runner).unwrap().current();
/// // A join agrees with itself on every case
/// assert!(check_join_agreement(&case, &hash_join, &hash_join).is_ok());
/// // A test runs this through `arrow_pbt::run::check`, with its known-bug
/// // filters, since other columns of any types reach open arrow-rs bugs
/// let _ = check_join_agreement(&case, &hash_join, &nested_loop_join);
/// ```
pub fn check_join_agreement(
    case: &JoinCase,
    a: &JoinPlan,
    b: &JoinPlan,
) -> Result<(), TestCaseError> {
    compare(
        ("the first join", outcome(case, a, &[])?),
        ("the second join", outcome(case, b, &[])?),
    )
}

/// Checks the config law: the join `plan` of `case` gives the same result as
/// in [`check_join_agreement`] with the case's config and with each
/// `(key, value)` of `options` set on it
///
/// # Errors
///
/// A failure that says how the two differ, or an option DataFusion rejects
///
/// # Example
///
/// ```
/// use arrow_pbt_datafusion::{check_join_config, hash_join, joins};
/// use arrow_schema::DataType;
/// use proptest::strategy::{Strategy, ValueTree};
/// use proptest::test_runner::TestRunner;
/// let mut runner = TestRunner::deterministic();
/// let case = joins(&[DataType::Int8]).unwrap().new_tree(&mut runner).unwrap().current();
/// let options = [("datafusion.execution.perfect_hash_join_small_build_threshold", "0")];
/// assert!(check_join_config(&case, &hash_join, &options).is_ok());
/// assert!(check_join_config(&case, &hash_join, &[("datafusion.no_such_option", "1")]).is_err());
/// ```
pub fn check_join_config(
    case: &JoinCase,
    plan: &JoinPlan,
    options: &[(&str, &str)],
) -> Result<(), TestCaseError> {
    compare(
        ("the case's config", outcome(case, plan, &[])?),
        ("the config with the options", outcome(case, plan, options)?),
    )
}

/// A join's output: its schema and its rows, each as the oracle's values
/// in `Debug` form, sorted, so that equal multisets compare equal
type Output = (SchemaRef, Vec<String>);

/// Builds and runs the join `plan` of `case` with `options` set
fn outcome(
    case: &JoinCase,
    plan: &JoinPlan,
    options: &[(&str, &str)],
) -> Result<Result<Output, DataFusionError>, TestCaseError> {
    let mut config = SessionConfig::new().with_batch_size(case.batch_size);
    for (key, value) in options {
        config
            .options_mut()
            .set(key, value)
            .map_err(|e| fail(format!("setting {key} to {value}: {e}")))?;
    }
    let context = Arc::new(TaskContext::default().with_session_config(config));
    // Inputs that cannot be built are a failure of the case, not of a join,
    // so that a known bug in building them is counted rather than passing
    // as two joins that both fail
    let input = |schema, batches| match &case.sort_options {
        None => source(schema, batches),
        Some(options) => sorted_source(schema, batches, options),
    };
    let (left, right) = input(&case.left_schema, &case.left)
        .and_then(|l| Ok((l, input(&case.right_schema, &case.right)?)))
        .map_err(|e| fail(format!("building the inputs: {e}")))?;
    let run = || -> datafusion_common::Result<(SchemaRef, Vec<RecordBatch>)> {
        let join = plan(case, left, right)?;
        let schema = join.schema();
        // A current-thread runtime keeps DataFusion's tasks on this thread,
        // off the other workers of the seed pool
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        Ok((schema, runtime.block_on(collect(join, context))?))
    };
    match run() {
        Err(e) => Ok(Err(e)),
        Ok((schema, batches)) => Ok(Ok((Arc::clone(&schema), rows(&schema, &batches)?))),
    }
}

/// The rows of `batches`, checked to be valid output of `schema`
fn rows(schema: &SchemaRef, batches: &[RecordBatch]) -> Result<Vec<String>, TestCaseError> {
    let mut out = vec![];
    for b in batches {
        let mut columns = vec![];
        for (f, c) in schema.fields().iter().zip(b.columns()) {
            if c.data_type() != f.data_type() {
                return Err(fail(format!(
                    "column {} is {} and its field promises {}",
                    f.name(),
                    c.data_type(),
                    f.data_type()
                )));
            }
            if !f.is_nullable() && c.logical_null_count() > 0 {
                return Err(fail(format!(
                    "column {} holds nulls under a non-nullable field",
                    f.name()
                )));
            }
            let v = vals(c.as_ref());
            check_rows(c.as_ref(), f.data_type(), &v)?;
            columns.push(v);
        }
        out.extend(
            (0..b.num_rows())
                .map(|i| format!("{:?}", columns.iter().map(|c| &c[i]).collect::<Vec<_>>())),
        );
    }
    out.sort_unstable();
    Ok(out)
}

/// Fails unless both outcomes failed, or both have the same field types and
/// rows
fn compare(
    (a_name, a): (&str, Result<Output, DataFusionError>),
    (b_name, b): (&str, Result<Output, DataFusionError>),
) -> Result<(), TestCaseError> {
    let ((a_schema, a), (b_schema, b)) = match (a, b) {
        (Err(_), Err(_)) => return Ok(()),
        (Err(e), Ok(_)) => return Err(fail(format!("{a_name} fails and {b_name} succeeds: {e}"))),
        (Ok(_), Err(e)) => return Err(fail(format!("{b_name} fails and {a_name} succeeds: {e}"))),
        (Ok(a), Ok(b)) => (a, b),
    };
    let types = |s: &Schema| {
        s.fields()
            .iter()
            .map(|f| f.data_type().clone())
            .collect::<Vec<_>>()
    };
    if types(&a_schema) != types(&b_schema) {
        return Err(fail(format!(
            "the output types differ: {a_name} gives {:?} and {b_name} gives {:?}",
            types(&a_schema),
            types(&b_schema)
        )));
    }
    if a != b {
        let (only_a, only_b) = difference(&a, &b);
        return Err(fail(format!(
            "the rows differ: {a_name} gives {} rows and {b_name} {}. Only from {a_name}, first 3: {:?}. Only from {b_name}, first 3: {:?}",
            a.len(),
            b.len(),
            &only_a[..only_a.len().min(3)],
            &only_b[..only_b.len().min(3)],
        )));
    }
    Ok(())
}

/// The rows of sorted `a` missing from sorted `b`, and the reverse, counting
/// repeats
fn difference<'a>(a: &'a [String], b: &'a [String]) -> (Vec<&'a str>, Vec<&'a str>) {
    let (mut i, mut j) = (0, 0);
    let (mut only_a, mut only_b) = (vec![], vec![]);
    while i < a.len() || j < b.len() {
        match (a.get(i), b.get(j)) {
            (Some(x), Some(y)) if x == y => (i, j) = (i + 1, j + 1),
            (Some(x), y) if y.is_none_or(|y| x < y) => {
                only_a.push(x.as_str());
                i += 1;
            }
            (_, Some(y)) => {
                only_b.push(y.as_str());
                j += 1;
            }
            (_, None) => unreachable!("the loop condition and the arm above"),
        }
    }
    (only_a, only_b)
}

//! Properties for kernels over record batches: `concat_batches`,
//! `BatchCoalescer`, `take_record_batch` and `interleave_record_batch`

use arrow_array::{Array, ArrayRef, BooleanArray, Int32Array, RecordBatch, RecordBatchOptions};
use arrow_schema::SchemaRef;
use arrow_select::coalesce::BatchCoalescer;
use arrow_select::filter::FilterBuilder;
use proptest::test_runner::TestCaseError;

use crate::props::R;
use crate::props::can_hold_null;
use crate::props::{check_in, fail, known_panic};
use crate::run::{caught, panic_message};
use crate::types::{any_type, is_view_dict};
use crate::val::check_rows;
use crate::val::{Val, vals};
use arrow_schema::DataType;
use std::panic::{AssertUnwindSafe, catch_unwind};

/// How a batch is pushed into the coalescer
#[derive(Debug, Clone)]
pub enum Push {
    Plain,
    Filter(BooleanArray),
    /// Through a `FilterBuilder` that is optimized first
    FilterOptimized(BooleanArray),
    /// Through a `FilterBuilder` given the number of selected rows
    FilterCounted(BooleanArray),
    Indices(Int32Array),
    /// `(batch, row)` pairs into every batch of the case, through
    /// `push_batch_interleaved`
    Interleaved(Vec<(usize, usize)>),
}

/// A schema, batches of it with how each is pushed, the target batch size,
/// and the size above which batches bypass coalescing
pub type CoalesceCase = (SchemaRef, Vec<(RecordBatch, Push)>, usize, Option<usize>);

/// Batches of one schema, and `(batch, row)` pairs into them
pub type InterleaveBatches = (Vec<RecordBatch>, Vec<(usize, usize)>);

fn selected(m: &BooleanArray, i: usize) -> bool {
    m.is_valid(i) && m.value(i)
}

/// The rows each push contributes, per column, as `filter_record_batch`,
/// `take_record_batch` or `interleave_record_batch` of `all` would give them
fn pushed_rows(batch: &RecordBatch, push: &Push, all: &[RecordBatch]) -> Vec<Vec<Val>> {
    (0..batch.num_columns())
        .map(|c| {
            let v = || vals(batch.column(c).as_ref());
            match push {
                Push::Plain => v(),
                Push::Filter(m) | Push::FilterOptimized(m) | Push::FilterCounted(m) => {
                    let v = v();
                    (0..v.len())
                        .filter(|&i| selected(m, i))
                        .map(|i| v[i].clone())
                        .collect()
                }
                Push::Indices(idx) => {
                    let v = v();
                    (0..idx.len())
                        .map(|i| {
                            if idx.is_valid(i) {
                                v[idx.value(i) as usize].clone()
                            } else {
                                Val::Null
                            }
                        })
                        .collect()
                }
                Push::Interleaved(pairs) => {
                    let vs: Vec<Vec<Val>> =
                        all.iter().map(|b| vals(b.column(c).as_ref())).collect();
                    pairs.iter().map(|&(b, r)| vs[b][r].clone()).collect()
                }
            }
        })
        .collect()
}

/// Checks that `out` holds `expected` rows per column, in order, as valid
/// arrays of the schema's types
fn check_concatenated(schema: &SchemaRef, out: &[RecordBatch], expected: &[Vec<Val>]) -> R {
    for b in out {
        if b.schema() != *schema {
            return Err(fail(format!("output schema {:?}", b.schema())));
        }
    }
    for (c, f) in schema.fields().iter().enumerate() {
        let mut at = 0;
        for (k, b) in out.iter().enumerate() {
            let n = b.num_rows();
            let want = expected[c].get(at..at + n).ok_or_else(|| {
                fail(format!(
                    "{} rows out, more than {}",
                    at + n,
                    expected[c].len()
                ))
            })?;
            check_rows(b.column(c), f.data_type(), want)
                .map_err(|e| fail(format!("batch {k} column {c}: {e:?}")))?;
            at += n;
        }
        if at != expected[c].len() {
            return Err(fail(format!(
                "{at} rows out, expected {}",
                expected[c].len()
            )));
        }
    }
    Ok(())
}

/// Known failure, by signature: dictionaries of views are not merged, so
/// their keys overflow (apache/arrow-rs#10925)
fn known(r: R, schema: &SchemaRef, indices: bool) -> R {
    let any =
        |f: &dyn Fn(&DataType) -> bool| schema.fields().iter().any(|x| any_type(x.data_type(), f));
    let view_dict = any(&is_view_dict);
    let no_null = schema
        .fields()
        .iter()
        .any(|f| !can_hold_null(f.data_type()));
    match r {
        // Correct: a null index into a column that cannot hold a null
        Err(TestCaseError::Fail(m))
            if indices && no_null && m.to_string().contains("Cannot take null indices") =>
        {
            Err(TestCaseError::reject(
                "outside the domain: a null index into a column that cannot hold a null",
            ))
        }
        Err(TestCaseError::Fail(m))
            if view_dict
                && m.to_string()
                    .contains("Dictionary key bigger than the key type") =>
        {
            Err(crate::known::known(
                "known: dictionaries of views are not merged, so keys overflow (apache/arrow-rs#10925)",
            ))
        }
        r => r,
    }
}

/// `concat_batches` returns every batch's rows in order
pub fn concat_batches_prop(case: CoalesceCase) -> R {
    let schema = case.0.clone();
    known(caught(|| concat_batches_body(case)), &schema, false)
}

fn concat_batches_body((schema, batches, _, _): CoalesceCase) -> R {
    for (b, _) in &batches {
        for c in b.columns() {
            check_in(c.as_ref())?;
        }
    }
    let mut expected: Vec<Vec<Val>> = vec![vec![]; schema.fields().len()];
    for (b, _) in &batches {
        for (c, rows) in pushed_rows(b, &Push::Plain, &[]).into_iter().enumerate() {
            expected[c].extend(rows);
        }
    }
    let out = arrow_select::concat::concat_batches(&schema, batches.iter().map(|(b, _)| b))
        .map_err(|e| fail(format!("concat_batches: {e}")))?;
    check_concatenated(&schema, &[out], &expected)
}

/// `out` is a valid batch of `schema` with `n` rows, whose column `c` holds
/// `expected(c)`
fn check_batch(
    out: &RecordBatch,
    schema: &SchemaRef,
    n: usize,
    expected: impl Fn(usize) -> Vec<Val>,
) -> R {
    if out.schema() != *schema {
        return Err(fail(format!("schema {:?} != {schema:?}", out.schema())));
    }
    if out.num_rows() != n {
        return Err(fail(format!("{} rows != {n}", out.num_rows())));
    }
    // The checks `RecordBatch::try_new` makes, such as no null in a column
    // whose field is not nullable
    let options = RecordBatchOptions::new().with_row_count(Some(n));
    RecordBatch::try_new_with_options(schema.clone(), out.columns().to_vec(), &options)
        .map_err(|e| fail(format!("invalid batch: {e}")))?;
    for (c, f) in schema.fields().iter().enumerate() {
        check_rows(out.column(c).as_ref(), f.data_type(), &expected(c))
            .map_err(|e| fail(format!("column {c}: {e:?}")))?;
    }
    Ok(())
}

/// `take_record_batch` takes the same rows from every column, as `take`
/// does. A valid index out of bounds is an error, and so is a null index
/// into a column whose field or type cannot hold a null
pub fn take_record_batch_prop(case: (RecordBatch, Int32Array)) -> R {
    let schema = case.0.schema();
    known(caught(|| take_record_batch_body(case)), &schema, false)
}

fn take_record_batch_body((batch, indices): (RecordBatch, Int32Array)) -> R {
    use arrow_select::take::take_record_batch;
    batch
        .columns()
        .iter()
        .try_for_each(|c| check_in(c.as_ref()))?;
    check_in(&indices)?;
    let (schema, n) = (batch.schema(), batch.num_rows() as i64);
    // An index out of bounds is a panic, as `take` documents when bounds are
    // not checked, or an error
    let out_of_bounds = (0..indices.len())
        .find(|&i| indices.is_valid(i) && !(0..n).contains(&(indices.value(i) as i64)));
    if let Some(i) = out_of_bounds {
        return match catch_unwind(AssertUnwindSafe(|| take_record_batch(&batch, &indices))) {
            Err(_) | Ok(Err(_)) => Ok(()),
            Ok(Ok(out)) => Err(fail(format!(
                "take_record_batch returned {} rows for index {} of {n} rows",
                out.num_rows(),
                indices.value(i)
            ))),
        };
    }
    let r = take_record_batch(&batch, &indices);
    let no_null = indices.null_count() > 0
        && schema
            .fields()
            .iter()
            .any(|f| !f.is_nullable() || !can_hold_null(f.data_type()));
    if no_null {
        // Known: `RecordBatch::try_new` checks a non-nullable field with
        // `null_count`, so it accepts a column whose nulls are only logical
        // (apache/arrow-rs#11832)
        let logical_only = |out: &RecordBatch| {
            out.columns()
                .iter()
                .zip(schema.fields())
                .any(|(c, f)| !f.is_nullable() && c.null_count() == 0 && c.logical_null_count() > 0)
        };
        return match r {
            Err(_) => Ok(()),
            Ok(out) if logical_only(&out) => Err(crate::known::known(
                "known: RecordBatch::try_new accepts logical nulls in a non-nullable column (apache/arrow-rs#11832)",
            )),
            Ok(out) => Err(fail(format!(
                "take_record_batch returned {} rows for a null index into a column that cannot hold a null",
                out.num_rows()
            ))),
        };
    }
    let out = r.map_err(|e| fail(format!("take_record_batch: {e}")))?;
    check_batch(&out, &schema, indices.len(), |c| {
        let v = vals(batch.column(c).as_ref());
        (0..indices.len())
            .map(|i| match indices.is_valid(i) {
                true => v[indices.value(i) as usize].clone(),
                false => Val::Null,
            })
            .collect()
    })
}

/// `interleave_record_batch` takes row `r` of batch `b` for each `(b, r)`
/// from every column, as `interleave` does. No batches is an error, as it
/// is for `interleave`
pub fn interleave_record_batch_prop(case: InterleaveBatches) -> R {
    let Some(schema) = case.0.first().map(RecordBatch::schema) else {
        return caught(|| interleave_record_batch_body(case));
    };
    known(
        caught(|| interleave_record_batch_body(case)),
        &schema,
        false,
    )
}

fn interleave_record_batch_body((batches, indices): InterleaveBatches) -> R {
    use arrow_select::interleave::interleave_record_batch;
    for b in &batches {
        b.columns().iter().try_for_each(|c| check_in(c.as_ref()))?;
    }
    let refs: Vec<&RecordBatch> = batches.iter().collect();
    let r = catch_unwind(AssertUnwindSafe(|| {
        interleave_record_batch(&refs, &indices)
    }));
    let Some(first) = batches.first() else {
        return match r {
            Ok(Err(_)) => Ok(()),
            Ok(Ok(_)) => Err(fail(
                "interleave_record_batch of no batches returned a batch",
            )),
            Err(p) => Err(fail(format!(
                "interleave_record_batch of no batches panicked: {}",
                panic_message(&*p)
            ))),
        };
    };
    let schema = first.schema();
    // A struct of the columns, so that a known bug in any column matches
    let columns = DataType::Struct(schema.fields().clone());
    let r = r.map_err(|p| known_panic(&columns, &panic_message(&*p)))?;
    let out = r.map_err(|e| fail(format!("interleave_record_batch: {e}")))?;
    check_batch(&out, &schema, indices.len(), |c| {
        let vs: Vec<Vec<Val>> = batches.iter().map(|b| vals(b.column(c).as_ref())).collect();
        indices.iter().map(|&(b, r)| vs[b][r].clone()).collect()
    })
}

/// The coalescer emits every pushed row once, in order, as filter and take
/// would select them. Without a bypass size, every batch but the last has
/// exactly the target size, as documented
pub fn coalesce_prop(case: CoalesceCase) -> R {
    let schema = case.0.clone();
    let indices = case.1.iter().any(|(_, p)| matches!(p, Push::Indices(_)));
    known(caught(|| coalesce_body(case)), &schema, indices)
}

fn coalesce_body((schema, batches, target, biggest): CoalesceCase) -> R {
    for (b, push) in &batches {
        for c in b.columns() {
            check_in(c.as_ref())?;
        }
        if let Push::Filter(m) | Push::FilterOptimized(m) | Push::FilterCounted(m) = push {
            check_in(m)?;
        }
        if let Push::Indices(i) = push {
            check_in(i)?;
        }
    }
    // The bypass size through the builder for even targets and the setter
    // for odd ones, so both run
    let mut co = BatchCoalescer::new(schema.clone(), target);
    if target % 2 == 0 {
        co = co.with_biggest_coalesce_batch_size(biggest);
    } else {
        co.set_biggest_coalesce_batch_size(biggest);
    }
    if co.biggest_coalesce_batch_size() != biggest {
        return Err(fail(format!(
            "biggest_coalesce_batch_size is {:?}, set to {biggest:?}",
            co.biggest_coalesce_batch_size()
        )));
    }
    let all: Vec<RecordBatch> = batches.iter().map(|(b, _)| b.clone()).collect();
    let columns = DataType::Struct(schema.fields().clone());
    let mut expected: Vec<Vec<Val>> = vec![vec![]; schema.fields().len()];
    let mut out = vec![];
    let (mut pushed, mut emitted) = (0, 0);
    for (k, (b, push)) in batches.into_iter().enumerate() {
        let held = pushed - emitted;
        let mut n = 0;
        for (c, rows) in pushed_rows(&b, &push, &all).into_iter().enumerate() {
            n = if c == 0 { rows.len() } else { n };
            expected[c].extend(rows);
        }
        pushed += n;
        // A known bug's panic in any column, such as `interleave` of a list
        // view of dictionaries (apache/arrow-rs#11620), is discarded
        let r = catch_unwind(AssertUnwindSafe(|| match &push {
            Push::Plain => co.push_batch(b),
            Push::Filter(m) => co.push_batch_with_filter(b, m),
            Push::FilterOptimized(m) => {
                co.push_batch_with_filter_builder(b, FilterBuilder::new(m).optimize())
            }
            Push::FilterCounted(m) => {
                let count = (0..m.len()).filter(|&i| selected(m, i)).count();
                debug_assert_eq!(count, m.iter().filter(|b| *b == Some(true)).count());
                // SAFETY: `count` is the number of rows `m` selects
                co.push_batch_with_filter_builder(b, unsafe {
                    FilterBuilder::new(m).with_count(count)
                })
            }
            Push::Indices(i) => co.push_batch_with_indices(b, i),
            Push::Interleaved(pairs) => co.push_batch_interleaved(&all, pairs),
        }))
        .map_err(|p| known_panic(&columns, &panic_message(&*p)))?;
        r.map_err(|e| fail(format!("push {k} ({push:?}): {e}")))?;
        // No batches is an error, as it is for `interleave_record_batch`, and
        // pushes nothing
        if let Push::Interleaved(_) = push {
            let rows = co.get_buffered_rows();
            if co.push_batch_interleaved(&[], &[]).is_ok() || co.get_buffered_rows() != rows {
                return Err(fail(format!(
                    "push {k}: push_batch_interleaved of no batches did not fail, or changed the {rows} buffered rows to {}",
                    co.get_buffered_rows()
                )));
            }
        }
        let had = co.has_completed_batch();
        let before = out.len();
        while let Some(done) = co.next_completed_batch() {
            emitted += done.num_rows();
            out.push(done);
        }
        // Every pushed row is either in a completed batch or buffered
        let buffered = pushed - emitted;
        if had != (out.len() > before) || co.has_completed_batch() {
            return Err(fail(format!(
                "push {k}: has_completed_batch was {had} before draining {} batches, {} after",
                out.len() - before,
                co.has_completed_batch()
            )));
        }
        if co.get_buffered_rows() != buffered || co.is_empty() != (buffered == 0) {
            return Err(fail(format!(
                "push {k}: {} buffered rows and is_empty {}, expected {buffered}",
                co.get_buffered_rows(),
                co.is_empty()
            )));
        }
        // As documented, rows above the bypass size skip coalescing when
        // nothing is buffered or the buffered rows exceed that size: they come
        // out as one batch, after any buffered rows
        let bypass = biggest.is_some_and(|limit| n > limit && (held == 0 || held > limit));
        if bypass && (buffered != 0 || out.last().map(RecordBatch::num_rows) != Some(n)) {
            return Err(fail(format!(
                "push {k}: {n} rows above bypass size {biggest:?}, with {held} buffered, did not bypass: {buffered} rows buffered after, last batch of {:?} rows",
                out.last().map(RecordBatch::num_rows)
            )));
        }
        // Otherwise the rows are coalesced, and each batch the push completes
        // has the target size, "the number of rows in each output batch"
        if !bypass && let Some(b) = out[before..].iter().find(|b| b.num_rows() != target) {
            return Err(fail(format!(
                "push {k}: {n} rows with {held} buffered and bypass size {biggest:?} completed a batch of {} rows, not the target {target}",
                b.num_rows()
            )));
        }
    }
    co.finish_buffered_batch()
        .map_err(|e| fail(format!("finish_buffered_batch: {e}")))?;
    while let Some(done) = co.next_completed_batch() {
        out.push(done);
    }
    if !co.is_empty() || co.get_buffered_rows() != 0 {
        return Err(fail(format!(
            "{} rows still buffered after finishing",
            co.get_buffered_rows()
        )));
    }
    check_concatenated(&schema, &out, &expected)?;
    // Only a batch that bypasses coalescing, which is larger than the bypass
    // size, can be larger than the target
    if let Some(limit) = biggest
        && let Some(b) = out
            .iter()
            .find(|b| b.num_rows() > target && b.num_rows() <= limit)
    {
        return Err(fail(format!(
            "a batch of {} rows, above target {target} and not above bypass size {limit}",
            b.num_rows()
        )));
    }
    if biggest.is_none() {
        let sizes: Vec<usize> = out.iter().map(|b| b.num_rows()).collect();
        let total = expected.first().map_or(0, |e| e.len());
        let ok = sizes.iter().rev().skip(1).all(|&s| s == target)
            && sizes.last().is_none_or(|&s| s >= 1 && s <= target)
            && sizes.len() == total.div_ceil(target);
        if !ok {
            return Err(fail(format!(
                "batch sizes {sizes:?} for {total} rows and target {target}"
            )));
        }
    }
    Ok(())
}

/// The arrays in a case, for the inventory
pub fn case_arrays(case: &CoalesceCase) -> Vec<ArrayRef> {
    case.1
        .iter()
        .flat_map(|(b, _)| b.columns().to_vec())
        .collect()
}

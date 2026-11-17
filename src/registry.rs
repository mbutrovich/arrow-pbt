//! Every property with its input strategy, and the runners that the tests
//! and the `arrow-pbt` binary share

use proptest::prelude::*;

use crate::cases;
use crate::inventory::{Inputs, Sample, sample};
pub use crate::props::ARITH_ROWS;
use crate::props::{self, R};
use crate::run::{Outcome, check_with, run};
use crate::{batches, logic, numeric, order, strings, temporal};
use std::fmt::Debug;

/// A property, the name of the case strategy it draws from, and closures to
/// run it and to sample its cases. Properties that draw from one strategy
/// give it the same name, so the inventory samples it once
pub struct Entry {
    pub name: &'static str,
    pub cases: &'static str,
    run: Box<dyn Fn(u64, u32, u32) -> Outcome + Send + Sync>,
    sample: Box<dyn Fn(usize, u64) -> Sample + Send + Sync>,
}

impl Entry {
    /// Runs the property on `cases` inputs from one seed
    pub fn run(&self, seed: u64, cases: u32, shrink: u32) -> Outcome {
        (self.run)(seed, cases, shrink)
    }

    /// The type families and layout features of `n` sampled cases
    pub fn sample(&self, n: usize, seed: u64) -> Sample {
        (self.sample)(n, seed)
    }
}

fn entry<T: Inputs + Debug + Clone + 'static>(
    name: &'static str,
    cases: &'static str,
    strategy: fn() -> BoxedStrategy<T>,
    property: fn(T) -> R,
) -> Entry {
    Entry {
        name,
        cases,
        run: Box::new(move |seed, n, shrink| {
            run(
                &strategy(),
                &|t| props::with_known(property, t),
                seed,
                n,
                shrink,
            )
        }),
        sample: Box::new(move |n, seed| sample(strategy(), n, seed)),
    }
}

/// Every property, grouped by the arrow-rs crate it tests
pub fn all_properties() -> Vec<Entry> {
    vec![
        // arrow-select
        entry("take", "take", cases::take, props::take),
        entry("filter", "filter", cases::filter, props::filter),
        entry(
            "interleave",
            "interleave",
            cases::interleave,
            props::interleave,
        ),
        entry("concat", "concat", cases::concat, props::concat),
        entry("shift", "shift", cases::shift, props::shift),
        entry(
            "union_extract",
            "union_extract",
            cases::union_extract,
            props::union_extract,
        ),
        entry("zip", "zip", cases::zip, logic::zip_prop),
        entry("merge", "merge", cases::merge, logic::merge_prop),
        entry("merge_n", "merge_n", cases::merge_n, logic::merge_n_prop),
        entry("nullif", "filter", cases::filter, logic::nullif_prop),
        entry(
            "concat_batches",
            "coalesce",
            cases::coalesce,
            batches::concat_batches_prop,
        ),
        entry(
            "take_record_batch",
            "take_record_batch",
            cases::take_record_batch,
            batches::take_record_batch_prop,
        ),
        entry(
            "interleave_record_batch",
            "interleave_record_batch",
            cases::interleave_record_batch,
            batches::interleave_record_batch_prop,
        ),
        entry(
            "coalesce",
            "coalesce",
            cases::coalesce,
            batches::coalesce_prop,
        ),
        entry(
            "law_filter_take",
            "filter",
            cases::filter,
            props::law_filter_take,
        ),
        entry(
            "law_take_take",
            "take_take",
            cases::take_take,
            props::law_take_take,
        ),
        entry(
            "law_concat_slices",
            "slice",
            cases::slice,
            props::law_concat_slices,
        ),
        entry(
            "law_interleave",
            "interleave",
            cases::interleave,
            props::law_interleave,
        ),
        // arrow-array and arrow-data: slicing, equality, FFI, ArrayData
        entry("slice", "slice", cases::slice, props::slice),
        entry(
            "law_slice_slice",
            "slice_slice",
            cases::slice_slice,
            props::law_slice_slice,
        ),
        entry("eq", "eq", cases::eq, props::eq),
        entry(
            "round_trip_ffi",
            "slice",
            cases::slice,
            props::round_trip_ffi,
        ),
        entry("data_ops", "slice", cases::slice, props::data_ops),
        entry("is_null", "any", cases::any_array, logic::is_null_prop),
        // arrow-ipc
        entry(
            "round_trip_ipc",
            "slice",
            cases::slice,
            props::round_trip_ipc,
        ),
        // arrow-arith
        entry("bools", "bools", cases::bools, props::bools),
        entry("kleene", "bools", cases::bools, logic::kleene),
        entry("arith", "arith", cases::arith, props::arith),
        entry("neg", "neg", cases::neg, numeric::neg_prop),
        entry("arity", "arity", cases::arity, numeric::arity_prop),
        entry("bitwise", "arity", cases::arity, numeric::bitwise_prop),
        entry("divisor", "divisor", cases::divisor, numeric::divisor_prop),
        entry(
            "aggregate",
            "aggregate",
            cases::aggregate,
            numeric::agg_prop,
        ),
        entry(
            "fixed_point",
            "fixed_point",
            cases::fixed_point,
            numeric::fixed_point_prop,
        ),
        entry(
            "temporal_arith",
            "temporal_arith",
            cases::temporal_arith,
            temporal::temporal_arith,
        ),
        entry(
            "interval_scale",
            "interval_scale",
            cases::interval_scale,
            temporal::interval_scale,
        ),
        // arrow-cast
        entry("cast", "cast", cases::cast, props::cast_prop),
        entry(
            "cast_text",
            "cast_text",
            cases::cast_text,
            strings::cast_text,
        ),
        entry(
            "temporal_cast",
            "temporal_cast",
            cases::temporal_cast,
            temporal::temporal_cast,
        ),
        entry(
            "decimal_precision",
            "decimal_precision",
            cases::decimal_precision,
            numeric::decimal_precision_prop,
        ),
        entry(
            "decimal_format",
            "decimal",
            cases::decimal,
            numeric::decimal_format_prop,
        ),
        // arrow-string
        entry("length", "length", cases::length, strings::length_prop),
        entry(
            "string_ops",
            "string_ops",
            cases::string_ops,
            strings::string_ops,
        ),
        // arrow-ord and arrow-row
        entry("cmp", "cmp", cases::cmp, props::cmp_ops),
        entry("sort", "sort", cases::sort, props::sort_prop),
        entry(
            "lexsort",
            "sort_columns",
            cases::sort_columns,
            order::lexsort_prop,
        ),
        entry(
            "row_format",
            "sort_columns",
            cases::sort_columns,
            order::row_format,
        ),
        entry(
            "radix_sort",
            "sort_columns",
            cases::sort_columns,
            order::radix_sort,
        ),
    ]
}

/// The properties named by `names`: "all", one name, or a comma-separated
/// list. An unknown name is an error, so a typo does not shrink a run
pub fn properties(names: &str) -> Result<Vec<Entry>, String> {
    let all = all_properties();
    if names == "all" {
        return Ok(all);
    }
    let wanted: Vec<&str> = names.split(',').collect();
    if let Some(bad) = wanted.iter().find(|&&n| !all.iter().any(|e| e.name == n)) {
        return Err(format!("no property named {bad}"));
    }
    Ok(all
        .into_iter()
        .filter(|e| wanted.contains(&e.name))
        .collect())
}

/// Runs every (property, seed) job on `threads` workers. Each job seeds its
/// own runner, so the results do not depend on scheduling, only timings do
pub fn run_jobs(
    props: &[Entry],
    seeds: &[u64],
    cases: u32,
    shrink: u32,
    threads: usize,
) -> Vec<(&'static str, u64, Outcome)> {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let jobs: Vec<(usize, u64)> = (0..props.len())
        .flat_map(|p| seeds.iter().map(move |&s| (p, s)))
        .collect();
    let next = AtomicUsize::new(0);
    let done = Mutex::new(vec![]);
    std::thread::scope(|scope| {
        for _ in 0..threads.min(jobs.len()).max(1) {
            scope.spawn(|| {
                loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&(p, seed)) = jobs.get(k) else { break };
                    let outcome = props[p].run(seed, cases, shrink);
                    done.lock().unwrap().push((k, props[p].name, seed, outcome));
                }
            });
        }
    });
    let mut done = done.into_inner().unwrap();
    done.sort_by_key(|d| d.0);
    done.into_iter().map(|(_, p, s, o)| (p, s, o)).collect()
}

/// Runs one registered property for the test harness, with the settings
/// `run::check` reads
pub fn check(name: &str) {
    let property = properties(name)
        .unwrap_or_else(|e| panic!("{e}"))
        .pop()
        .expect("one name gives one property");
    check_with(name, |seed, cases, shrink| {
        property.run(seed, cases, shrink)
    });
}

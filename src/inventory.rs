//! What each property actually tests: for sampled cases, which type families
//! and which layout features its input arrays contain, anywhere in their
//! nesting. Written as Markdown tables, where an empty cell is a gap.

use std::collections::{BTreeMap, BTreeSet};

use crate::batches::case_arrays;
use crate::batches::{CoalesceCase, InterleaveBatches};
use crate::order::SortCase;
use crate::props::Operands;
use crate::run::seeded_runner;
use crate::temporal::TemporalCase;
use arrow_array::{Array, ArrayRef, BooleanArray, Int32Array, RecordBatch};
use arrow_schema::{DataType, UnionMode};
use arrow_select::coalesce::BatchCoalescer;
use proptest::strategy::{BoxedStrategy, Strategy, ValueTree};
use proptest::test_runner::Config;
use std::fmt::Debug;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

/// The arrays a property case feeds to kernels
pub trait Inputs {
    fn arrays(&self) -> Vec<ArrayRef>;
}

impl Inputs for (ArrayRef, Int32Array) {
    fn arrays(&self) -> Vec<ArrayRef> {
        vec![self.0.clone(), Arc::new(self.1.clone())]
    }
}
impl Inputs for (ArrayRef, BooleanArray) {
    fn arrays(&self) -> Vec<ArrayRef> {
        vec![self.0.clone(), Arc::new(self.1.clone())]
    }
}
impl Inputs for (Vec<ArrayRef>, Vec<(usize, usize)>) {
    fn arrays(&self) -> Vec<ArrayRef> {
        self.0.clone()
    }
}
impl Inputs for Vec<ArrayRef> {
    fn arrays(&self) -> Vec<ArrayRef> {
        self.clone()
    }
}
impl Inputs for (Vec<ArrayRef>, Vec<Option<usize>>) {
    fn arrays(&self) -> Vec<ArrayRef> {
        self.0.clone()
    }
}
impl Inputs for (RecordBatch, Int32Array) {
    fn arrays(&self) -> Vec<ArrayRef> {
        let mut out = self.0.columns().to_vec();
        out.push(Arc::new(self.1.clone()));
        out
    }
}
impl Inputs for InterleaveBatches {
    fn arrays(&self) -> Vec<ArrayRef> {
        self.0.iter().flat_map(|b| b.columns().to_vec()).collect()
    }
}
impl Inputs for (ArrayRef, i8, bool) {
    fn arrays(&self) -> Vec<ArrayRef> {
        vec![self.0.clone()]
    }
}
impl Inputs for (ArrayRef, i64) {
    fn arrays(&self) -> Vec<ArrayRef> {
        vec![self.0.clone()]
    }
}
impl Inputs for (ArrayRef, usize, usize) {
    fn arrays(&self) -> Vec<ArrayRef> {
        vec![self.0.clone()]
    }
}
impl Inputs for (ArrayRef, ArrayRef) {
    fn arrays(&self) -> Vec<ArrayRef> {
        vec![self.0.clone(), self.1.clone()]
    }
}
impl Inputs for (ArrayRef, ArrayRef, i8) {
    fn arrays(&self) -> Vec<ArrayRef> {
        vec![self.0.clone(), self.1.clone()]
    }
}
impl Inputs for (BooleanArray, BooleanArray) {
    fn arrays(&self) -> Vec<ArrayRef> {
        vec![Arc::new(self.0.clone()), Arc::new(self.1.clone())]
    }
}
impl Inputs for (ArrayRef, Int32Array, Int32Array) {
    fn arrays(&self) -> Vec<ArrayRef> {
        vec![
            self.0.clone(),
            Arc::new(self.1.clone()),
            Arc::new(self.2.clone()),
        ]
    }
}
impl Inputs for (ArrayRef, usize, usize, usize, usize) {
    fn arrays(&self) -> Vec<ArrayRef> {
        vec![self.0.clone()]
    }
}
impl Inputs for (ArrayRef, ArrayRef, bool) {
    fn arrays(&self) -> Vec<ArrayRef> {
        vec![self.0.clone(), self.1.clone()]
    }
}
impl Inputs for (ArrayRef, ArrayRef, Operands) {
    fn arrays(&self) -> Vec<ArrayRef> {
        vec![self.0.clone(), self.1.clone()]
    }
}
impl Inputs for (ArrayRef, bool, bool, Option<usize>) {
    fn arrays(&self) -> Vec<ArrayRef> {
        vec![self.0.clone()]
    }
}
impl Inputs for (ArrayRef, DataType, bool) {
    fn arrays(&self) -> Vec<ArrayRef> {
        vec![self.0.clone()]
    }
}

impl Inputs for ArrayRef {
    fn arrays(&self) -> Vec<ArrayRef> {
        vec![self.clone()]
    }
}
impl Inputs for (BooleanArray, ArrayRef, ArrayRef, bool, bool) {
    fn arrays(&self) -> Vec<ArrayRef> {
        vec![Arc::new(self.0.clone()), self.1.clone(), self.2.clone()]
    }
}
impl Inputs for (ArrayRef, ArrayRef, bool, i64, Option<u64>) {
    fn arrays(&self) -> Vec<ArrayRef> {
        vec![self.0.clone(), self.1.clone()]
    }
}

impl Inputs for SortCase {
    fn arrays(&self) -> Vec<ArrayRef> {
        self.0.clone()
    }
}

impl Inputs for TemporalCase {
    fn arrays(&self) -> Vec<ArrayRef> {
        vec![self.0.clone(), self.1.clone()]
    }
}
impl Inputs for CoalesceCase {
    fn arrays(&self) -> Vec<ArrayRef> {
        case_arrays(self)
    }
}

/// Which `BatchCoalescer` paths each sampled case reaches: how batches are
/// pushed, which per-type in-progress builder each column uses, and whether
/// a batch is large enough to bypass coalescing
pub fn coalesce_paths(
    strategy: BoxedStrategy<CoalesceCase>,
    n: usize,
    seed: u64,
) -> BTreeMap<&'static str, usize> {
    use crate::batches::Push;
    let mut runner = seeded_runner(Config::default(), seed);
    let mut out = BTreeMap::new();
    for _ in 0..n {
        let (schema, batches, target, biggest) = strategy.new_tree(&mut runner).unwrap().current();
        let mut seen = BTreeSet::new();
        for (b, p) in &batches {
            seen.insert(match p {
                Push::Plain => "push: plain",
                Push::Filter(_) => "push: filter",
                Push::FilterOptimized(_) => "push: optimized filter",
                Push::FilterCounted(_) => "push: filter with count",
                Push::Indices(_) => "push: indices",
                Push::Interleaved(_) => "push: interleaved",
            });
            if b.num_rows() > target {
                seen.insert("a batch longer than the target");
            }
            if biggest.is_some_and(|l| b.num_rows() > l) {
                seen.insert("a batch above the bypass size");
            }
        }
        // The builder `create_in_progress_array` picks in arrow-select's
        // coalesce.rs
        for f in schema.fields() {
            seen.insert(match f.data_type() {
                DataType::Utf8View | DataType::BinaryView => "column: view builder",
                DataType::FixedSizeBinary(_) => "column: fixed-size binary builder",
                DataType::Boolean => "column: boolean builder",
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary => {
                    "column: byte builder"
                }
                DataType::List(_) | DataType::LargeList(_) => "column: list builder",
                d if d.is_primitive() => "column: primitive builder",
                _ => "column: generic (concat) builder",
            });
        }
        if batches.len() >= 2 {
            seen.insert("two or more batches");
        }
        // The fused sparse filter copy, which copies the kept rows column by
        // column instead of filtering the batch first. `coalesce.rs` takes it
        // when every column has a direct filter copy, 1 to 1/16 of the
        // filter's rows are kept (1/1024 with an offset-based byte column),
        // they fit the rows left in the buffer, and they do not exceed the
        // bypass size. The buffered rows come from a coalescer fed the same
        // batches
        let direct = schema.fields().iter().all(|f| {
            let d = f.data_type();
            d.is_primitive()
                || matches!(
                    d,
                    DataType::Boolean
                        | DataType::Utf8
                        | DataType::LargeUtf8
                        | DataType::Binary
                        | DataType::LargeBinary
                        | DataType::Utf8View
                        | DataType::BinaryView
                        | DataType::List(_)
                        | DataType::LargeList(_)
                )
        });
        let bytes = schema.fields().iter().any(|f| {
            matches!(
                f.data_type(),
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary
            )
        });
        // A known arrow-rs bug can make the coalescer fail or panic, after
        // which its buffered rows say nothing, so the case stops there
        let all: Vec<RecordBatch> = batches.iter().map(|(b, _)| b.clone()).collect();
        let mut co = BatchCoalescer::new(schema.clone(), target);
        co.set_biggest_coalesce_batch_size(biggest);
        for (b, p) in &batches {
            if let Push::Filter(m) | Push::FilterOptimized(m) | Push::FilterCounted(m) = p {
                let kept = (0..m.len())
                    .filter(|&i| m.is_valid(i) && m.value(i))
                    .count();
                let sparse = kept > 0
                    && kept <= m.len() / 16
                    && (!bytes || kept <= m.len() / 1024)
                    && kept <= target - co.get_buffered_rows()
                    && biggest.is_none_or(|l| kept <= l);
                if direct && sparse {
                    seen.insert("push: sparse filter copy");
                    if bytes {
                        seen.insert("push: sparse filter copy with a byte column");
                    }
                }
            }
            let pushed = catch_unwind(AssertUnwindSafe(|| match p {
                Push::Plain => co.push_batch(b.clone()),
                Push::Filter(m) | Push::FilterOptimized(m) | Push::FilterCounted(m) => {
                    co.push_batch_with_filter(b.clone(), m)
                }
                Push::Indices(i) => co.push_batch_with_indices(b.clone(), i),
                Push::Interleaved(pairs) => co.push_batch_interleaved(&all, pairs),
            }));
            if !matches!(pushed, Ok(Ok(()))) {
                break;
            }
            while co.next_completed_batch().is_some() {}
        }
        for k in seen {
            *out.entry(k).or_default() += 1;
        }
    }
    out
}

/// Which `lexsort_to_indices` path each sampled case reaches: one column
/// goes to `sort_to_indices`, a limit of at most a tenth of the rows to a
/// heap (specialized for 2 to 5 columns), anything else to a partial sort
pub fn lexsort_paths(
    strategy: BoxedStrategy<SortCase>,
    n: usize,
    seed: u64,
) -> BTreeMap<&'static str, usize> {
    let mut runner = seeded_runner(Config::default(), seed);
    let mut out = BTreeMap::new();
    for _ in 0..n {
        let (cols, _, limit) = strategy.new_tree(&mut runner).unwrap().current();
        let rows = cols[0].len();
        let path = match (cols.len(), limit) {
            (1, _) => "one column: sort_to_indices",
            (2..=5, Some(l)) if l <= rows / 10 => "heap, 2 to 5 columns",
            (_, Some(l)) if l <= rows / 10 => "heap, 6 columns",
            _ => "partial sort",
        };
        *out.entry(path).or_default() += 1;
        if rows >= 1024 {
            *out.entry("1024 rows or more").or_default() += 1;
        }
    }
    out
}

/// Which `zip` implementation each sampled case reaches: the scalar
/// zippers, the `interleave` path (both arrays, length >= 1024, more than
/// length / 8 runs of true), or `MutableArrayData`
pub fn zip_paths(
    strategy: BoxedStrategy<(BooleanArray, ArrayRef, ArrayRef, bool, bool)>,
    n: usize,
    seed: u64,
) -> BTreeMap<&'static str, usize> {
    let mut runner = seeded_runner(Config::default(), seed);
    let mut out = BTreeMap::new();
    for _ in 0..n {
        let (m, t, f, ts, fs) = strategy.new_tree(&mut runner).unwrap().current();
        let (ts, fs) = (ts && !t.is_empty(), fs && !f.is_empty());
        let sel: Vec<bool> = (0..m.len()).map(|i| m.is_valid(i) && m.value(i)).collect();
        let runs = (0..sel.len())
            .filter(|&i| sel[i] && (i == 0 || !sel[i - 1]))
            .count();
        let mixed = sel.contains(&true) && sel.contains(&false);
        let path = match (ts, fs) {
            (true, true) if mixed => "both scalars, mixed mask",
            (true, true) => "both scalars, constant mask",
            (true, false) | (false, true) => "one scalar",
            _ if m.len() >= 1024 && runs > m.len() / 8 => "arrays, interleave path",
            _ => "arrays, MutableArrayData path",
        };
        *out.entry(path).or_default() += 1;
        if !ts && !fs && m.len() >= 1024 {
            *out.entry("arrays, length >= 1024").or_default() += 1;
        }
        if m.null_count() > 0 {
            *out.entry("mask has nulls").or_default() += 1;
        }
    }
    out
}

/// The family of a type, coarse enough for a table row
fn family(dt: &DataType) -> &'static str {
    use DataType::*;
    match dt {
        Null => "Null",
        Boolean => "Boolean",
        Int8 | Int16 | Int32 | Int64 => "signed integers",
        UInt8 | UInt16 | UInt32 | UInt64 => "unsigned integers",
        Float16 | Float32 | Float64 => "floats",
        Decimal32(..) | Decimal64(..) | Decimal128(..) | Decimal256(..) => "decimals",
        Date32 | Date64 | Time32(_) | Time64(_) | Timestamp(..) | Duration(_) | Interval(_) => {
            "temporal"
        }
        Utf8 | LargeUtf8 => "Utf8, LargeUtf8",
        Binary | LargeBinary => "Binary, LargeBinary",
        Utf8View | BinaryView => "Utf8View, BinaryView",
        FixedSizeBinary(_) => "FixedSizeBinary",
        List(_) | LargeList(_) => "List, LargeList",
        ListView(_) | LargeListView(_) => "ListView, LargeListView",
        FixedSizeList(..) => "FixedSizeList",
        Struct(_) => "Struct",
        Map(..) => "Map",
        Union(_, UnionMode::Sparse) => "sparse Union",
        Union(_, UnionMode::Dense) => "dense Union",
        Dictionary(..) => "Dictionary",
        RunEndEncoded(..) => "RunEndEncoded",
    }
}

pub const FAMILIES: &[&str] = &[
    "Null",
    "Boolean",
    "signed integers",
    "unsigned integers",
    "floats",
    "decimals",
    "temporal",
    "Utf8, LargeUtf8",
    "Binary, LargeBinary",
    "Utf8View, BinaryView",
    "FixedSizeBinary",
    "List, LargeList",
    "ListView, LargeListView",
    "FixedSizeList",
    "Struct",
    "Map",
    "sparse Union",
    "dense Union",
    "Dictionary",
    "RunEndEncoded",
];

fn families(dt: &DataType, out: &mut BTreeSet<&'static str>) {
    use DataType::*;
    out.insert(family(dt));
    match dt {
        List(c)
        | LargeList(c)
        | ListView(c)
        | LargeListView(c)
        | FixedSizeList(c, _)
        | Map(c, _) => families(c.data_type(), out),
        Struct(fs) => fs.iter().for_each(|f| families(f.data_type(), out)),
        Union(fs, _) => fs.iter().for_each(|(_, f)| families(f.data_type(), out)),
        Dictionary(_, v) => families(v, out),
        RunEndEncoded(_, v) => families(v.data_type(), out),
        _ => {}
    }
}

/// Per sampled case: the type families and layout features of its inputs
pub struct Sample {
    pub cases: usize,
    pub families: BTreeMap<&'static str, usize>,
    pub features: BTreeMap<&'static str, usize>,
}

pub fn sample<T: Inputs + Debug>(strategy: BoxedStrategy<T>, n: usize, seed: u64) -> Sample {
    let mut runner = seeded_runner(Config::default(), seed);
    let mut s = Sample {
        cases: n,
        families: BTreeMap::new(),
        features: BTreeMap::new(),
    };
    for _ in 0..n {
        let case = strategy.new_tree(&mut runner).unwrap().current();
        let (mut fam, mut feat) = (BTreeSet::new(), BTreeSet::new());
        for a in case.arrays() {
            families(a.data_type(), &mut fam);
            crate::coverage::features(a.as_ref(), &mut feat);
        }
        for f in fam {
            *s.families.entry(f).or_default() += 1;
        }
        for f in feat {
            *s.features.entry(f).or_default() += 1;
        }
    }
    s
}

/// A Markdown table: one row per key, one column per property, cells as the
/// percentage of that property's cases, blank when none
pub fn table(
    title: &str,
    keys: &[&'static str],
    samples: &[(&'static str, Sample)],
    pick: fn(&Sample) -> &BTreeMap<&'static str, usize>,
) -> String {
    let header = format!(
        "## {title}\n\n| | {} |\n|---|{}\n",
        samples
            .iter()
            .map(|(p, _)| *p)
            .collect::<Vec<_>>()
            .join(" | "),
        "---|".repeat(samples.len())
    );
    let rows = keys.iter().map(|k| {
        let cells: Vec<String> = samples
            .iter()
            .map(|(_, s)| match pick(s).get(k).copied().unwrap_or(0) {
                0 => String::new(),
                c => format!("{:.0}", 100.0 * c as f64 / s.cases as f64).replace(".0", ""),
            })
            .collect();
        format!("| {k} | {} |\n", cells.join(" | "))
    });
    std::iter::once(header).chain(rows).collect()
}

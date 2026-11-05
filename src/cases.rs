//! Case strategies: for each property, the strategy that generates its
//! cases, built from the generator in `generator.rs`. Each is named
//! `<name>_case`, and docs/design.md defines the terms

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, BooleanArray, Int32Array, make_array};
use proptest::collection::vec;
use proptest::prelude::*;
use proptest::sample::select;

use crate::batches::{CoalesceCase, InterleaveBatches};
use crate::generator::value_types;
use crate::generator::{
    GeneratedArray, GeneratorType, ValueDomain, all_types, array, array_of, batch, build, indices,
    len, mask, rows, schemas, seed,
};
use crate::numeric::FixedPointCase;
use crate::order::SortCase;
use crate::props::{Interleave, Operands};
use crate::temporal::IntervalScaleCase;
use crate::temporal::TemporalCase;
use crate::temporal::TemporalCastCase;
use crate::types::any_type;
use crate::val::Val;
use arrow_array::RecordBatch;
use arrow_array::RecordBatchOptions;
use arrow_schema::DataType;
use arrow_schema::SchemaRef;
use arrow_schema::SortOptions;
use proptest::sample::Index;
use std::mem::discriminant;

/// `k` arrays of one type, each with its own length and layout
fn same_type(k: std::ops::RangeInclusive<usize>) -> BoxedStrategy<(GeneratorType, Vec<ArrayRef>)> {
    all_types()
        .prop_flat_map(move |t| {
            let t2 = t.clone();
            let part = len().prop_flat_map(move |n| (rows(&t2, n), seed()));
            (Just(t), vec(part, k.clone()))
        })
        .prop_map(|(t, parts)| {
            let arrays = parts.iter().map(|(r, s)| build(&t, r, *s)).collect();
            (t, arrays)
        })
        .boxed()
}

pub fn take_in_bounds() -> BoxedStrategy<(ArrayRef, Int32Array)> {
    array_of(all_types())
        .prop_flat_map(
            |GeneratedArray {
                 ty: t,
                 rows: r,
                 seed: s,
                 ..
             }| (Just(build(&t, &r, s)), indices(r.len())),
        )
        .prop_map(|(a, i)| (a, i.build()))
        .boxed()
}

pub fn filter() -> BoxedStrategy<(ArrayRef, BooleanArray)> {
    array_of(all_types())
        .prop_flat_map(
            |GeneratedArray {
                 ty: t,
                 rows: r,
                 seed: s,
                 ..
             }| (Just(build(&t, &r, s)), mask(r.len())),
        )
        .prop_map(|(a, m)| (a, m.build()))
        .boxed()
}

/// `(input, row)` pairs into inputs of lengths `lens`
fn pairs(lens: Vec<usize>) -> BoxedStrategy<Vec<(usize, usize)>> {
    let nonempty: Vec<usize> = (0..lens.len()).filter(|&i| lens[i] > 0).collect();
    if nonempty.is_empty() {
        return Just(vec![]).boxed();
    }
    let pair = select(nonempty).prop_flat_map(move |a| (Just(a), 0..lens[a]));
    len().prop_flat_map(move |c| vec(pair.clone(), c)).boxed()
}

pub fn interleave() -> BoxedStrategy<Interleave> {
    same_type(1..=4)
        .prop_flat_map(|(_, arrays)| {
            let lens: Vec<usize> = arrays.iter().map(|a| a.len()).collect();
            (Just(arrays), pairs(lens))
        })
        .boxed()
}

/// A built batch of a generated schema with `n` rows
fn built_batch(schema: &SchemaRef, n: usize) -> BoxedStrategy<RecordBatch> {
    batch(schema, n, ValueDomain::Arrow)
        .expect("every generated type converts")
        .prop_map(|g| g.build())
        .boxed()
}

/// `cols` as a batch of `n` rows. A batch with no columns needs the count
fn record_batch(schema: SchemaRef, cols: Vec<ArrayRef>, n: usize) -> RecordBatch {
    let options = RecordBatchOptions::new().with_row_count(Some(n));
    RecordBatch::try_new_with_options(schema, cols, &options).unwrap()
}

/// A batch and indices into it, as `take` draws them
pub fn take_record_batch() -> BoxedStrategy<(RecordBatch, Int32Array)> {
    (schemas(), len())
        .prop_flat_map(|(schema, n)| (built_batch(&schema, n), take_indices(n)))
        .boxed()
}

/// Zero to four batches of one schema, each with its own length, and
/// `(batch, row)` pairs into them
pub fn interleave_record_batch() -> BoxedStrategy<InterleaveBatches> {
    (schemas(), prop_oneof![1 => Just(0usize), 9 => 1..=4usize])
        .prop_flat_map(|(schema, k)| {
            let one = len().prop_flat_map(move |n| built_batch(&schema, n));
            vec(one, k)
        })
        .prop_flat_map(|batches| {
            let lens: Vec<usize> = batches.iter().map(|b| b.num_rows()).collect();
            (Just(batches), pairs(lens))
        })
        .boxed()
}

pub fn concat() -> BoxedStrategy<Vec<ArrayRef>> {
    same_type(1..=4).prop_map(|(_, arrays)| arrays).boxed()
}

/// An array and a shift: 0, one row or all but one either way, anything
/// shorter than the length either way, the length or one more either way,
/// or the `i64` limits. The last two make every row null, so they get a
/// quarter of the cases
pub fn shift() -> BoxedStrategy<(ArrayRef, i64)> {
    array_of(all_types())
        .prop_flat_map(|g| {
            let n = g.rows.len() as i64;
            let within = prop_oneof![
                select(vec![1, -1, n - 1, 1 - n]),
                (1..n.max(2), any::<bool>()).prop_map(|(o, left)| if left { -o } else { o }),
            ];
            let offset = prop_oneof![
                1 => Just(0),
                5 => within,
                1 => select(vec![n, -n, n + 1, -n - 1]),
                1 => select(vec![i64::MIN, i64::MIN + 1, i64::MAX]),
            ];
            (Just(g.build()), offset)
        })
        .boxed()
}

/// A union of one to three fields of any types, each nullable three times
/// in four, with default or other type ids, and names that repeat in a
/// quarter of the unions. Then a target type id, which one time in ten
/// names no field, and whether to extract by name
pub fn union_extract() -> BoxedStrategy<(ArrayRef, i8, bool)> {
    use arrow_schema::{Field, UnionFields, UnionMode};
    let mode = prop_oneof![Just(UnionMode::Sparse), Just(UnionMode::Dense)];
    let nullable = prop_oneof![3 => Just(true), 1 => Just(false)];
    (
        vec((all_types(), nullable), 1..=3),
        mode,
        any::<bool>(),
        prop_oneof![3 => Just(false), 1 => Just(true)],
    )
        .prop_map(|(cs, mode, odd_ids, repeat)| {
            let ids: Vec<i8> = (0..cs.len() as i8)
                .map(|i| if odd_ids { 5 + 3 * i } else { i })
                .collect();
            let fields = cs.iter().enumerate().map(|(i, (t, nullable))| {
                let name = format!("u{}", if repeat { i / 2 } else { i });
                // A Null array is all null, so a field with one inside must
                // be nullable
                let nullable = *nullable || any_type(&t.data_type(), &DataType::is_null);
                Field::new(name, t.data_type(), nullable)
            });
            let dt = DataType::Union(UnionFields::try_new(ids.clone(), fields).unwrap(), mode);
            (GeneratorType::try_from(&dt).unwrap(), ids)
        })
        .prop_flat_map(|(t, ids)| {
            // 4 is neither a default nor another type id of up to three fields
            let target = prop_oneof![9 => select(ids), 1 => Just(4i8)];
            (array_of(Just(t).boxed()), target, any::<bool>())
        })
        .prop_map(|(g, target, by_name)| (g.build(), target, by_name))
        .boxed()
}

pub fn slice() -> BoxedStrategy<(ArrayRef, usize, usize)> {
    array_of(all_types())
        .prop_flat_map(
            |GeneratedArray {
                 ty: t,
                 rows: r,
                 seed: s,
                 ..
             }| {
                let n = r.len();
                (Just(build(&t, &r, s)), 0..=n)
                    .prop_flat_map(move |(a, o)| (Just(a), Just(o), 0..=n - o))
            },
        )
        .boxed()
}

pub fn eq() -> BoxedStrategy<(ArrayRef, ArrayRef)> {
    array_of(all_types())
        .prop_flat_map(
            |GeneratedArray {
                 ty: t,
                 rows: r1,
                 seed: s1,
                 ..
             }| {
                let n = r1.len();
                let t2 = t.clone();
                // Equal values in another layout, other values of the same
                // length, or another length
                let other: BoxedStrategy<Vec<Val>> = prop_oneof![
                    2 => Just(r1.clone()),
                    1 => rows(&t, n),
                    1 => len().prop_flat_map(move |m| rows(&t2, m)),
                ]
                .boxed();
                (Just(t), Just(r1), Just(s1), other, seed())
            },
        )
        .prop_map(|(t, r1, s1, r2, s2)| (build(&t, &r1, s1), build(&t, &r2, s2)))
        .boxed()
}

pub fn bools() -> BoxedStrategy<(BooleanArray, BooleanArray)> {
    len()
        .prop_flat_map(|n| (mask(n), mask(n)))
        .prop_map(|(a, b)| (a.build(), b.build()))
        .boxed()
}

/// Types the `cmp` kernels and sort accept: leaves, and dictionaries and
/// run-end encoded arrays of leaves
fn comparable() -> BoxedStrategy<GeneratorType> {
    use value_types;
    let leaf = prop_oneof![
        1 => Just(GeneratorType::Bool),
        1 => Just(GeneratorType::I32),
        1 => Just(GeneratorType::I64),
        1 => Just(GeneratorType::Utf8),
        1 => Just(GeneratorType::LargeUtf8),
        1 => Just(GeneratorType::Utf8View),
        1 => Just(GeneratorType::Binary),
        1 => Just(GeneratorType::LargeBinary),
        1 => Just(GeneratorType::BinaryView),
        1 => (1..=4i32).prop_map(GeneratorType::Fsb),
        4 => select(value_types()).prop_map(GeneratorType::Prim),
    ];
    prop_oneof![
        8 => leaf,
        1 => (select(vec![DataType::Int8, DataType::UInt16, DataType::Int32]), prop_oneof![Just(GeneratorType::Utf8), Just(GeneratorType::I32)])
            .prop_map(|(k, v)| GeneratorType::Dict(k, Box::new(v))),
        1 => prop_oneof![Just(GeneratorType::I32), Just(GeneratorType::Utf8)].prop_map(|v| GeneratorType::Ree(DataType::Int32, Box::new(v))),
    ]
    .boxed()
}

/// Two arrays of one comparable type and length, and whether to compare
/// against the second's first row as a scalar. Half the time the second
/// array's rows are the first's, so equal pairs are common
pub fn cmp() -> BoxedStrategy<(ArrayRef, ArrayRef, bool)> {
    array_of(comparable())
        .prop_flat_map(
            |GeneratedArray {
                 ty: t,
                 rows: r1,
                 seed: s1,
                 ..
             }| {
                let n = r1.len();
                let other = prop_oneof![1 => Just(r1.clone()), 1 => rows(&t, n)];
                (Just(t), Just(r1), Just(s1), other, seed(), any::<bool>())
            },
        )
        .prop_map(|(t, r1, s1, r2, s2, scalar)| (build(&t, &r1, s1), build(&t, &r2, s2), scalar))
        .boxed()
}

/// An array of a comparable type, sort options and an optional limit
pub fn sort() -> BoxedStrategy<(ArrayRef, bool, bool, Option<usize>)> {
    array_of(comparable())
        .prop_flat_map(
            |GeneratedArray {
                 ty: t,
                 rows: r,
                 seed: s,
                 ..
             }| {
                let n = r.len();
                (
                    Just(build(&t, &r, s)),
                    any::<bool>(),
                    any::<bool>(),
                    proptest::option::of(0..=n + 2),
                )
            },
        )
        .boxed()
}

/// The generator type of `dt`, one of the leaf types the strategies here
/// draw. Those all convert, so a failure is a bug in this module
fn leaf(dt: &DataType) -> GeneratorType {
    GeneratorType::try_from(dt).expect("a leaf type the generator builds")
}

/// Two arrays of numeric type and one length, and whether to use the first
/// row of one as a scalar. Decimals pair with another decimal of the same
/// width whose precision and scale may differ
pub fn arith() -> BoxedStrategy<(ArrayRef, ArrayRef, Operands)> {
    use DataType as D;
    let numeric: Vec<D> = value_types()
        .into_iter()
        .filter(DataType::is_numeric)
        .chain([D::Int32, D::Int64])
        .collect();
    select(numeric)
        .prop_flat_map(|lt| {
            let same_width: Vec<D> = value_types()
                .into_iter()
                .filter(|d| discriminant(d) == discriminant(&lt))
                .collect();
            // A decimal's right side may have another precision and scale
            let rt = if lt.is_decimal() && !same_width.is_empty() {
                prop_oneof![Just(lt.clone()), select(same_width)].boxed()
            } else {
                Just(lt.clone()).boxed()
            };
            (Just(lt), rt, len())
        })
        .prop_flat_map(|(lt, rt, n)| {
            let (tl, tr) = (leaf(&lt), leaf(&rt));
            (
                Just(tl.clone()),
                Just(tr.clone()),
                rows(&tl, n),
                rows(&tr, n),
                seed(),
                seed(),
                prop_oneof![
                    2 => Just(Operands::Arrays),
                    1 => Just(Operands::ScalarLeft),
                    1 => Just(Operands::ScalarRight),
                ],
            )
        })
        .prop_map(|(tl, tr, rl, rr, sl, sr, operands)| {
            (build(&tl, &rl, sl), build(&tr, &rr, sr), operands)
        })
        .boxed()
}

/// An interval array and an `Int64` factor array of one length, or a
/// `MonthDayNano` array and a `Float64` one, and which is a one-row
/// scalar. Half the `Float64` factors are ratios of small integers, which
/// leave fractions of months and days to cascade
pub fn interval_scale() -> BoxedStrategy<IntervalScaleCase> {
    use arrow_array::Float64Array;
    use arrow_schema::{DataType as D, IntervalUnit as I};
    let by_int = (
        select(vec![I::YearMonth, I::DayTime, I::MonthDayNano]),
        len(),
    )
        .prop_flat_map(|(u, n)| {
            let (ti, tf) = (leaf(&D::Interval(u)), leaf(&D::Int64));
            (rows(&ti, n), rows(&tf, n), seed(), seed())
                .prop_map(move |(ri, rf, si, sf)| (build(&ti, &ri, si), build(&tf, &rf, sf)))
        });
    let by_float = len().prop_flat_map(|n| {
        let (ti, tf) = (leaf(&D::Interval(I::MonthDayNano)), leaf(&D::Float64));
        let ratio = prop_oneof![
            1 => Just(None),
            7 => (-1000i64..=1000, 1i64..=64).prop_map(|(p, q)| Some(p as f64 / q as f64)),
        ];
        let factor = prop_oneof![
            (rows(&tf, n), seed()).prop_map(move |(r, s)| build(&tf, &r, s)),
            vec(ratio, n).prop_map(|v| Arc::new(Float64Array::from(v)) as ArrayRef),
        ];
        (rows(&ti, n), seed(), factor).prop_map(move |(ri, si, f)| (build(&ti, &ri, si), f))
    });
    let operands = prop_oneof![
        2 => Just(Operands::Arrays),
        1 => Just(Operands::ScalarLeft),
        1 => Just(Operands::ScalarRight),
    ];
    (prop_oneof![by_int, by_float], operands)
        .prop_map(|((i, f), o)| (i, f, o))
        .boxed()
}

/// Two integer arrays of one type and length
pub fn arity() -> BoxedStrategy<(ArrayRef, ArrayRef)> {
    use DataType as D;
    let ints = vec![
        D::Int8,
        D::Int16,
        D::Int32,
        D::Int64,
        D::UInt8,
        D::UInt16,
        D::UInt32,
        D::UInt64,
    ];
    (select(ints), len())
        .prop_flat_map(|(dt, n)| {
            let t = leaf(&dt);
            (Just(t.clone()), rows(&t, n), rows(&t, n), seed(), seed())
        })
        .prop_map(|(t, ra, rb, sa, sb)| (build(&t, &ra, sa), build(&t, &rb, sb)))
        .boxed()
}

/// A numeric array, a numeric target type, and whether the cast is `safe`
pub fn cast() -> BoxedStrategy<(ArrayRef, DataType, bool)> {
    use DataType as D;
    let numeric: Vec<D> = value_types()
        .into_iter()
        .filter(DataType::is_numeric)
        .chain([D::Int32, D::Int64])
        .collect();
    (
        select(numeric.clone()),
        select(numeric),
        len(),
        any::<bool>(),
    )
        .prop_flat_map(|(from, to, n, safe)| {
            let t = leaf(&from);
            (Just(t.clone()), rows(&t, n), seed(), Just(to), Just(safe))
        })
        .prop_map(|(t, r, s, to, safe)| (build(&t, &r, s), to, safe))
        .boxed()
}

/// A take case where, one time in five, one index is valid and outside the
/// array, which `take` documents as an error when it checks bounds
pub fn take() -> BoxedStrategy<(ArrayRef, Int32Array)> {
    array_of(all_types())
        .prop_flat_map(
            |GeneratedArray {
                 ty: t,
                 rows: r,
                 seed: s,
                 ..
             }| { (Just(build(&t, &r, s)), take_indices(r.len())) },
        )
        .boxed()
}

/// Indices into `n` rows, one of them a valid index out of bounds one time
/// in five
fn take_indices(n: usize) -> BoxedStrategy<Int32Array> {
    let m = n as i64;
    let out = prop_oneof![
        4 => Just(None),
        1 => (any::<Index>(), select(vec![m, m + 1, i32::MAX as i64, -1, i32::MIN as i64])).prop_map(Some),
    ];
    (indices(n), out)
        .prop_map(|(mut idx, out)| {
            if let (Some((at, v)), false) = (out, idx.values.is_empty()) {
                let k = at.index(idx.values.len());
                idx.values[k] = Some(v as i32);
            }
            idx.build()
        })
        .boxed()
}

/// Arrays of the types `length` documents: strings, binary, lists, list
/// views and fixed-size lists, and dictionaries and run-end encoded arrays
/// of those
pub fn length() -> BoxedStrategy<ArrayRef> {
    let accepted = |t: &GeneratorType| match t {
        GeneratorType::Utf8
        | GeneratorType::LargeUtf8
        | GeneratorType::Utf8View
        | GeneratorType::Binary
        | GeneratorType::LargeBinary => true,
        GeneratorType::List(_)
        | GeneratorType::LargeList(_)
        | GeneratorType::ListView(_)
        | GeneratorType::LargeListView(_)
        | GeneratorType::Fsl(..) => true,
        GeneratorType::Dict(_, v) | GeneratorType::Ree(_, v) => {
            matches!(**v, GeneratorType::Utf8 | GeneratorType::Binary)
        }
        _ => false,
    };
    // The generator's own types, kept to those, so nothing is discarded
    crate::generator::ty(1)
        .prop_filter("a type length accepts", move |t| accepted(t))
        .prop_flat_map(|t| {
            (Just(t.clone()), len()).prop_flat_map(|(t, n)| (Just(t.clone()), rows(&t, n), seed()))
        })
        .prop_map(|(t, r, s)| build(&t, &r, s))
        .boxed()
}

/// A pattern for `like` and the substring predicates, often derived from
/// the value it is tested against so that matches are common
fn pattern_for(v: Val) -> BoxedStrategy<Val> {
    let Val::Str(s) = v else {
        return crate::generator::text().prop_map(Val::Str).boxed();
    };
    let cs: Vec<char> = s.chars().collect();
    let n = cs.len();
    (0..=n, 0..=n, 0..6usize, any::<bool>())
        .prop_map(move |(a, b, mode, escape)| {
            let (a, b) = (a.min(b), a.max(b));
            let piece: String = cs[a..b].iter().collect();
            // Escape LIKE syntax in the piece half the time
            let piece = if escape {
                piece
                    .replace('\\', "\\\\")
                    .replace('%', "\\%")
                    .replace('_', "\\_")
            } else {
                piece
            };
            let full: String = cs.iter().collect();
            Val::Str(match mode {
                0 => format!("{piece}%"),
                1 => format!("%{piece}"),
                2 => format!("%{piece}%"),
                3 => piece
                    .chars()
                    .map(|c| if c == 'a' { '_' } else { c })
                    .collect(),
                4 => full,
                _ => piece,
            })
        })
        .boxed()
}

/// Two string or binary arrays of one type and length, the second often
/// patterns derived from the first, plus a scalar flag and substring bounds
pub fn string_ops() -> BoxedStrategy<(ArrayRef, ArrayRef, bool, i64, Option<u64>)> {
    let types = vec![
        GeneratorType::Utf8,
        GeneratorType::LargeUtf8,
        GeneratorType::Utf8View,
        GeneratorType::Binary,
        GeneratorType::LargeBinary,
        GeneratorType::BinaryView,
    ];
    (select(types), len())
        .prop_flat_map(|(t, n)| (Just(t.clone()), rows(&t, n)))
        .prop_flat_map(|(t, r1)| {
            let t2 = t.clone();
            let derived: Vec<BoxedStrategy<Val>> = r1
                .iter()
                .map(|v| match v {
                    // Patterns only for text; binary gets values of its type
                    Val::Str(_) => {
                        prop_oneof![3 => pattern_for(v.clone()), 1 => crate::generator::val(&t2)]
                            .boxed()
                    }
                    _ => crate::generator::val(&t2),
                })
                .collect();
            (
                Just(t),
                Just(r1),
                derived,
                seed(),
                seed(),
                any::<bool>(),
                prop_oneof![
                    select(vec![0i64, 1, -1, 2, -3, 12, -12, 100, -100]),
                    -40..40i64
                ],
                proptest::option::of(prop_oneof![select(vec![0u64, 1, 2, 12, 13]), 0..40u64]),
            )
        })
        .prop_map(|(t, r1, r2, s1, s2, scalar, start, len)| {
            (build(&t, &r1, s1), build(&t, &r2, s2), scalar, start, len)
        })
        .boxed()
}

/// A cast between a number type and a text type, either way
pub fn cast_text() -> BoxedStrategy<(ArrayRef, DataType, bool)> {
    use DataType as D;
    let numeric: Vec<D> = value_types()
        .into_iter()
        .filter(DataType::is_numeric)
        .chain([D::Int32, D::Int64])
        .collect();
    let text = vec![D::Utf8, D::LargeUtf8, D::Utf8View];
    (
        select(numeric),
        select(text),
        any::<bool>(),
        len(),
        any::<bool>(),
    )
        .prop_flat_map(|(num, txt, to_text, n, safe)| {
            let (from, to) = if to_text { (num, txt) } else { (txt, num) };
            let t = leaf(&from);
            (Just(t.clone()), rows(&t, n), seed(), Just(to), Just(safe))
        })
        .prop_map(|(t, r, s, to, safe)| (build(&t, &r, s), to, safe))
        .boxed()
}

/// A mask and two arrays of one type and the mask's length, and whether to
/// pass each array's first row as a scalar instead. Lengths favor 1024 and
/// above more than `len` does, where `zip` switches to `interleave`
pub fn zip() -> BoxedStrategy<(BooleanArray, ArrayRef, ArrayRef, bool, bool)> {
    let n = prop_oneof![3 => len(), 1 => select(vec![1023usize, 1024, 1025, 2048])];
    (all_types(), n)
        .prop_flat_map(|(t, n)| {
            (
                mask(n),
                rows(&t, n),
                rows(&t, n),
                seed(),
                seed(),
                any::<bool>(),
                any::<bool>(),
                Just(t),
            )
        })
        .prop_map(|(m, rt, rf, st, sf, ts, fs, t)| {
            (m.build(), build(&t, &rt, st), build(&t, &rf, sf), ts, fs)
        })
        .boxed()
}

/// A mask, a `truthy` array with a row for each true slot of the mask, and
/// a `falsy` array with a row for each false or null slot, or a one-row
/// scalar on either side
pub fn merge() -> BoxedStrategy<(BooleanArray, ArrayRef, ArrayRef, bool, bool)> {
    (all_types(), len())
        .prop_flat_map(|(t, n)| (Just(t), mask(n), any::<bool>(), any::<bool>()))
        .prop_flat_map(|(t, m, ts, fs)| {
            let trues = m.bits.iter().filter(|b| **b == Some(true)).count();
            let (nt, nf) = (
                if ts { 1 } else { trues },
                if fs { 1 } else { m.bits.len() - trues },
            );
            (
                rows(&t, nt),
                rows(&t, nf),
                seed(),
                seed(),
                Just((t, m, ts, fs)),
            )
        })
        .prop_map(|(rt, rf, st, sf, (t, m, ts, fs))| {
            (m.build(), build(&t, &rt, st), build(&t, &rf, sf), ts, fs)
        })
        .boxed()
}

/// Zero to four arrays of one type and indices into them, drawn row by row
/// or in runs, each array with the rows its indices use and sometimes a few
/// more
pub fn merge_n() -> BoxedStrategy<(Vec<ArrayRef>, Vec<Option<usize>>)> {
    (
        all_types(),
        prop_oneof![1 => Just(0usize), 30 => 1..=4usize],
    )
        .prop_flat_map(|(t, k)| {
            let pick = if k == 0 {
                Just(None).boxed()
            } else {
                prop_oneof![1 => Just(None), 4 => (0..k).prop_map(Some)].boxed()
            };
            let p = pick.clone();
            let rowwise = len().prop_flat_map(move |n| vec(p.clone(), n)).boxed();
            let runs = vec((pick, 1..=70usize), 0..=12)
                .prop_map(|rs| rs.into_iter().flat_map(|(i, n)| vec![i; n]).collect())
                .boxed();
            (Just(t), Just(k), prop_oneof![rowwise, runs])
        })
        .prop_flat_map(|(t, k, idx): (GeneratorType, usize, Vec<Option<usize>>)| {
            let parts: Vec<_> = (0..k)
                .map(|j| {
                    let used = idx.iter().filter(|i| **i == Some(j)).count();
                    let t = t.clone();
                    prop_oneof![3 => Just(0usize), 1 => 1..=3usize]
                        .prop_flat_map(move |extra| (rows(&t, used + extra), seed()))
                })
                .collect();
            (Just(t), parts, Just(idx))
        })
        .prop_map(|(t, parts, idx)| {
            let arrays = parts.iter().map(|(r, s)| build(&t, r, *s)).collect();
            (arrays, idx)
        })
        .boxed()
}

/// An array of any type
pub fn any_array() -> BoxedStrategy<ArrayRef> {
    array_of(all_types()).prop_map(|g| g.build()).boxed()
}

/// Arrays of the types `neg` documents: signed and unsigned integers (an
/// error for unsigned), floats, decimals, durations and intervals
pub fn neg() -> BoxedStrategy<ArrayRef> {
    use DataType as D;
    let types: Vec<GeneratorType> = value_types()
        .into_iter()
        .chain([D::Int32, D::Int64])
        .filter(|d| d.is_numeric() || matches!(d, D::Duration(_) | D::Interval(_)))
        .map(|d| leaf(&d))
        .collect();
    array_of(select(types).boxed())
        .prop_map(|g| g.build())
        .boxed()
}

/// Arrays the aggregate kernels take: integers, floats and decimals,
/// booleans, strings and binary of every layout, and dictionaries and
/// run-end encoded arrays of Int32 and Int64
pub fn aggregate() -> BoxedStrategy<ArrayRef> {
    use DataType as D;
    let numeric: Vec<GeneratorType> = value_types()
        .into_iter()
        .chain([D::Int32, D::Int64])
        .filter(DataType::is_numeric)
        .map(|d| leaf(&d))
        .collect();
    let ints = || prop_oneof![Just(GeneratorType::I32), Just(GeneratorType::I64)];
    let types = prop_oneof![
        6 => select(numeric),
        1 => Just(GeneratorType::Bool),
        3 => prop_oneof![Just(GeneratorType::Utf8), Just(GeneratorType::LargeUtf8), Just(GeneratorType::Utf8View), Just(GeneratorType::Binary), Just(GeneratorType::LargeBinary), Just(GeneratorType::BinaryView), (0..=4i32).prop_map(GeneratorType::Fsb)],
        1 => (select(vec![D::Int8, D::UInt16, D::Int32]), ints()).prop_map(|(k, v)| GeneratorType::Dict(k, Box::new(v))),
        1 => (select(vec![D::Int16, D::Int32, D::Int64]), ints()).prop_map(|(r, v)| GeneratorType::Ree(r, Box::new(v))),
    ];
    array_of(types.boxed()).prop_map(|g| g.build()).boxed()
}

/// One to six columns of any types and one length, sort options for each,
/// and a row count: a limit for `lexsort`, often at most a tenth of the rows
/// where it uses a heap, or a split point for converting rows in two parts
pub fn sort_columns() -> BoxedStrategy<SortCase> {
    let opts = (any::<bool>(), any::<bool>()).prop_map(|(descending, nulls_first)| SortOptions {
        descending,
        nulls_first,
    });
    (vec((all_types(), opts), 1..=6), len())
        .prop_flat_map(|(cols, n)| {
            let built: Vec<_> = cols.iter().map(|(t, _)| (rows(t, n), seed())).collect();
            let opts: Vec<_> = cols.iter().map(|(_, o)| *o).collect();
            let types: Vec<GeneratorType> = cols.into_iter().map(|(t, _)| t).collect();
            let count = prop_oneof![1 => Just(None), 2 => (0..=n / 10).prop_map(Some), 1 => (0..=n + 2).prop_map(Some)];
            (Just(types), built, Just(opts), count)
        })
        .prop_map(|(types, built, opts, count)| {
            let cols = types.iter().zip(built).map(|(t, (r, s))| build(t, &r, s)).collect();
            (cols, opts, count)
        })
        .boxed()
}

/// A schema of one to four columns of any types, zero to six batches of it
/// with their own lengths and layouts, each pushed plainly, through a
/// filter, through take indices, or as `(batch, row)` pairs into all the
/// batches, a target batch size, and sometimes a size above which batches
/// bypass coalescing
pub fn coalesce() -> BoxedStrategy<CoalesceCase> {
    use crate::batches::Push;
    use arrow_schema::{Field, Schema};
    let target = prop_oneof![2 => select(vec![1usize, 2, 7, 8, 64, 1024, 4096]), 1 => 1..=300usize];
    let biggest = prop_oneof![3 => Just(None), 1 => (0..=600usize).prop_map(Some)];
    // Columns of any types, or only flat ones, since one nested column
    // changes how the coalescer handles the whole batch
    let types =
        prop_oneof![3 => vec(all_types(), 1..=4), 2 => vec(crate::generator::leaf(), 1..=4)];
    // The bypass applies to batches "larger than this limit", so sometimes
    // the limit is exactly the rows one push selects
    let at_batch = prop_oneof![3 => Just(None), 1 => any::<Index>().prop_map(Some)];
    (types, 0..=6usize, target, biggest, at_batch)
        .prop_flat_map(|(types, k, target, biggest, at_batch)| {
            let t2 = types.clone();
            let batch = len().prop_flat_map(move |n| {
                let cols: Vec<_> = t2.iter().map(|t| (rows(t, n), seed())).collect();
                let push = prop_oneof![
                    2 => Just(Push::Plain),
                    1 => mask(n).prop_map(|m| Push::Filter(m.build())),
                    1 => mask(n).prop_map(|m| Push::FilterOptimized(m.build())),
                    1 => mask(n).prop_map(|m| Push::FilterCounted(m.build())),
                    2 => indices(n).prop_map(|i| Push::Indices(i.build())),
                    // The pairs are drawn once every batch's length is known
                    1 => Just(Push::Interleaved(vec![])),
                ];
                (cols, push)
            });
            (
                Just(types),
                vec(batch, k),
                Just(target),
                Just(biggest),
                Just(at_batch),
            )
        })
        .prop_map(|(types, batches, target, biggest, at_batch)| {
            let schema = Arc::new(Schema::new(
                types
                    .iter()
                    .enumerate()
                    .map(|(i, t)| Field::new(format!("c{i}"), t.data_type(), true))
                    .collect::<Vec<_>>(),
            ));
            let batches = batches
                .into_iter()
                .map(|(cols, push)| {
                    let cols: Vec<ArrayRef> = types
                        .iter()
                        .zip(cols)
                        .map(|(t, (r, sd))| build(t, &r, sd))
                        .collect();
                    let n = cols[0].len();
                    (record_batch(schema.clone(), cols, n), push)
                })
                .collect::<Vec<_>>();
            (schema, batches, target, biggest, at_batch)
        })
        .prop_flat_map(|(schema, batches, target, biggest, at_batch)| {
            let lens: Vec<usize> = batches.iter().map(|(b, _)| b.num_rows()).collect();
            let batches: Vec<_> = batches
                .into_iter()
                .map(|(b, push)| match push {
                    Push::Interleaved(_) => pairs(lens.clone())
                        .prop_map(move |p| (b.clone(), Push::Interleaved(p)))
                        .boxed(),
                    push => Just((b, push)).boxed(),
                })
                .collect();
            (
                Just(schema),
                batches,
                Just(target),
                Just(biggest),
                Just(at_batch),
            )
        })
        .prop_map(|(schema, batches, target, mut biggest, at_batch)| {
            if let Some(i) = at_batch
                && !batches.is_empty()
            {
                let (b, push) = &batches[i.index(batches.len())];
                biggest = Some(match push {
                    Push::Plain => b.num_rows(),
                    Push::Filter(m) | Push::FilterOptimized(m) | Push::FilterCounted(m) => (0..m
                        .len())
                        .filter(|&i| m.is_valid(i) && m.value(i))
                        .count(),
                    Push::Indices(i) => i.len(),
                    Push::Interleaved(p) => p.len(),
                });
            }
            (schema, batches, target, biggest)
        })
        .boxed()
}

/// Two temporal arrays of one length that `add` or `sub` accept together,
/// an operation, and whether each side is a one-row scalar: timestamps
/// with durations, timestamps, and intervals, durations with durations,
/// dates with dates and intervals, intervals with intervals, and an interval
/// or duration on the left of `add`
pub fn temporal_arith() -> BoxedStrategy<TemporalCase> {
    use arrow_schema::{DataType as D, IntervalUnit as I, TimeUnit as U};
    let unit = || {
        select(vec![
            U::Second,
            U::Millisecond,
            U::Microsecond,
            U::Nanosecond,
        ])
    };
    let tz = || {
        select(vec![
            None,
            Some(Arc::<str>::from("+05:30")),
            Some(Arc::<str>::from("UTC")),
            Some(Arc::<str>::from("-03:00")),
        ])
    };
    let interval = || {
        select(vec![
            D::Interval(I::YearMonth),
            D::Interval(I::DayTime),
            D::Interval(I::MonthDayNano),
        ])
    };
    let date = || select(vec![D::Date32, D::Date64]);
    let ts = move || (unit(), tz()).prop_map(|(u, z)| D::Timestamp(u, z));
    // (left, right, operations allowed: 0 add, 1 sub, 2 add_wrapping, 3 sub_wrapping)
    let any_op = [0u8, 1, 2, 3];
    let pairs = prop_oneof![
        (unit(), tz()).prop_map(|(u, z)| (D::Timestamp(u, z), D::Duration(u))),
        (unit(), tz(), tz()).prop_map(|(u, a, b)| (D::Timestamp(u, a), D::Timestamp(u, b))),
        unit().prop_map(|u| (D::Duration(u), D::Duration(u))),
        date().prop_map(|d| (d.clone(), d)),
        interval().prop_map(|i| (i.clone(), i)),
        (ts(), interval()),
        (date(), interval()),
        (interval(), prop_oneof![ts(), date()]),
        (unit(), tz()).prop_map(|(u, z)| (D::Duration(u), D::Timestamp(u, z))),
    ];
    (pairs, len(), any::<bool>(), any::<bool>())
        .prop_flat_map(move |((lt, rt), n, ls, rs)| {
            let sub_only = matches!(
                (&lt, &rt),
                (D::Timestamp(..), D::Timestamp(..))
                    | (D::Date32, D::Date32)
                    | (D::Date64, D::Date64)
            );
            let add_only = matches!(lt, D::Interval(_) | D::Duration(_))
                && matches!(rt, D::Timestamp(..) | D::Date32 | D::Date64);
            let ops: Vec<u8> = any_op
                .iter()
                .copied()
                .filter(|o| {
                    if sub_only {
                        o % 2 == 1
                    } else if add_only {
                        o % 2 == 0
                    } else {
                        true
                    }
                })
                .collect();
            let (tl, tr) = (GeneratorType::Prim(lt), GeneratorType::Prim(rt));
            (
                Just(tl.clone()),
                Just(tr.clone()),
                rows(&tl, n),
                rows(&tr, n),
                seed(),
                seed(),
                select(ops),
                Just(ls),
                Just(rs),
            )
        })
        .prop_map(|(tl, tr, rl, rr, sl, sr, op, ls, rs)| {
            (build(&tl, &rl, sl), build(&tr, &rr, sr), op, ls, rs)
        })
        .boxed()
}

/// A temporal array or its backing integer, a temporal cast target, and
/// whether the cast is `safe`
pub fn temporal_cast() -> BoxedStrategy<TemporalCastCase> {
    use arrow_schema::{DataType as D, TimeUnit as U};
    let units = vec![U::Second, U::Millisecond, U::Microsecond, U::Nanosecond];
    let tzs = vec![
        None,
        Some(Arc::<str>::from("+05:30")),
        Some(Arc::<str>::from("UTC")),
        Some(Arc::<str>::from("-03:00")),
    ];
    let ts = {
        let (units, tzs) = (units.clone(), tzs.clone());
        move || (select(units.clone()), select(tzs.clone())).prop_map(|(u, z)| D::Timestamp(u, z))
    };
    let ts_none = {
        let units = units.clone();
        move || select(units.clone()).prop_map(|u| D::Timestamp(u, None))
    };
    let time = || {
        select(vec![
            D::Time32(U::Second),
            D::Time32(U::Millisecond),
            D::Time64(U::Microsecond),
            D::Time64(U::Nanosecond),
        ])
    };
    let pairs = prop_oneof![
        (
            ts(),
            prop_oneof![
                ts().boxed(),
                Just(D::Date32).boxed(),
                Just(D::Date64).boxed(),
                time().boxed(),
                Just(D::Int64).boxed()
            ]
        ),
        (
            Just(D::Date32),
            prop_oneof![
                Just(D::Date64).boxed(),
                ts_none().boxed(),
                Just(D::Int32).boxed()
            ]
        ),
        (
            Just(D::Date64),
            prop_oneof![
                Just(D::Date32).boxed(),
                ts_none().boxed(),
                Just(D::Int64).boxed()
            ]
        ),
        time().prop_flat_map(move |t| {
            let backing = if t.primitive_width() == Some(4) {
                D::Int32
            } else {
                D::Int64
            };
            (Just(t), prop_oneof![time().boxed(), Just(backing).boxed()])
        }),
        (
            Just(D::Int32),
            prop_oneof![
                Just(D::Date32),
                Just(D::Time32(U::Second)),
                Just(D::Time32(U::Millisecond))
            ]
        ),
        (
            Just(D::Int64),
            prop_oneof![
                Just(D::Date64).boxed(),
                Just(D::Time64(U::Microsecond)).boxed(),
                Just(D::Time64(U::Nanosecond)).boxed(),
                ts().boxed()
            ]
        ),
    ];
    (pairs, len(), any::<bool>())
        .prop_flat_map(|((from, to), n, safe)| {
            let t = leaf(&from);
            (Just(t.clone()), rows(&t, n), seed(), Just(to), Just(safe))
        })
        .prop_map(|(t, r, s, to, safe)| (build(&t, &r, s), to, safe))
        .boxed()
}

/// Two Decimal128 arrays of one length and a required scale: above the
/// product scale (an error), at it, a few digits below, anywhere in the
/// scale range, or any i8, which the documentation does not restrict
pub fn fixed_point() -> BoxedStrategy<FixedPointCase> {
    use DataType as D;
    let types: Vec<D> = value_types()
        .into_iter()
        .filter(|d| matches!(d, D::Decimal128(..)))
        .collect();
    (select(types.clone()), select(types), len())
        .prop_flat_map(|(lt, rt, n)| {
            let scale = |d: &D| match d {
                D::Decimal128(_, s) => *s as i32,
                _ => unreachable!(),
            };
            let ps = scale(&lt) + scale(&rt);
            let req = prop_oneof![
                1 => Just(ps + 1),
                2 => Just(ps),
                4 => (ps - 8)..ps,
                2 => -38..=38i32,
                1 => any::<i8>().prop_map(i32::from),
            ]
            .prop_map(|x| x.clamp(i8::MIN as i32, i8::MAX as i32) as i8);
            let (tl, tr) = (GeneratorType::Prim(lt), GeneratorType::Prim(rt));
            (
                Just(tl.clone()),
                Just(tr.clone()),
                rows(&tl, n),
                rows(&tr, n),
                seed(),
                seed(),
                req,
            )
        })
        .prop_map(|(tl, tr, rl, rr, sl, sr, req)| (build(&tl, &rl, sl), build(&tr, &rr, sr), req))
        .boxed()
}

/// A decimal array of every width, of a precision and scale drawn for it,
/// negative scales included, holding values of the widest type of that width
/// so that some lie past the precision. Then a precision to check the values
/// against, and a precision and scale to set, each past its limits too
pub fn decimal_precision() -> BoxedStrategy<(ArrayRef, u8, u8, i8)> {
    use DataType as D;
    let widths = vec![
        (D::Decimal32(9, 0), 9u8),
        (D::Decimal64(18, 0), 18),
        (D::Decimal128(38, 0), 38),
        (D::Decimal256(76, 0), 76),
    ];
    (select(widths), len())
        .prop_flat_map(|((wide, max), n)| {
            let p0 = prop_oneof![select(vec![1, max]), 1..=max];
            let rows = array(&GeneratorType::Prim(wide.clone()), n, ValueDomain::Arrow);
            (Just(wide), Just(max), p0, rows)
        })
        .prop_flat_map(|(wide, max, p0, g)| {
            let m = max as i8;
            let top = p0 as i8;
            // The Arrow format bounds a scale only by its `int` type, and
            // arrow-rs only from above
            let s0 = prop_oneof![select(vec![0, -1, top, -m, i8::MIN]), -m..=top];
            let p = prop_oneof![select(vec![1, max, p0]), 1..=max];
            let p1 = prop_oneof![select(vec![0, 1, max, max + 1]), 0..=max + 1];
            let s1 = prop_oneof![select(vec![i8::MIN, -m, -1, 0, m, m + 1]), any::<i8>()];
            (Just(wide), Just(p0), s0, Just(g), p, p1, s1)
        })
        .prop_map(|(wide, p0, s0, g, p, p1, s1)| {
            let dt = match wide {
                D::Decimal32(..) => D::Decimal32(p0, s0),
                D::Decimal64(..) => D::Decimal64(p0, s0),
                D::Decimal128(..) => D::Decimal128(p0, s0),
                _ => D::Decimal256(p0, s0),
            };
            // Only the type changes, which keeps the layout `build` chose
            let data = g.build().to_data().into_builder().data_type(dt).build();
            (
                make_array(data.expect("a decimal of the same width")),
                p,
                p1,
                s1,
            )
        })
        .boxed()
}

/// Decimal arrays of every width, precision and scale the generator knows
pub fn decimal() -> BoxedStrategy<ArrayRef> {
    let types: Vec<GeneratorType> = value_types()
        .into_iter()
        .filter(DataType::is_decimal)
        .map(GeneratorType::Prim)
        .collect();
    array_of(select(types).boxed())
        .prop_map(|g| g.build())
        .boxed()
}

/// A take case plus indices into its output
pub fn take_take() -> BoxedStrategy<(ArrayRef, Int32Array, Int32Array)> {
    take_in_bounds()
        .prop_flat_map(|(a, i)| {
            let n = i.len();
            (Just(a), Just(i), indices(n))
        })
        .prop_map(|(a, i, j)| (a, i, j.build()))
        .boxed()
}

/// A slice case plus a slice within it
pub fn slice_slice() -> BoxedStrategy<(ArrayRef, usize, usize, usize, usize)> {
    slice()
        .prop_flat_map(|(a, o1, l1)| (Just(a), Just(o1), Just(l1), 0..=l1))
        .prop_flat_map(|(a, o1, l1, o2)| (Just(a), Just(o1), Just(l1), Just(o2), 0..=l1 - o2))
        .boxed()
}

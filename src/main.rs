//! Long runs and reports. Usage:
//!
//! - `arrow-pbt run <properties|all> <seeds> <cases> [shrink]` prints one TSV
//!   row per property and seed: property, seed, failed, cases to the
//!   failure, cases run, milliseconds, message, minimal input, discards.
//! - `arrow-pbt list` prints the name of every property.
//! - `arrow-pbt coverage <seed> <cases>` reports the share of generated
//!   arrays, masks and indices that show each feature.
//! - `arrow-pbt inventory <seed> <cases>` writes Markdown tables of the type
//!   families, layout features and kernel paths each property's inputs reach.
//!
//! `ARROW_PBT_THREADS` sets the workers (default: the available parallelism),
//! and `ARROW_PBT_PANICS=1` keeps panic messages. `ARROW_PBT_REASONS`,
//! `ARROW_PBT_FULL` and `ARROW_PBT_NO_KNOWN` work as in `arrow_pbt::run`.

use arrow_pbt::registry::ARITH_ROWS;
use arrow_pbt::run::{env, seeded_runner, seeds};
use arrow_pbt::{cases, coverage, generator, inventory, registry};
use std::env::var_os;
use std::num::NonZero;
use std::panic::set_hook;
use std::process::ExitCode;
use std::str::FromStr;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::thread::available_parallelism;

const USAGE: &str = "usage: arrow-pbt run <properties|all> <seeds> <cases> [shrink] | list | coverage <seed> <cases> | inventory <seed> <cases>";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    match cli(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}\n{USAGE}");
            ExitCode::FAILURE
        }
    }
}

/// Argument `i`, parsed, or an error that names it
fn arg<T: FromStr>(args: &[String], i: usize, name: &str) -> Result<T, String> {
    let v = args.get(i).ok_or_else(|| format!("missing <{name}>"))?;
    v.parse().map_err(|_| format!("<{name}> {v} is not valid"))
}

fn cli(args: &[String]) -> Result<(), String> {
    // Without a count from the OS, run on one thread rather than fail
    let default = available_parallelism().map_or(1, NonZero::get);
    let threads: usize = env("ARROW_PBT_THREADS", default)?;
    // Kernel panics are caught and reported per case. ARROW_PBT_PANICS=1
    // keeps the messages, for panics outside a property such as in the
    // generator
    if var_os("ARROW_PBT_PANICS").is_none() {
        set_hook(Box::new(|_| {}));
    }
    match args.get(1).map(String::as_str) {
        Some("run") => {
            let props = registry::properties(&arg::<String>(args, 2, "properties")?)
                .map_err(|e| format!("{e}. `arrow-pbt list` prints the names"))?;
            let seeds = seeds(&arg::<String>(args, 3, "seeds")?)?;
            let cases: u32 = arg(args, 4, "cases")?;
            let shrink: u32 = if args.len() > 5 {
                arg(args, 5, "shrink")?
            } else {
                1000
            };
            for (p, seed, o) in registry::run_jobs(&props, &seeds, cases, shrink, threads) {
                println!(
                    "{p}\t{seed}\t{}\t{}\t{}\t{:.0}\t{}\t{}\t{}",
                    o.failed,
                    o.cases_to_fail
                        .map_or_else(|| "-".into(), |c| c.to_string()),
                    o.execs,
                    o.ms,
                    o.message,
                    o.minimal,
                    o.discards,
                );
                if var_os("ARROW_PBT_REASONS").is_some() {
                    for (why, c) in &o.reasons {
                        println!("  discard\t{p}\t{seed}\t{c}\t{why}");
                    }
                }
            }
            if var_os("ARROW_PBT_REASONS").is_some() {
                let [seen, compared] = &ARITH_ROWS;
                let (seen, compared) = (
                    seen.load(Ordering::Relaxed),
                    compared.load(Ordering::Relaxed),
                );
                if seen > 0 {
                    println!(
                        "  arith: {compared} of {seen} non-null integer and decimal rows compared"
                    );
                }
            }
        }
        Some("list") => {
            for e in registry::all_properties() {
                println!("{}", e.name);
            }
        }
        Some("help" | "--help" | "-h") => println!("{USAGE}"),
        Some("coverage") => coverage_run(arg(args, 2, "seed")?, arg(args, 3, "cases")?),
        Some("inventory") => inventory_run(arg(args, 2, "seed")?, arg(args, 3, "cases")?, threads),
        Some(c) => return Err(format!("unknown command {c}")),
        None => return Err("missing command".into()),
    }
    Ok(())
}

/// Reports the share of generated arrays, masks and index arrays that
/// exhibit each feature
fn coverage_run(seed: u64, n: usize) {
    use proptest::strategy::{Strategy, ValueTree};
    use proptest::test_runner::Config;
    use std::collections::{BTreeMap, BTreeSet};
    let mut runner = seeded_runner(Config::default(), seed);
    let tally = |counts: &mut BTreeMap<&'static str, usize>, f: BTreeSet<&'static str>| {
        for k in f {
            *counts.entry(k).or_default() += 1;
        }
    };
    let arrays = generator::array_of(generator::all_types());
    let mut counts = BTreeMap::new();
    for _ in 0..n {
        let g = arrays.new_tree(&mut runner).unwrap().current();
        let mut f = BTreeSet::new();
        coverage::features(g.build().as_ref(), &mut f);
        tally(&mut counts, f);
    }
    coverage::report("arrays", n, &counts, coverage::EXPECTED);
    println!(
        "  {} arrays had a values buffer 8-byte but not 16-byte aligned before FFI import",
        generator::MISALIGNED.load(Ordering::Relaxed)
    );
    let numeric = generator::array_of(generator::numeric_types());
    let mut counts = BTreeMap::new();
    for _ in 0..n {
        let g = numeric.new_tree(&mut runner).unwrap().current();
        let mut f = BTreeSet::new();
        coverage::numeric_features(g.build().as_ref(), &mut f);
        tally(&mut counts, f);
    }
    coverage::report(
        "integer, decimal and float arrays",
        n,
        &counts,
        coverage::NUMERIC,
    );
    let masks = cases::filter();
    let mut counts = BTreeMap::new();
    for _ in 0..n {
        let (_, p) = masks.new_tree(&mut runner).unwrap().current();
        let mut f = BTreeSet::new();
        coverage::mask_features(&p, &mut f);
        tally(&mut counts, f);
    }
    coverage::report(
        "filter predicates",
        n,
        &counts,
        &[
            "mask: all selected",
            "mask: none selected",
            "mask: selectivity in (0.8, 1)",
            "mask: 1 row to 1/16 of rows selected",
            "mask: has nulls",
            "mask: a whole 64-bit word selected",
            "mask: two words keeping exactly 64 bits",
        ],
    );
    let takes = cases::take_in_bounds();
    let mut counts = BTreeMap::new();
    for _ in 0..n {
        let (a, idx) = takes.new_tree(&mut runner).unwrap().current();
        let mut f = BTreeSet::new();
        coverage::index_features(&idx, a.len(), &mut f);
        tally(&mut counts, f);
    }
    coverage::report(
        "take indices",
        n,
        &counts,
        &[
            "indices: empty",
            "indices: null hiding an out-of-bounds value",
            "indices: repeated in a row",
            "indices: consecutive run",
        ],
    );
}

/// Writes the type family and layout feature matrices over every case
/// strategy, and the paths `zip`, `lexsort` and `BatchCoalescer` take
fn inventory_run(seed: u64, n: usize, threads: usize) {
    // One sample per case strategy, labelled by the properties that use it
    let mut strategies: Vec<(&'static str, Vec<&'static str>, registry::Entry)> = vec![];
    for e in registry::all_properties() {
        match strategies.iter_mut().find(|(c, _, _)| *c == e.cases) {
            Some((_, names, _)) => names.push(e.name),
            None => strategies.push((e.cases, vec![e.name], e)),
        }
    }
    // One strategy per worker, as for property runs. The three path tallies
    // take a worker each
    let next = AtomicUsize::new(0);
    let done = Mutex::new(vec![]);
    let (mut zip_paths, mut lexsort_paths, mut coalesce_paths) =
        (Default::default(), Default::default(), Default::default());
    std::thread::scope(|s| {
        s.spawn(|| zip_paths = inventory::zip_paths(cases::zip(), n, seed));
        s.spawn(|| lexsort_paths = inventory::lexsort_paths(cases::sort_columns(), n, seed));
        s.spawn(|| coalesce_paths = inventory::coalesce_paths(cases::coalesce(), n, seed));
        for _ in 0..threads.saturating_sub(3).max(1) {
            s.spawn(|| {
                loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    let Some((name, _, e)) = strategies.get(k) else {
                        break;
                    };
                    let sample = e.sample(n, seed);
                    done.lock().unwrap().push((k, *name, sample));
                }
            });
        }
    });
    let mut done = done.into_inner().unwrap();
    done.sort_by_key(|d| d.0);
    let samples: Vec<(&'static str, inventory::Sample)> =
        done.into_iter().map(|(_, p, s)| (p, s)).collect();
    println!(
        "# Inventory\n\nShare of {n} sampled cases per case strategy whose input arrays contain each type family or exhibit each layout feature, anywhere in their nesting. Blank means never.\n\n| Case strategy | Properties |\n|---|---|"
    );
    for (c, names, _) in &strategies {
        println!("| {c} | {} |", names.join(", "));
    }
    println!();
    print!(
        "{}",
        inventory::table("Type families", inventory::FAMILIES, &samples, |s| {
            &s.families
        })
    );
    println!();
    print!(
        "{}",
        inventory::table("Layout features", coverage::EXPECTED, &samples, |s| {
            &s.features
        })
    );
    for (title, paths) in [
        ("zip", zip_paths),
        ("coalesce", coalesce_paths),
        ("lexsort", lexsort_paths),
    ] {
        println!("\n## {title} paths\n\n| Path | Cases of {n} |\n|---|---|");
        for (k, c) in paths {
            println!("| {k} | {c} |");
        }
    }
}

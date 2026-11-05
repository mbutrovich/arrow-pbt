//! The runner every property in this repository goes through, arrow-rs's
//! and the adapters'. [`check`] reads its settings from the environment:
//!
//! - `ARROW_PBT_CASES`: cases per seed, 256 by default
//! - `ARROW_PBT_SEEDS`: a seed, or a range such as `1-10`, 1 by default
//! - `ARROW_PBT_SHRINK`: shrink steps after a failure, 1000 by default
//! - `ARROW_PBT_REASONS`: when set, print the discards by reason
//! - `ARROW_PBT_FULL`: when set, print messages and inputs without truncation
//! - `ARROW_PBT_NO_KNOWN`: when set, fail instead of discarding known bugs
//!   (see [`crate::known`])
//!
//! A value that does not parse is an error rather than a fallback to the
//! default. The same seed and case count always give the same cases, and a
//! failure prints the command that reruns it with these settings.

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::env::{VarError, var_os};
use std::fmt::Debug;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::str::FromStr;
use std::time::Instant;

use proptest::strategy::Strategy;
use proptest::test_runner::{Config, RngAlgorithm, TestCaseError, TestError, TestRng, TestRunner};

/// Runs `test` on cases from `strategy` for each seed in `ARROW_PBT_SEEDS`,
/// and panics on the first failure with its message, the shrunk input, and
/// the settings that rerun it. A test that panics fails with the panic's
/// message. A run that discards too many cases fails too, since it checked
/// too little
///
/// # Example
///
/// ```
/// use arrow_pbt::generator::{GeneratorType, ValueDomain, array};
/// use arrow_pbt::run::check;
/// use arrow_pbt::val::check_rows;
/// check("built", array(&GeneratorType::Utf8, 10, ValueDomain::Arrow), |g| {
///     check_rows(&g.build(), &g.ty.data_type(), &g.rows)
/// });
/// ```
pub fn check<T, S, F>(name: &str, strategy: S, test: F)
where
    T: Debug + Clone,
    S: Strategy<Value = T>,
    F: Fn(T) -> Result<(), TestCaseError>,
{
    check_with(name, |seed, cases, shrink| {
        run(&strategy, &test, seed, cases, shrink)
    });
}

/// `check` for a runner given as a function of the seed, the cases and the
/// shrink steps
#[doc(hidden)]
pub fn check_with(name: &str, run_seed: impl Fn(u64, u32, u32) -> Outcome) {
    let settings = || -> Result<_, String> {
        Ok((
            env("ARROW_PBT_CASES", 256)?,
            env("ARROW_PBT_SHRINK", 1000)?,
            seeds(&env("ARROW_PBT_SEEDS", "1".to_string())?)?,
        ))
    };
    let (cases, shrink, seed_list) = settings().unwrap_or_else(|e| panic!("{e}"));
    // Cargo sets the package name when it runs a test
    let package = std::env::var("CARGO_PKG_NAME").map_or(String::new(), |p| format!(" -p {p}"));
    for seed in seed_list {
        let o = run_seed(seed, cases, shrink);
        if var_os("ARROW_PBT_REASONS").is_some() {
            for (why, c) in &o.reasons {
                eprintln!("{name} seed {seed}: {c} discarded: {why}");
            }
        }
        assert!(
            !o.failed,
            "{name} failed with seed {seed} after {} cases: {}\nminimal input: {}\nrerun with: {}",
            o.cases_to_fail.unwrap_or(o.execs),
            o.message,
            o.minimal,
            rerun(name, seed, cases, &package, |v| {
                var_os(v).map(|x| x.to_string_lossy().into_owned())
            })
        );
    }
}

/// The command that reruns `name` on `seed`, with each setting that is set
/// and changes the failure or what it prints. `var` reads a variable
fn rerun(
    name: &str,
    seed: u64,
    cases: u32,
    package: &str,
    var: impl Fn(&str) -> Option<String>,
) -> String {
    let mut settings = vec![
        format!("ARROW_PBT_SEEDS={seed}"),
        format!("ARROW_PBT_CASES={cases}"),
    ];
    settings.extend(
        ["ARROW_PBT_SHRINK", "ARROW_PBT_NO_KNOWN", "ARROW_PBT_FULL"]
            .into_iter()
            .filter_map(|v| var(v).map(|x| format!("{v}={x}"))),
    );
    format!("{} cargo test{package} {name}", settings.join(" "))
}

/// The result of running one property for one seed
#[doc(hidden)]
pub struct Outcome {
    pub failed: bool,
    pub cases_to_fail: Option<usize>,
    pub execs: usize,
    pub message: String,
    pub minimal: String,
    pub ms: f64,
    /// Cases discarded: inputs the property could not use
    pub discards: usize,
    /// Discards by reason
    pub reasons: BTreeMap<String, usize>,
}

/// Runs `test` on `cases` inputs from `strategy`, with a fixed seed and
/// shrink budget, catching panics as failures
#[doc(hidden)]
pub fn run<T, S, F>(strategy: &S, test: &F, seed: u64, cases: u32, max_shrink_iters: u32) -> Outcome
where
    T: Debug + Clone,
    S: Strategy<Value = T>,
    F: Fn(T) -> Result<(), TestCaseError>,
{
    let config = Config {
        cases,
        max_shrink_iters,
        failure_persistence: None,
        max_global_rejects: cases * 10,
        max_local_rejects: cases * 10,
        ..Config::default()
    };
    let mut runner = seeded_runner(config, seed);
    let execs = Cell::new(0usize);
    let discards = Cell::new(0usize);
    let reasons = RefCell::new(BTreeMap::new());
    let first = Cell::new(None);
    let t0 = Instant::now();
    let result = runner.run(strategy, |input| {
        execs.set(execs.get() + 1);
        let r = caught(|| test(input));
        if let Err(TestCaseError::Reject(why)) = &r {
            discards.set(discards.get() + 1);
            *reasons.borrow_mut().entry(why.to_string()).or_default() += 1;
        }
        if matches!(r, Err(TestCaseError::Fail(_))) && first.get().is_none() {
            first.set(Some(execs.get()));
        }
        r
    });
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let short = |s: String| -> String {
        let s: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
        // ARROW_PBT_FULL=1 keeps the whole message and input
        if var_os("ARROW_PBT_FULL").is_some() {
            s
        } else {
            s.chars().take(400).collect()
        }
    };
    let (failed, message, minimal) = match result {
        Ok(()) => (false, String::new(), String::new()),
        Err(TestError::Fail(reason, value)) => {
            (true, short(reason.to_string()), short(format!("{value:?}")))
        }
        Err(TestError::Abort(reason)) => (true, short(format!("aborted: {reason}")), String::new()),
    };
    Outcome {
        failed,
        cases_to_fail: first.get(),
        execs: execs.get(),
        message,
        minimal,
        ms,
        discards: discards.get(),
        reasons: reasons.into_inner(),
    }
}

/// A runner whose random choices come only from `seed`, so a run replays
#[doc(hidden)]
pub fn seeded_runner(config: Config, seed: u64) -> TestRunner {
    let mut bytes = [0u8; 32];
    bytes[..8].copy_from_slice(&seed.to_le_bytes());
    TestRunner::new_with_rng(config, TestRng::from_seed(RngAlgorithm::ChaCha, &bytes))
}

/// The value of environment variable `name`, or `default` when it is unset.
/// A value that does not parse is an error, so a typo does not fall back to
/// the default unnoticed
#[doc(hidden)]
pub fn env<T: FromStr>(name: &str, default: T) -> Result<T, String> {
    match std::env::var(name) {
        Err(VarError::NotPresent) => Ok(default),
        Err(VarError::NotUnicode(_)) => Err(format!("{name} is not valid Unicode")),
        Ok(v) => v.parse().map_err(|_| format!("{name}={v} is not valid")),
    }
}

/// Seeds from `7` or `1-3`
#[doc(hidden)]
pub fn seeds(arg: &str) -> Result<Vec<u64>, String> {
    let parse = |s: &str| {
        s.parse::<u64>()
            .map_err(|_| format!("{arg} is not a seed or a range of seeds"))
    };
    match arg.split_once('-') {
        Some((a, b)) => Ok((parse(a)?..=parse(b)?).collect()),
        None => Ok(vec![parse(arg)?]),
    }
}

/// The message of a caught panic
pub(crate) fn panic_message(p: &(dyn Any + Send)) -> String {
    p.downcast_ref::<String>()
        .cloned()
        .or_else(|| p.downcast_ref::<&str>().map(|s| (*s).to_string()))
        .unwrap_or_default()
}

/// Runs `body`, turning a panic into a failure with its message, so a
/// known-bug filter can match it
///
/// # Example
///
/// ```
/// use arrow_pbt::run::caught;
/// use proptest::test_runner::TestCaseError;
/// let r = caught(|| panic!("index out of bounds"));
/// let Err(TestCaseError::Fail(m)) = r else { unreachable!() };
/// assert_eq!(m.to_string(), "panic: index out of bounds");
/// ```
pub fn caught(body: impl FnOnce() -> Result<(), TestCaseError>) -> Result<(), TestCaseError> {
    catch_unwind(AssertUnwindSafe(body)).unwrap_or_else(|p| {
        Err(TestCaseError::fail(format!(
            "panic: {}",
            panic_message(&*p)
        )))
    })
}

#[cfg(test)]
mod tests {
    use super::rerun;

    #[test]
    fn rerun_carries_the_switches_that_are_set() {
        let none = |_: &str| None;
        assert_eq!(
            rerun("take", 3, 256, " -p arrow-pbt", none),
            "ARROW_PBT_SEEDS=3 ARROW_PBT_CASES=256 cargo test -p arrow-pbt take"
        );
        let set = |v: &str| match v {
            "ARROW_PBT_NO_KNOWN" => Some("1".to_string()),
            "ARROW_PBT_SHRINK" => Some("0".to_string()),
            // Only prints discards, so it does not change the failure
            "ARROW_PBT_REASONS" => Some("1".to_string()),
            _ => None,
        };
        assert_eq!(
            rerun("arity", 1, 10, "", set),
            "ARROW_PBT_SEEDS=1 ARROW_PBT_CASES=10 ARROW_PBT_SHRINK=0 ARROW_PBT_NO_KNOWN=1 cargo test arity"
        );
    }
}

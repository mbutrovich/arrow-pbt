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
//! - `ARROW_PBT_THREADS`: the most seeds that run at once across every check
//!   in the process, the available parallelism by default
//!
//! A value that does not parse is an error rather than a fallback to the
//! default. The same seed and case count always give the same cases, and a
//! failure prints the command that reruns it with these settings.

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::env::{VarError, var_os};
use std::fmt::Debug;
use std::num::NonZero;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock, PoisonError};
use std::thread::{available_parallelism, scope};
use std::time::Instant;

use proptest::strategy::Strategy;
use proptest::test_runner::{Config, RngAlgorithm, TestCaseError, TestError, TestRng, TestRunner};

/// Runs `test` on cases from the strategy `strategy` builds, for each seed
/// in `ARROW_PBT_SEEDS`, and panics on the lowest failing seed with its
/// message, the shrunk input, and the settings that rerun it. Seeds run in
/// parallel, so each worker builds its own strategy and `test` must be
/// `Sync`. A test that panics fails with the panic's message. A run that
/// discards too many cases fails too, since it checked too little
///
/// # Example
///
/// ```
/// use arrow_pbt::generator::{GeneratorType, ValueDomain, array};
/// use arrow_pbt::run::check;
/// use arrow_pbt::val::check_rows;
/// check("built", || array(&GeneratorType::Utf8, 10, ValueDomain::Arrow), |g| {
///     check_rows(&g.build(), &g.ty.data_type(), &g.rows)
/// });
/// ```
pub fn check<T, S, F>(name: &str, strategy: impl Fn() -> S + Sync, test: F)
where
    T: Debug + Clone,
    S: Strategy<Value = T>,
    F: Fn(T) -> Result<(), TestCaseError> + Sync,
{
    check_with(name, |seed, cases, shrink| {
        run(&strategy(), &test, seed, cases, shrink)
    });
}

/// `check` for a runner given as a function of the seed, the cases and the
/// shrink steps
#[doc(hidden)]
pub fn check_with(name: &str, run_seed: impl Fn(u64, u32, u32) -> Outcome + Sync) {
    let settings = || -> Result<_, String> {
        Ok((
            env("ARROW_PBT_CASES", 256)?,
            env("ARROW_PBT_SHRINK", 1000)?,
            seeds(&env("ARROW_PBT_SEEDS", "1".to_string())?)?,
        ))
    };
    let (cases, shrink, seed_list) = settings().unwrap_or_else(|e| panic!("{e}"));
    check_seeds(name, &seed_list, cases, shrink, pool(), run_seed);
}

/// Runs `run_seed` for each of `seeds` on `pool`, and panics on the lowest
/// failing seed. Once a seed fails, no higher seed starts, so the failure
/// reported does not depend on which worker finished first
fn check_seeds(
    name: &str,
    seeds: &[u64],
    cases: u32,
    shrink: u32,
    pool: &Pool,
    run_seed: impl Fn(u64, u32, u32) -> Outcome + Sync,
) {
    let next = AtomicUsize::new(0);
    let lowest_failed = AtomicUsize::new(usize::MAX);
    let outcomes: Vec<OnceLock<Outcome>> = seeds.iter().map(|_| OnceLock::new()).collect();
    scope(|s| {
        for _ in 0..seeds.len().min(pool.size) {
            s.spawn(|| {
                loop {
                    // Seeds are taken in order, so every seed below the
                    // lowest failure has started
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= seeds.len() || i > lowest_failed.load(Ordering::Relaxed) {
                        break;
                    }
                    let o = pool.run(|| {
                        // A panic outside a case, such as in a strategy, is
                        // this seed's failure, with its message
                        catch_unwind(AssertUnwindSafe(|| run_seed(seeds[i], cases, shrink)))
                            .unwrap_or_else(|p| Outcome::panicked(&*p))
                    });
                    if o.failed {
                        lowest_failed.fetch_min(i, Ordering::Relaxed);
                    }
                    let _ = outcomes[i].set(o);
                }
            });
        }
    });
    // Cargo sets the package name when it runs a test
    let package = std::env::var("CARGO_PKG_NAME").map_or(String::new(), |p| format!(" -p {p}"));
    for (&seed, o) in seeds.iter().zip(outcomes) {
        // Seeds above the lowest failure did not run
        let Some(o) = o.into_inner() else { break };
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

/// A limit on the seeds running at once, shared by every check in the
/// process, so that tests running in parallel do not oversubscribe the cores
struct Pool {
    size: usize,
    free: Mutex<usize>,
    freed: Condvar,
}

impl Pool {
    fn new(size: usize) -> Self {
        Self {
            size,
            free: Mutex::new(size),
            freed: Condvar::new(),
        }
    }

    /// Runs `f` once a worker is free
    fn run<R>(&self, f: impl FnOnce() -> R) -> R {
        let mut free = self.free.lock().unwrap_or_else(PoisonError::into_inner);
        while *free == 0 {
            free = self
                .freed
                .wait(free)
                .unwrap_or_else(PoisonError::into_inner);
        }
        *free -= 1;
        drop(free);
        // Freed on drop, so a panic in `f` frees it too
        struct Taken<'a>(&'a Pool);
        impl Drop for Taken<'_> {
            fn drop(&mut self) {
                *self.0.free.lock().unwrap_or_else(PoisonError::into_inner) += 1;
                self.0.freed.notify_one();
            }
        }
        let _taken = Taken(self);
        f()
    }
}

/// The pool every check in the process shares, of `ARROW_PBT_THREADS`
/// workers
fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        // Without a count from the OS, run on one thread rather than fail
        let default = available_parallelism().map_or(1, NonZero::get);
        let size: usize = env("ARROW_PBT_THREADS", default).unwrap_or_else(|e| panic!("{e}"));
        Pool::new(size.max(1))
    })
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

impl Outcome {
    fn panicked(p: &(dyn Any + Send)) -> Self {
        Self {
            failed: true,
            cases_to_fail: None,
            execs: 0,
            message: format!("panic: {}", panic_message(p)),
            minimal: String::new(),
            ms: 0.0,
            discards: 0,
            reasons: BTreeMap::new(),
        }
    }
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
    use std::collections::BTreeMap;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread::{scope, sleep};
    use std::time::Duration;

    use super::{Outcome, Pool, check_seeds, rerun};

    fn outcome(failed: bool) -> Outcome {
        Outcome {
            failed,
            cases_to_fail: failed.then_some(1),
            execs: 1,
            message: if failed {
                "broken".to_string()
            } else {
                String::new()
            },
            minimal: String::new(),
            ms: 0.0,
            discards: 0,
            reasons: BTreeMap::new(),
        }
    }

    /// The failure reported is the lowest failing seed, even when a higher
    /// one fails first
    #[test]
    fn reports_the_lowest_failing_seed() {
        let r = catch_unwind(AssertUnwindSafe(|| {
            check_seeds(
                "p",
                &[1, 2, 3, 4, 5, 6],
                10,
                0,
                &Pool::new(4),
                |seed, _, _| {
                    if seed == 3 {
                        sleep(Duration::from_millis(200));
                    }
                    outcome(seed == 3 || seed == 6)
                },
            );
        }));
        let message = r.err().unwrap().downcast::<String>().unwrap();
        assert!(message.contains("with seed 3 "), "{message}");
    }

    /// Seeds run on as many threads as the limit allows, and no more
    #[test]
    fn runs_seeds_in_parallel_up_to_the_limit() {
        let running = AtomicUsize::new(0);
        let most = AtomicUsize::new(0);
        check_seeds(
            "p",
            &[1, 2, 3, 4, 5, 6, 7, 8],
            10,
            0,
            &Pool::new(3),
            |_, _, _| {
                let now = running.fetch_add(1, Ordering::SeqCst) + 1;
                most.fetch_max(now, Ordering::SeqCst);
                sleep(Duration::from_millis(50));
                running.fetch_sub(1, Ordering::SeqCst);
                outcome(false)
            },
        );
        assert_eq!(most.load(Ordering::SeqCst), 3);
    }

    /// Checks running at once share the pool's limit, as tests do when the
    /// harness runs them in parallel
    #[test]
    fn checks_share_the_limit() {
        let pool = Pool::new(3);
        let running = AtomicUsize::new(0);
        let most = AtomicUsize::new(0);
        let seed = |_, _, _| {
            let now = running.fetch_add(1, Ordering::SeqCst) + 1;
            most.fetch_max(now, Ordering::SeqCst);
            sleep(Duration::from_millis(50));
            running.fetch_sub(1, Ordering::SeqCst);
            outcome(false)
        };
        scope(|s| {
            for name in ["a", "b"] {
                s.spawn(|| check_seeds(name, &[1, 2, 3, 4, 5, 6], 10, 0, &pool, seed));
            }
        });
        assert_eq!(most.load(Ordering::SeqCst), 3);
    }

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

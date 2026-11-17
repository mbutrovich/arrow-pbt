//! Known bugs. A property that matches a known bug's failure signature
//! discards the case with [`known`] instead of failing, so the run keeps
//! looking for other bugs. Each reason ends with the issue, qualified by its
//! repository, such as `apache/arrow-rs#11620`.
//!
//! Set `ARROW_PBT_NO_KNOWN=1` to turn every known-bug discard back into a
//! failure, for example to check whether an upstream fix removed the bug.

use proptest::test_runner::TestCaseError;
use std::env::var_os;
use std::sync::LazyLock;

/// Read once: the switch cannot change during a run, and `known` runs for
/// every discarded case
static NO_KNOWN: LazyLock<bool> = LazyLock::new(|| var_os("ARROW_PBT_NO_KNOWN").is_some());

/// A discard for a known bug, or a failure when `ARROW_PBT_NO_KNOWN` is set
///
/// # Example
///
/// ```
/// use arrow_pbt::known::known;
/// use proptest::test_runner::TestCaseError;
/// // Without ARROW_PBT_NO_KNOWN, the case is discarded under this reason
/// let e = known("known: an example bug (apache/arrow-rs#1)");
/// assert!(matches!(e, TestCaseError::Reject(_)));
/// ```
pub fn known(reason: impl Into<String>) -> TestCaseError {
    let reason = reason.into();
    if *NO_KNOWN {
        TestCaseError::fail(format!("known bug, filter disabled: {reason}"))
    } else {
        TestCaseError::reject(reason)
    }
}

/// For a known bug that fails one check on every input of a kind, where
/// discarding would discard every case: `Ok`, so the property goes on to its
/// other checks, or a failure when `ARROW_PBT_NO_KNOWN` is set. Unlike
/// [`known`], it is not counted among the discards
///
/// # Example
///
/// ```
/// use arrow_pbt::known::tolerated;
/// // Without ARROW_PBT_NO_KNOWN, the property skips the check and goes on
/// assert!(tolerated("known: an example bug (apache/arrow-rs#1)").is_ok());
/// ```
pub fn tolerated(reason: impl Into<String>) -> Result<(), TestCaseError> {
    if *NO_KNOWN {
        Err(TestCaseError::fail(format!(
            "known bug, filter disabled: {}",
            reason.into()
        )))
    } else {
        Ok(())
    }
}

/// Known: `filter`, `interleave` and the row format rebuild run-end encoded
/// arrays with `RunArray::try_new`, which declares the values field
/// nullable (apache/arrow-rs#11607). Such a
/// failure names one run-end encoded type in both forms, through
/// `Display` in a type comparison or `Debug` in the coalescer's assertion.
/// Any other result is returned unchanged
///
/// # Example
///
/// ```
/// use arrow_pbt::known::ree_values_nullable;
/// use proptest::test_runner::TestCaseError;
/// let m = "expected RunEndEncoded(non-null Int16, non-null Utf8) \
///     but found RunEndEncoded(non-null Int16, Utf8)";
/// let r = ree_values_nullable(Err(TestCaseError::fail(m)));
/// assert!(matches!(r, Err(TestCaseError::Reject(_))));
/// assert!(ree_values_nullable(Ok(())).is_ok());
/// ```
pub fn ree_values_nullable(r: Result<(), TestCaseError>) -> Result<(), TestCaseError> {
    let Err(TestCaseError::Fail(m)) = &r else {
        return r;
    };
    let m = m.to_string();
    // The run ends and values of each `RunEndEncoded(..)` in the message
    let mut forms = vec![];
    for (i, _) in m.match_indices("RunEndEncoded(") {
        let rest = &m[i + "RunEndEncoded(".len()..];
        let mut depth = 1;
        let Some(end) = rest.char_indices().find_map(|(k, c)| {
            depth += (c == '(') as i32 - (c == ')') as i32;
            (depth == 0).then_some(k)
        }) else {
            continue;
        };
        if let Some((ends, values)) = rest[..end].split_once(", ") {
            forms.push((
                ends.to_string(),
                values
                    .strip_prefix("non-null ")
                    .map_or_else(|| (values.to_string(), true), |v| (v.to_string(), false)),
            ));
        }
    }
    let display = forms
        .iter()
        .any(|(e, (v, nullable))| !nullable && forms.contains(&(e.clone(), (v.clone(), true))));
    let debug = m.contains("RunEndEncoded(Field")
        && m.contains("assertion `left == right` failed")
        && m.contains("\"values\"");
    if display || debug {
        return Err(known(
            "known: run-end encoded values field made nullable (apache/arrow-rs#11607)",
        ));
    }
    r
}

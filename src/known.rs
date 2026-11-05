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

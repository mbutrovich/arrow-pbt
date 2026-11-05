//! Property-based tests for arrow-rs, and the pieces to write more.
//!
//! [`generator`] generates valid Arrow arrays of every type in every physical
//! layout, as proptest strategies. [`val`] reads any array back into plain
//! logical values, as an oracle independent of arrow-rs's kernels, and checks
//! outputs against them. [`spec`] checks arrays against the Arrow columnar
//! format. [`types`] has predicates on `DataType`, [`known`] the discard for
//! known bugs, and [`run`] the runner every property goes through. See
//! README.md and docs/.
//!
//! # Example
//!
//! A property over every type and layout: slicing an array keeps the values
//! the oracle reads.
//!
//! ```
//! use arrow_array::Array;
//! use arrow_pbt::generator::{all_types, array_of};
//! use arrow_pbt::run::check;
//! use arrow_pbt::val::{check_rows, vals};
//!
//! check("slice", || array_of(all_types()), |g| {
//!     let a = g.build();
//!     let v = vals(a.as_ref());
//!     for start in 0..=a.len() {
//!         check_rows(&a.slice(start, a.len() - start), a.data_type(), &v[start..])?;
//!     }
//!     Ok(())
//! });
//! ```

pub mod generator;
pub mod known;
pub mod run;
pub mod spec;
pub mod types;
pub mod val;

// Used by this crate's tests and the `arrow-pbt` binary, not a stable API
#[doc(hidden)]
pub mod cases;
#[doc(hidden)]
pub mod coverage;
#[doc(hidden)]
pub mod inventory;
#[doc(hidden)]
pub mod registry;

mod batches;
mod logic;
mod numeric;
mod order;
mod props;
mod strings;
mod temporal;

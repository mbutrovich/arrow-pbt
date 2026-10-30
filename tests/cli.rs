//! The `arrow-pbt` binary's commands that list what it can run.

use std::process::{Command, Output};

use arrow_pbt::registry::all_properties;

fn arrow_pbt(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_arrow-pbt"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn list_prints_every_property() {
    let out = arrow_pbt(&["list"]);
    assert!(out.status.success());
    let names: Vec<_> = all_properties().iter().map(|e| e.name).collect();
    assert_eq!(
        String::from_utf8(out.stdout)
            .unwrap()
            .lines()
            .collect::<Vec<_>>(),
        names
    );
}

#[test]
fn help_prints_usage() {
    for arg in ["help", "--help", "-h"] {
        let out = arrow_pbt(&[arg]);
        assert!(out.status.success(), "{arg}");
        assert!(
            String::from_utf8(out.stdout)
                .unwrap()
                .starts_with("usage: arrow-pbt"),
            "{arg}"
        );
    }
}

#[test]
fn unknown_property_points_to_list() {
    let out = arrow_pbt(&["run", "nope", "1", "1"]);
    assert!(!out.status.success());
    let err = String::from_utf8(out.stderr).unwrap();
    assert!(
        err.contains("no property named nope") && err.contains("arrow-pbt list"),
        "{err}"
    );
}

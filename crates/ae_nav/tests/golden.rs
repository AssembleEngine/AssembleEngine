//! Golden digests for the navigation services.
//!
//! The expected values were produced by running the same code
//! (`common/golden_body.rs`) against the original `pathsim` prototype, so a pass
//! means the lift is bit-faithful. The test must pass unchanged on every
//! platform the engine supports (x86-64 Linux/Windows, aarch64): it is the
//! cross-platform float determinism check for the f32 body math.

extern crate ae_nav as lib;
include!("common/golden_body.rs");

const EXPECTED: &[(&str, u64)] = &[
    ("terrain", 0x06002f817ce0cac0),
    ("hierarchy_stats", 0x85d457b54c9da41e),
    ("context", 0x019b9297739e17be),
    ("flow_field", 0x290d596eff7f1038),
    ("crowd", 0x644a01026fa862f6),
];

fn check(threads: usize) {
    let actual = golden(threads);
    for ((stage, got), (expected_stage, want)) in actual.iter().zip(EXPECTED) {
        assert_eq!(stage, expected_stage);
        assert_eq!(
            *got, *want,
            "stage `{stage}` differs at {threads} thread(s): got {got:#018x}, want {want:#018x}"
        );
    }
    assert_eq!(actual.len(), EXPECTED.len());
}

#[test]
fn golden_digests_single_thread() {
    check(1);
}

#[test]
fn golden_digests_four_threads() {
    check(4);
}

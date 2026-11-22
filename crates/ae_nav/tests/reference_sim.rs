//! Golden state hashes for the lifted pathsim simulation (`ae_nav::reference`).
//!
//! The expected values come from running `common/sim_golden_body.rs` against
//! the original pathsim crate. A pass means the lift is bit-faithful for the
//! whole tick: commands, cohorts, flow-field jobs, a terrain repair, ORCA
//! avoidance and collision. The columnar navigation service is checked
//! against this reference.

mod p {
    pub use ae_nav::math;
    pub use ae_nav::reference::{commands, sim};
}
include!("common/sim_golden_body.rs");

const EXPECTED: &[(u64, u64)] = &[
    (60, 0x3f6cb446c7800fe9),
    (120, 0x6358d8701137daa9),
    (180, 0xef07809764f1bf56),
    (240, 0xd1d30baf014e0c33),
    (300, 0x8c488805c5659242),
    (360, 0x11f3d7b2c255ce81),
    (420, 0x298b2b04d2acd085),
];

fn check(threads: usize) {
    let actual = sim_golden(threads);
    assert_eq!(actual.len(), EXPECTED.len());
    for (&(tick, got), &(_, want)) in actual.iter().zip(EXPECTED) {
        assert_eq!(
            got, want,
            "tick {tick} at {threads} thread(s): got {got:#018x}, want {want:#018x}"
        );
    }
}

#[test]
fn reference_sim_single_thread() {
    check(1);
}

#[test]
fn reference_sim_four_threads() {
    check(4);
}

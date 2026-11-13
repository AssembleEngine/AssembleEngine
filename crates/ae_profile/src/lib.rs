//! Profiling instrumentation for the whole engine.
//!
//! With the `tracy` cargo feature the macros and functions here report to a
//! Tracy client; without it they expand to nothing, so instrumentation can
//! stay in hot paths permanently.
//!
//! ```
//! ae_profile::start();
//! {
//!     ae_profile::zone!("tick");
//!     let _system = ae_profile::zone_dyn("movement");
//!     ae_profile::plot!("entities", 10_000);
//! }
//! ae_profile::frame_mark!("sim");
//! ```

#[cfg(feature = "tracy")]
#[doc(hidden)]
pub use tracy_client;

/// True when the crate was built with Tracy support.
pub const ENABLED: bool = cfg!(feature = "tracy");

/// Start the profiler client. Zones recorded before this call are dropped.
/// Safe to call more than once; a no-op without the `tracy` feature.
pub fn start() {
    #[cfg(feature = "tracy")]
    {
        // The client stays alive for the rest of the process.
        std::mem::forget(tracy_client::Client::start());
    }
}

/// Name the calling thread in the profiler.
pub fn set_thread_name(_name: &str) {
    #[cfg(feature = "tracy")]
    if let Some(client) = tracy_client::Client::running() {
        client.set_thread_name(_name);
    }
}

/// Guard returned by [`zone_dyn`]; the zone ends when it is dropped.
#[must_use = "the zone ends when the guard is dropped"]
pub struct Zone {
    #[cfg(feature = "tracy")]
    _span: Option<tracy_client::Span>,
}

/// Open a zone whose name is only known at run time (a guest system name,
/// a frame graph pass name). Prefer [`zone!`] for fixed names: it is cheaper.
#[inline]
#[track_caller]
pub fn zone_dyn(_name: &str) -> Zone {
    #[cfg(feature = "tracy")]
    {
        let location = std::panic::Location::caller();
        let span = tracy_client::Client::running()
            .map(|client| client.span_alloc(Some(_name), "", location.file(), location.line(), 0));
        Zone { _span: span }
    }
    #[cfg(not(feature = "tracy"))]
    Zone {}
}

/// Open a zone with a fixed name that lasts until the end of the enclosing block.
#[cfg(feature = "tracy")]
#[macro_export]
macro_rules! zone {
    ($name:literal) => {
        let _ae_zone = $crate::tracy_client::Client::running().map(|_| $crate::tracy_client::span!($name));
    };
}

/// Open a zone with a fixed name that lasts until the end of the enclosing block.
#[cfg(not(feature = "tracy"))]
#[macro_export]
macro_rules! zone {
    ($name:literal) => {};
}

/// Mark the end of a frame on a named timeline (`"sim"`, `"render"`).
#[cfg(feature = "tracy")]
#[macro_export]
macro_rules! frame_mark {
    ($name:literal) => {
        if let Some(client) = $crate::tracy_client::Client::running() {
            client.secondary_frame_mark($crate::tracy_client::frame_name!($name));
        }
    };
}

/// Mark the end of a frame on a named timeline (`"sim"`, `"render"`).
#[cfg(not(feature = "tracy"))]
#[macro_export]
macro_rules! frame_mark {
    ($name:literal) => {};
}

/// Record one sample of a named numeric series.
#[cfg(feature = "tracy")]
#[macro_export]
macro_rules! plot {
    ($name:literal, $value:expr) => {
        if let Some(client) = $crate::tracy_client::Client::running() {
            client.plot($crate::tracy_client::plot_name!($name), $value as f64);
        }
    };
}

/// Record one sample of a named numeric series.
#[cfg(not(feature = "tracy"))]
#[macro_export]
macro_rules! plot {
    ($name:literal, $value:expr) => {{
        let _ = &$value;
    }};
}

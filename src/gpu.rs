//! GPU detection for routing decisions.
//!
//! Uses `nvml-wrapper`, which dlopen's `libnvidia-ml.so` at runtime — so the
//! binary still builds and runs on machines with no NVIDIA driver; we simply
//! return `None` and the router treats that as "don't prefer local".
//!
//! NVML is initialized once and cached: symbol loading is somewhat expensive and
//! the handle is reusable.

use nvml_wrapper::Nvml;
use std::sync::OnceLock;

fn nvml() -> Option<&'static Nvml> {
    static NVML: OnceLock<Option<Nvml>> = OnceLock::new();
    NVML.get_or_init(|| Nvml::init().ok()).as_ref()
}

/// `(free_bytes, total_bytes)` for GPU 0, or `None` if no NVIDIA GPU/driver is
/// available. `None` means: caller should not prefer the local route.
pub fn vram() -> Option<(u64, u64)> {
    let nvml = nvml()?;
    let device = nvml.device_by_index(0).ok()?;
    let mem = device.memory_info().ok()?;
    Some((mem.free, mem.total))
}

/// Free VRAM in MiB for GPU 0, or `None` if unavailable.
pub fn free_vram_mb() -> Option<u64> {
    vram().map(|(free, _)| free / (1024 * 1024))
}

/// Short human-readable GPU summary for diagnostics, e.g.
/// "NVIDIA GeForce RTX 3050 — 6020/8192 MiB free". `None` if no GPU.
pub fn summary() -> Option<String> {
    let nvml = nvml()?;
    let device = nvml.device_by_index(0).ok()?;
    let name = device.name().unwrap_or_else(|_| "NVIDIA GPU".to_string());
    let mem = device.memory_info().ok()?;
    Some(format!(
        "{name} — {}/{} MiB free",
        mem.free / (1024 * 1024),
        mem.total / (1024 * 1024)
    ))
}

//! Lightweight tool-usage telemetry (Phase 4).
//!
//! Appends one JSON line per tool invocation to ~/.config/mge/usage.jsonl
//! (local only). `mge stats` aggregates it — this is the data layer for deciding
//! which MCP servers/tools are actually used (auto-prune input). Best-effort:
//! any failure is silently ignored so telemetry never affects tool execution.

use crate::config::Config;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn log_path() -> Option<PathBuf> {
    Config::default_path()
        .ok()?
        .parent()
        .map(|d| d.join("usage.jsonl"))
}

/// Record one tool invocation (best-effort).
pub fn record(tool: &str, ok: bool) {
    let Some(path) = log_path() else {
        return;
    };
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // tool is JSON-escaped via {:?} (Rust debug = valid JSON string for plain ASCII names).
    let line = format!("{{\"t\":{ts},\"tool\":{tool:?},\"ok\":{ok}}}\n");
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = f.write_all(line.as_bytes());
    }
}

/// Aggregate the log into `tool -> (calls, failures)`.
pub fn stats() -> BTreeMap<String, (u64, u64)> {
    let mut agg: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    let Some(path) = log_path() else {
        return agg;
    };
    let Ok(raw) = std::fs::read_to_string(&path) else {
        return agg;
    };
    for line in raw.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let tool = v
            .get("tool")
            .and_then(|t| t.as_str())
            .unwrap_or("?")
            .to_string();
        let ok = v.get("ok").and_then(|o| o.as_bool()).unwrap_or(true);
        let e = agg.entry(tool).or_insert((0, 0));
        e.0 += 1;
        if !ok {
            e.1 += 1;
        }
    }
    agg
}

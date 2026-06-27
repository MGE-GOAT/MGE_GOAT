//! Live model-catalog discovery across configured providers (`mge models`).
//!
//! Queries each non-local provider's OpenAI-style `/models` endpoint so the user
//! can browse the full catalog (OpenRouter alone is hundreds of models) and pick
//! any id to use with `/model <id>` in the TUI. Best-effort: a provider that
//! errors or has no key is silently skipped.

use crate::config::Config;
use serde_json::Value;
use std::time::Duration;

pub struct ModelInfo {
    pub provider: String,
    pub id: String,
    pub free: bool,
    pub modalities: Vec<String>,
}

/// List models across providers, filtered by a case-insensitive substring on the
/// id (empty = all). Sorted free-first, then by id.
pub async fn list(cfg: &Config, query: &str) -> Vec<ModelInfo> {
    let q = query.to_lowercase();
    // Explicit fallback (unwrap_or_default would silently drop the timeout).
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());
    let mut out = Vec::new();
    for (name, pc) in &cfg.providers {
        if pc.local {
            continue;
        }
        let url = format!("{}/models", pc.base_url.trim_end_matches('/'));
        let mut req = client.get(&url);
        if let Some(k) = pc.api_key() {
            req = req.bearer_auth(k);
        }
        let Ok(resp) = req.send().await else { continue };
        let Ok(bytes) = resp.bytes().await else {
            continue;
        };
        if bytes.len() > 16 * 1024 * 1024 {
            continue; // cap a misbehaving provider's response
        }
        let Ok(v) = serde_json::from_slice::<Value>(&bytes) else {
            continue;
        };
        let Some(arr) = v.get("data").and_then(|d| d.as_array()) else {
            continue;
        };
        for m in arr {
            let id = m
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            if id.is_empty() || (!q.is_empty() && !id.to_lowercase().contains(&q)) {
                continue;
            }
            let free = id.ends_with(":free")
                || m.get("pricing")
                    .and_then(|p| p.get("prompt"))
                    .and_then(Value::as_str)
                    .map(|s| s == "0" || s == "0.0")
                    .unwrap_or(false);
            let modalities = m
                .get("architecture")
                .and_then(|a| a.get("input_modalities"))
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            out.push(ModelInfo {
                provider: name.clone(),
                id,
                free,
                modalities,
            });
        }
    }
    out.sort_by(|a, b| b.free.cmp(&a.free).then_with(|| a.id.cmp(&b.id)));
    out
}

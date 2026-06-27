//! Model routing with fallback.
//!
//! A logical route (e.g. "main") expands into an ordered list of [`Candidate`]s:
//! itself plus its declared `fallback` routes. The agent streams from the first
//! candidate that connects; on a *retriable* error (rate-limit / 5xx /
//! connection) before any output, it advances to the next.
//!
//! This is also where GPU-aware local-vs-remote selection will live: a "local"
//! candidate that points at a reachable llama-server simply goes first in the
//! chain. (No local server configured yet — see project notes.)

use crate::config::{Config, ProviderConfig};
use crate::llm::LlmProvider;
use crate::llm::openai_compat::OpenAiCompat;
use anyhow::{Result, bail};
use std::collections::BTreeSet;
use std::sync::Arc;

/// One concrete (provider, model) the agent can try.
#[derive(Clone)]
pub struct Candidate {
    pub provider: Arc<dyn LlmProvider>,
    pub model: String,
    /// Per-route max output tokens, if configured.
    pub max_tokens: Option<u32>,
    /// Display label, e.g. `main:openrouter/qwen/qwen3-coder:free`.
    pub label: String,
    /// Parse tool calls from the model's text (provider lacks native tool-calling).
    pub text_tool_calls: bool,
}

/// Construct a provider client from its config entry.
pub fn build_provider(name: &str, pc: &ProviderConfig) -> Arc<dyn LlmProvider> {
    let mut client = OpenAiCompat::new(name, pc.base_url.clone(), pc.api_key());
    if name == "openrouter" {
        client = client
            .with_header("HTTP-Referer", "https://github.com/mge-goat")
            .with_header("X-Title", "MGE_GOAT");
    }
    Arc::new(client)
}

/// Build a single one-shot [`Candidate`] for an arbitrary model spec, so the user
/// can switch to ANY model on a provider, not just a preconfigured route.
///
/// Spec syntax: `[<provider>:]<model-id>`. The `<provider>:` prefix is only taken
/// when it names a configured provider — so `qwen/qwen3-coder:free` (where `:free`
/// is part of the id) resolves to the default provider, while
/// `nim:qwen/qwen3.5` explicitly selects the `nim` provider.
pub fn candidate_for_model(cfg: &Config, spec: &str) -> Result<Candidate> {
    let spec = spec.trim();
    if spec.is_empty() {
        bail!("empty model spec");
    }
    let (provider_name, model) = match spec.split_once(':') {
        Some((p, rest)) if cfg.providers.contains_key(p) && !rest.is_empty() => {
            (p.to_string(), rest.to_string())
        }
        _ => (default_provider(cfg)?, spec.to_string()),
    };
    let pc = cfg
        .providers
        .get(&provider_name)
        .ok_or_else(|| anyhow::anyhow!("unknown provider '{provider_name}'"))?;
    let needs_key =
        !pc.local && !pc.api_key_env.eq_ignore_ascii_case("none") && !pc.api_key_env.is_empty();
    if needs_key && pc.api_key().is_none() {
        bail!(
            "provider '{provider_name}' has no API key set (env {})",
            pc.api_key_env
        );
    }
    Ok(Candidate {
        provider: build_provider(&provider_name, pc),
        model: model.clone(),
        max_tokens: None,
        label: format!("custom:{provider_name}/{model}"),
        text_tool_calls: pc.parses_text_tool_calls(),
    })
}

/// Pick a sensible default provider for a bare model id: prefer `openrouter`,
/// else the default route's provider, else the first configured provider.
fn default_provider(cfg: &Config) -> Result<String> {
    if cfg.providers.contains_key("openrouter") {
        return Ok("openrouter".to_string());
    }
    if let Ok((_, mr)) = cfg.resolve(&cfg.default_route) {
        return Ok(mr.provider.clone());
    }
    cfg.providers
        .keys()
        .next()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("no providers configured"))
}

/// Expand a route into its ordered candidate chain (route + transitive
/// fallbacks), skipping routes with no model id or a missing required key.
pub fn candidates_for(cfg: &Config, route: &str) -> Result<Vec<Candidate>> {
    // Not a named route? Treat it as an arbitrary model spec (`provider:id` or a
    // bare id) so `--route openai/gpt-4o` and `/model <id>` work for ANY model.
    if !cfg.models.contains_key(route) {
        return Ok(vec![candidate_for_model(cfg, route)?]);
    }
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    let mut order = vec![route.to_string()];
    let mut i = 0;
    while i < order.len() {
        let name = order[i].clone();
        i += 1;
        if !seen.insert(name.clone()) {
            continue; // already expanded — also breaks cycles
        }
        let Ok((pc, mr)) = cfg.resolve(&name) else {
            continue;
        };
        // Enqueue this route's fallbacks FIRST, so a primary that is itself
        // unusable (missing key / empty model) still cascades to its fallbacks.
        order.extend(mr.fallback.iter().cloned());

        if mr.model.trim().is_empty() {
            continue;
        }
        let needs_key =
            !pc.local && !pc.api_key_env.eq_ignore_ascii_case("none") && !pc.api_key_env.is_empty();
        if needs_key && pc.api_key().is_none() {
            continue;
        }
        // GPU-aware gate: skip a local route when the GPU can't fit it (or there
        // is no GPU). Reachability when the server is simply down is still
        // handled by the runtime retriable-fallback, so we only gate on VRAM.
        if pc.local
            && let Some(min_mb) = mr.min_free_vram_mb
        {
            match crate::gpu::free_vram_mb() {
                Some(free) if free >= min_mb => {}
                _ => continue, // no GPU or not enough free VRAM → don't prefer local
            }
        }
        out.push(Candidate {
            provider: build_provider(&mr.provider, pc),
            model: mr.model.clone(),
            max_tokens: mr.max_tokens,
            label: format!("{name}:{}/{}", mr.provider, mr.model),
            text_tool_calls: pc.parses_text_tool_calls(),
        });
    }
    if out.is_empty() {
        bail!("no usable candidates for route '{route}' — check model ids and API keys");
    }
    Ok(out)
}

/// Classify a user prompt into a task tier route name: "fast" for trivial edits,
/// "heavy" for reasoning-heavy work, otherwise "main". The caller falls back to
/// its default route if the chosen tier isn't configured. Heuristic only — no
/// extra LLM call, so it's free and instant.
pub fn classify(prompt: &str) -> &'static str {
    let p = prompt.to_lowercase();
    let len = prompt.chars().count();

    const HEAVY: &[&str] = &[
        "architect",
        "design",
        "refactor",
        "redesign",
        "debug",
        "diagnose",
        "why ",
        "explain",
        "plan ",
        "optimi",
        "algorithm",
        "trade-off",
        "tradeoff",
        "review",
        "security",
        "concurren",
        "race condition",
        "complex",
        "rewrite",
        "migrate",
        "benchmark",
    ];
    const FAST: &[&str] = &[
        "rename",
        "typo",
        "format",
        "lint",
        "list ",
        "what is",
        "print",
        "show ",
        "add a comment",
        "fix the import",
        "bump",
        "version",
        "spelling",
    ];

    if len > 400 || HEAVY.iter().any(|k| p.contains(k)) {
        "heavy"
    } else if len < 60 || FAST.iter().any(|k| p.contains(k)) {
        "fast"
    } else {
        "main"
    }
}

/// Whether an error from `stream_chat` is worth retrying on the next candidate.
pub fn is_retriable(e: &anyhow::Error) -> bool {
    let s = e.to_string().to_lowercase();
    s.contains("429")
        || s.contains("rate-limit")
        || s.contains("rate limit")
        || s.contains(" 500")
        || s.contains(" 502")
        || s.contains(" 503")
        || s.contains(" 504")
        || s.contains("connecting to provider")
        || s.contains("timed out")
        || s.contains("timeout")
        // Model unavailable / unknown id → try the next candidate. Lets the
        // cascade self-heal when a configured/default model id goes stale.
        || s.contains("404")
        || s.contains("no endpoints")
        || s.contains("model_not_found")
        || s.contains("not_found_error")
        || s.contains("does not exist")
        // Auth/credit problems on one provider → fall back to a working one
        // (e.g. an expired NIM key or unfunded OpenAI/Claude key cascades to
        // free/local). A bad key on one provider says nothing about the next.
        || s.contains("insufficient_quota")
        || s.contains("credit balance")
        || s.contains("quota")
        || s.contains(" 401")
        || s.contains(" 403")
        || s.contains("unauthorized")
        || s.contains("invalid_api_key")
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;

    #[test]
    fn retriable_classification() {
        assert!(is_retriable(&anyhow!(
            "provider 'x' returned HTTP 429 Too Many Requests"
        )));
        assert!(is_retriable(&anyhow!("connecting to provider 'local'")));
        // A stale key on one provider must cascade to the next candidate.
        assert!(is_retriable(&anyhow!(
            "provider 'x' returned HTTP 401 Unauthorized"
        )));
        assert!(!is_retriable(&anyhow!("bad regex")));
    }

    #[test]
    fn classify_tiers() {
        assert_eq!(classify("rename foo to bar"), "fast");
        assert_eq!(
            classify("please architect a caching layer for the API"),
            "heavy"
        );
        assert_eq!(
            classify("implement a parser that reads the config file and returns a typed struct"),
            "main"
        );
    }

    #[test]
    fn chain_expands_and_dedups() {
        let toml = r#"
            default_route = "main"
            [providers.openrouter]
            base_url = "https://openrouter.ai/api/v1"
            api_key_env = "NONE"
            [models.main]
            provider = "openrouter"
            model = "a"
            fallback = ["b", "main"]
            [models.b]
            provider = "openrouter"
            model = "bb"
        "#;
        let cfg: Config = toml::from_str(toml).unwrap();
        let c = candidates_for(&cfg, "main").unwrap();
        // main -> b, with main's self-reference deduped away.
        assert_eq!(c.len(), 2);
        assert!(c[0].label.contains("/a"));
        assert!(c[1].label.contains("/bb"));
    }
}

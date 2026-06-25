//! Configuration: providers, API keys, and model routing.
//!
//! Config is loaded from (in order, later overrides earlier):
//!   1. built-in defaults (documented base URLs; NO model IDs are assumed)
//!   2. ~/.config/mge/config.toml
//!   3. environment variables for secrets (never store keys in the TOML)
//!
//! IMPORTANT (project rule): we never guess model IDs. The user supplies them in
//! config.toml or via `mge` flags. Base URLs below are the providers' documented
//! OpenAI-compatible endpoints.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// How a provider speaks. For now everything is OpenAI-compatible (OpenRouter,
/// NVIDIA NIM, and llama.cpp's `llama-server` all expose this surface).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    /// OpenAI-compatible `/chat/completions` (streaming via SSE).
    Openai,
}

impl Default for ProviderKind {
    fn default() -> Self {
        ProviderKind::Openai
    }
}

/// A single backend the agent can talk to.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    #[serde(default)]
    pub kind: ProviderKind,
    /// Base URL including the version path, e.g. `https://openrouter.ai/api/v1`.
    pub base_url: String,
    /// Name of the env var holding the bearer token. Empty/"NONE" => no auth
    /// (used for a locally hosted llama-server).
    #[serde(default)]
    pub api_key_env: String,
    /// Whether this provider runs on the local machine (affects routing: the
    /// local GPU should not be used for parallel subagents).
    #[serde(default)]
    pub local: bool,
}

impl ProviderConfig {
    /// Resolve the API key from the environment, if this provider needs one.
    pub fn api_key(&self) -> Option<String> {
        let var = self.api_key_env.trim();
        if var.is_empty() || var.eq_ignore_ascii_case("none") {
            return None;
        }
        std::env::var(var).ok().filter(|v| !v.trim().is_empty())
    }
}

/// A named model route: which provider serves it and under what model id.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRoute {
    pub provider: String,
    /// The provider-specific model id (e.g. an OpenRouter slug). User-supplied;
    /// never defaulted/guessed.
    pub model: String,
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Logical model routes keyed by role, e.g. "main", "agent", "fast".
    #[serde(default)]
    pub models: BTreeMap<String, ModelRoute>,
    /// Backends keyed by name, e.g. "openrouter", "nim", "local".
    #[serde(default)]
    pub providers: BTreeMap<String, ProviderConfig>,
    /// Which route to use for the main interactive loop.
    #[serde(default = "default_main_route")]
    pub default_route: String,
}

fn default_main_route() -> String {
    "main".to_string()
}

impl Default for Config {
    fn default() -> Self {
        let mut providers = BTreeMap::new();
        // Documented OpenAI-compatible endpoints. Model ids are intentionally absent.
        providers.insert(
            "openrouter".to_string(),
            ProviderConfig {
                kind: ProviderKind::Openai,
                base_url: "https://openrouter.ai/api/v1".to_string(),
                api_key_env: "OPENROUTER_API_KEY".to_string(),
                local: false,
            },
        );
        providers.insert(
            "nim".to_string(),
            ProviderConfig {
                kind: ProviderKind::Openai,
                base_url: "https://integrate.api.nvidia.com/v1".to_string(),
                api_key_env: "NVIDIA_NIM_API_KEY".to_string(),
                local: false,
            },
        );
        providers.insert(
            "local".to_string(),
            ProviderConfig {
                kind: ProviderKind::Openai,
                // llama.cpp `llama-server` default OpenAI-compatible address.
                base_url: "http://localhost:8080/v1".to_string(),
                api_key_env: "NONE".to_string(),
                local: true,
            },
        );

        Config {
            models: BTreeMap::new(),
            providers,
            default_route: default_main_route(),
        }
    }
}

impl Config {
    /// Standard config file path: ~/.config/mge/config.toml
    pub fn default_path() -> Result<PathBuf> {
        let dirs = directories::ProjectDirs::from("dev", "mge", "mge")
            .context("could not determine config directory")?;
        Ok(dirs.config_dir().join("config.toml"))
    }

    /// Load config, merging the on-disk file over built-in defaults. Missing file
    /// is not an error — defaults are returned so first-run still works.
    pub fn load() -> Result<Self> {
        let path = Self::default_path()?;
        Self::load_from(&path)
    }

    pub fn load_from(path: &PathBuf) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("reading config at {}", path.display()))?;
        let parsed: Config = toml::from_str(&raw)
            .with_context(|| format!("parsing config at {}", path.display()))?;
        Ok(parsed)
    }

    /// Resolve a logical route name to (provider config, model id).
    pub fn resolve(&self, route: &str) -> Result<(&ProviderConfig, &ModelRoute)> {
        let mr = self
            .models
            .get(route)
            .with_context(|| format!("no model route named '{route}' in config"))?;
        let pc = self.providers.get(&mr.provider).with_context(|| {
            format!("route '{route}' references unknown provider '{}'", mr.provider)
        })?;
        Ok((pc, mr))
    }

    /// Write a starter config file with comments to the default path.
    pub fn write_starter() -> Result<PathBuf> {
        let path = Self::default_path()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, STARTER_TOML)?;
        Ok(path)
    }
}

/// Starter config shipped on `mge init`. Model ids are left blank on purpose —
/// the user fills these in (we do not guess provider model slugs).
pub const STARTER_TOML: &str = r#"# MGE_GOAT configuration 🐐🍦
# Secrets are read from environment variables, never stored here.

default_route = "main"

# ── Providers ────────────────────────────────────────────────────────────────
[providers.openrouter]
kind = "openai"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"

[providers.nim]
kind = "openai"
base_url = "https://integrate.api.nvidia.com/v1"
api_key_env = "NVIDIA_NIM_API_KEY"

[providers.local]
kind = "openai"
base_url = "http://localhost:8080/v1"   # llama.cpp `llama-server`
api_key_env = "NONE"
local = true

# ── Model routes ─────────────────────────────────────────────────────────────
# Fill in the `model` field with the exact id from your provider's catalog.
# (Left blank intentionally — MGE_GOAT never guesses model ids.)
#
# [models.main]
# provider = "openrouter"
# model = "<paste model id here>"
#
# [models.agent]      # subagents always route to a remote API
# provider = "nim"
# model = "<paste model id here>"
#
# [models.local]
# provider = "local"
# model = "<the model you loaded into llama-server>"
"#;

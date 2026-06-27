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
#[derive(Default)]
pub enum ProviderKind {
    /// OpenAI-compatible `/chat/completions` (streaming via SSE).
    #[default]
    Openai,
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
    /// Whether to parse tool calls from the model's TEXT (Qwen/Hermes style).
    /// **Defaults to `true`**, because the open models this tool targets (Qwen,
    /// gpt-oss, etc., on llama.cpp / NIM / OpenRouter / HF) emit tool calls as text
    /// rather than the native structured field — without text parsing they can't act
    /// as agents at all. Set `false` for providers whose models DO native structured
    /// tool-calling (OpenAI, Anthropic, GitHub Models gpt-4.1) to harden them.
    ///
    /// SECURITY: when on, a model that quotes untrusted content containing
    /// `<function=…>` markup could have it parsed as a real call. Bounded by (a) the
    /// known-tool-name filter and (b) text parsing only firing when the model returned
    /// NO structured call (so native-tool models never hit this path). NOTE: the
    /// `bash`/`delegate` approval prompt covers only MUTATING tools — a prompt-injected
    /// `read_file`/`web_fetch` auto-allows in `default`/`acceptEdits` mode (web_fetch
    /// still SSRF-guarded). Use `plan` mode, or set this `false`, on untrusted repos.
    #[serde(default)]
    pub text_tool_calls: Option<bool>,
}

impl ProviderConfig {
    /// Whether to parse tool calls from model text. Defaults to `true` — see the
    /// field docs (open models require it; set `false` per native-tool provider).
    pub fn parses_text_tool_calls(&self) -> bool {
        self.text_tool_calls.unwrap_or(true)
    }

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
    /// Other route names to try, in order, if this one errors retriably
    /// (rate-limit / 5xx / connection failure).
    #[serde(default)]
    pub fallback: Vec<String>,
    /// For local (GPU) routes: minimum free VRAM (MiB) required to even attempt
    /// this route. If set and the GPU has less free (or no GPU is present), the
    /// route is skipped so we don't waste a turn on a model that can't load.
    /// User-declared per the never-guess rule (don't auto-estimate GGUF size).
    #[serde(default)]
    pub min_free_vram_mb: Option<u64>,
}

/// MCP (Model Context Protocol) servers the agent connects to. Their tools are
/// exposed to the agent as `mcp__<server>__<tool>`. A broken server is skipped.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct McpConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub servers: BTreeMap<String, McpServerConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// "stdio" (spawn a child process) or "http"/"sse" (remote). Accepts the
    /// Claude Code `type` key as an alias so plugin `.mcp.json` files just work.
    #[serde(default = "default_stdio", alias = "type")]
    pub transport: String,
    #[serde(default)]
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub url: String,
    /// Sandbox tier for a spawned stdio server: "process" (default — set
    /// NO_NEW_PRIVS so it can't gain privileges) or "off".
    #[serde(default = "default_sandbox")]
    pub sandbox: String,
    /// If non-empty, only register tools whose name starts with one of these
    /// prefixes — keeps a huge server (e.g. 300+ tools) from flooding the model.
    #[serde(default)]
    pub tools_allow: Vec<String>,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_stdio() -> String {
    "stdio".to_string()
}
fn default_sandbox() -> String {
    "process".to_string()
}
fn default_true() -> bool {
    true
}

/// Markdown skills (Claude Code-compatible `SKILL.md`). The agent sees a listing
/// of skill names+descriptions and loads a skill's body on demand via `use_skill`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillsConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Load skills from the CURRENT project (./.mge/skills, ./.claude/skills).
    /// OFF by default: a repo's skill files are untrusted content that could
    /// carry prompt injection, and the TUI auto-approves bash. Opt in per-project.
    #[serde(default)]
    pub trust_project_skills: bool,
    /// Extra skill roots beyond ~/.config/mge/skills (trusted; you control them).
    #[serde(default)]
    pub extra_dirs: Vec<String>,
}

impl Default for SkillsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            trust_project_skills: false,
            extra_dirs: vec![],
        }
    }
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
    /// When true, the agent picks a route per task (fast/main/heavy) by
    /// classifying each prompt, instead of using a single fixed route.
    #[serde(default)]
    pub auto_route: bool,
    /// MCP servers to connect to and expose as tools.
    #[serde(default)]
    pub mcp: McpConfig,
    /// Markdown skill loading.
    #[serde(default)]
    pub skills: SkillsConfig,
    /// Background marketplace daemon (notify-only auto-prune reminders).
    #[serde(default)]
    pub marketplace: MarketplaceConfig,
    /// Tiered approval mode + per-tool allow/ask/deny rules.
    #[serde(default)]
    pub permissions: PermissionsConfig,
    /// Optional test/lint feedback loop (post-edit checks).
    #[serde(default)]
    pub checks: ChecksConfig,
    /// Repo map injected into the system prompt for codebase orientation.
    #[serde(default)]
    pub repo_map: RepoMapConfig,
    /// External agent CLIs the `delegate` tool can hand subtasks to (e.g. Codex
    /// or Claude Code — using THEIR subscription auth via their official client).
    #[serde(default)]
    pub agents: BTreeMap<String, AgentSpec>,
    /// Lifecycle hooks fired around tool calls (auto-format, blocking gates).
    #[serde(default)]
    pub hooks: HooksConfig,
    /// Language servers the `lsp_diagnostics` tool can query, keyed by file
    /// extension (e.g. "rs" -> rust-analyzer).
    #[serde(default)]
    pub lsp: LspConfig,
    /// Optional semantic-retrieval config for the `semantic_search` tool.
    #[serde(default)]
    pub rag: RagConfig,
}

/// `[hooks]` — user shell commands fired around tool calls. A `PreToolUse` hook
/// that exits non-zero BLOCKS the tool; `PostToolUse` hooks run after (e.g. auto-
/// format). Hooks come only from this trusted config.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HooksConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_hook_timeout")]
    pub timeout_secs: u64,
    #[serde(default)]
    pub hooks: Vec<HookEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookEntry {
    /// "PreToolUse" or "PostToolUse".
    pub event: String,
    /// Tool-name wildcard (e.g. "edit_file", "write_file", "*"). Default "*".
    #[serde(default = "default_hook_matcher")]
    pub matcher: String,
    /// Shell command to run (`sh -c`). Gets MGE_TOOL / MGE_TOOL_ARGS env vars.
    pub command: String,
}

fn default_hook_timeout() -> u64 {
    30
}
fn default_hook_matcher() -> String {
    "*".to_string()
}

/// `[lsp]` — language servers the `lsp_diagnostics` tool may spawn, keyed by file
/// extension. The command comes only from this trusted config (never the model).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LspConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_lsp_timeout")]
    pub timeout_secs: u64,
    /// File extension (no dot) -> `[command, args...]`, e.g. `rs = ["rust-analyzer"]`.
    #[serde(default)]
    pub servers: BTreeMap<String, Vec<String>>,
}

fn default_lsp_timeout() -> u64 {
    30
}

/// `[rag]` — optional SEMANTIC retrieval. Off by default (lexical BM25 covers the
/// common case offline + dependency-free). When `endpoint` is set, the
/// `semantic_search` tool embeds the codebase via an OpenAI-compatible embeddings
/// endpoint (e.g. GitHub Models) for conceptual "where do we handle X" queries.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RagConfig {
    #[serde(default)]
    pub enabled: bool,
    /// OpenAI-compatible base URL (e.g. https://models.github.ai/inference). The
    /// tool POSTs to `<endpoint>/embeddings`.
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default = "default_embed_model")]
    pub model: String,
    #[serde(default = "default_embed_key_env")]
    pub api_key_env: String,
}

fn default_embed_model() -> String {
    "openai/text-embedding-3-small".to_string()
}
fn default_embed_key_env() -> String {
    "GITHUB_TOKEN".to_string()
}

/// An external agent CLI invoked as `<command> <args...> "<task>"`. Used to tap a
/// Codex / Claude Code (etc.) SUBSCRIPTION through its official client — no API
/// key or token extraction, so it stays within the provider's terms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSpec {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    /// One-line hint shown to the model (what this agent is good for).
    #[serde(default)]
    pub description: String,
    #[serde(default = "default_agent_timeout")]
    pub timeout_secs: u64,
}

fn default_agent_timeout() -> u64 {
    600
}

/// `[permissions]` — approval posture + per-tool rules. See [`crate::permissions`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PermissionsConfig {
    /// "default" | "acceptEdits" | "plan" | "yolo" (None → default).
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub allow: Vec<String>,
    #[serde(default)]
    pub ask: Vec<String>,
    #[serde(default)]
    pub deny: Vec<String>,
}

/// `[checks]` — a shell command run after each successful write/edit; failures
/// are fed back to the model so it fixes them in-session. Opt-in.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChecksConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub after_edit_cmd: Option<String>,
    #[serde(default = "default_check_timeout")]
    pub timeout_secs: u64,
}

impl Default for ChecksConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            after_edit_cmd: None,
            timeout_secs: default_check_timeout(),
        }
    }
}

fn default_check_timeout() -> u64 {
    60
}

/// `[repo_map]` — a cheap symbol map injected into the system prompt for
/// whole-codebase orientation. On by default; no new deps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoMapConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_map_budget")]
    pub char_budget: usize,
    #[serde(default = "default_top_files")]
    pub top_files: usize,
    #[serde(default = "default_top_symbols")]
    pub top_symbols: usize,
}

impl Default for RepoMapConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            char_budget: default_map_budget(),
            top_files: default_top_files(),
            top_symbols: default_top_symbols(),
        }
    }
}

fn default_map_budget() -> usize {
    16_000
}
fn default_top_files() -> usize {
    30
}
fn default_top_symbols() -> usize {
    15
}

/// Background marketplace daemon. Notify-only by design: it reminds you about
/// unused MCP tools on an interval — it never silently installs or deletes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketplaceConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_interval")]
    pub interval_mins: u64,
}

impl Default for MarketplaceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            interval_mins: 60,
        }
    }
}

fn default_interval() -> u64 {
    60
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
                text_tool_calls: None,
            },
        );
        providers.insert(
            "nim".to_string(),
            ProviderConfig {
                kind: ProviderKind::Openai,
                base_url: "https://integrate.api.nvidia.com/v1".to_string(),
                api_key_env: "NVIDIA_NIM_API_KEY".to_string(),
                local: false,
                text_tool_calls: None,
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
                text_tool_calls: None,
            },
        );

        Config {
            models: BTreeMap::new(),
            providers,
            default_route: default_main_route(),
            auto_route: false,
            mcp: McpConfig::default(),
            skills: SkillsConfig::default(),
            marketplace: MarketplaceConfig::default(),
            permissions: PermissionsConfig::default(),
            checks: ChecksConfig::default(),
            repo_map: RepoMapConfig::default(),
            agents: BTreeMap::new(),
            hooks: HooksConfig::default(),
            lsp: LspConfig::default(),
            rag: RagConfig::default(),
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

    /// Load `~/.config/mge/secrets.env` into the process environment so API keys
    /// are available without the user manually sourcing the file. Existing env
    /// vars win (so a shell-exported key overrides the file). Missing file is fine.
    pub fn load_secrets() {
        let Ok(path) = Self::default_path() else {
            return;
        };
        let Some(dir) = path.parent() else { return };
        let Ok(raw) = std::fs::read_to_string(dir.join("secrets.env")) else {
            return;
        };
        for line in raw.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let line = line.strip_prefix("export ").unwrap_or(line);
            if let Some((k, v)) = line.split_once('=') {
                let k = k.trim();
                let v = v.trim().trim_matches('"').trim_matches('\'');
                // Skip malformed entries rather than letting set_var panic
                // (empty key, '=' or NUL in key, NUL in value, whitespace key).
                let bad_key =
                    k.is_empty() || k.contains(['=', '\0']) || k.chars().any(char::is_whitespace);
                if bad_key || v.contains('\0') {
                    continue;
                }
                if std::env::var_os(k).is_none() {
                    // SAFETY: load_secrets() runs in main() before the Tokio
                    // runtime is created, so no other threads exist yet.
                    unsafe { std::env::set_var(k, v) };
                }
            }
        }
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
            format!(
                "route '{route}' references unknown provider '{}'",
                mr.provider
            )
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

/// Load project + user instruction files (the **AGENTS.md / CLAUDE.md** de-facto
/// standard) as a system-prompt addendum, size-capped. The global
/// `~/.config/mge/AGENTS.md` (next to the config) is always included if present.
/// The **project** files (`AGENTS.md`/`.mge/AGENTS.md`/`CLAUDE.md` in the cwd) are
/// untrusted repo content — a malicious repo's CLAUDE.md is a prompt-injection
/// vector — so they load only when `[skills].trust_project_skills = true`, the
/// same opt-in gate used for project skills.
pub fn project_memory(cfg: &Config) -> Option<String> {
    const CAP: usize = 16 * 1024;
    let mut parts: Vec<(String, String)> = Vec::new();

    if let Ok(cfgpath) = Config::default_path()
        && let Some(dir) = cfgpath.parent()
        && let Ok(s) = std::fs::read_to_string(dir.join("AGENTS.md"))
    {
        parts.push(("user AGENTS.md".to_string(), s));
    }
    if cfg.skills.trust_project_skills {
        for cand in ["AGENTS.md", ".mge/AGENTS.md", "CLAUDE.md"] {
            if let Ok(s) = std::fs::read_to_string(cand) {
                parts.push((format!("project {cand}"), s));
                break;
            }
        }
    }
    if parts.is_empty() {
        return None;
    }

    let mut out = String::from("# Project & user instructions (honor these)\n");
    for (label, body) in &parts {
        out.push_str("\n## ");
        out.push_str(label);
        out.push('\n');
        out.push_str(body.trim());
        out.push('\n');
    }
    Some(crate::util::clip(&out, CAP))
}

/// Starter config shipped on `mge init`: a working free-first auto-cascade
/// (NIM-primary → free OpenRouter → local) with OpenAI/Claude routes that stay
/// idle until keys are added. Model ids are real-but-drift — `mge models` lists
/// current ones; the cascade self-heals past a stale id (404 → next candidate).
pub const STARTER_TOML: &str = r#"# MGE_GOAT configuration 🐐🍦
# Secrets are read from environment variables, never stored here.

default_route = "main"

# ── Providers (all OpenAI-compatible; keys come from secrets.env) ─────────────
[providers.openrouter]
kind = "openai"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"

[providers.nim]
kind = "openai"
base_url = "https://integrate.api.nvidia.com/v1"
api_key_env = "NVIDIA_NIM_API_KEY"

# text_tool_calls defaults to TRUE so open models (Qwen/gpt-oss on NIM/OpenRouter/
# HF/local) — which emit tool calls as text — can actually use tools. Native-tool
# providers below set it FALSE to harden against prose-injected tool markup; they
# return structured calls anyway, so they never need text parsing.
[providers.openai]
kind = "openai"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
text_tool_calls = false   # OpenAI does native structured tool-calling

[providers.anthropic]
kind = "openai"
base_url = "https://api.anthropic.com/v1"   # Anthropic's OpenAI-compat endpoint
api_key_env = "ANTHROPIC_API_KEY"
text_tool_calls = false   # Claude does native structured tool-calling

[providers.github]                           # GitHub Models — FREE tier, just a GitHub PAT
kind = "openai"
base_url = "https://models.github.ai/inference"
api_key_env = "GITHUB_TOKEN"
text_tool_calls = false   # gpt-4.1 family does native structured tool-calling

[providers.local]
kind = "openai"
base_url = "http://localhost:8080/v1"        # llama.cpp `llama-server`
api_key_env = "NONE"
local = true

# ── Model routes — free-first AUTO-CASCADE ────────────────────────────────────
# Each route falls back (on rate-limit / 5xx / timeout / unavailable-model /
# no-credit) down its chain, ending at `local`. Routes whose provider key is
# missing are skipped automatically (so openai/claude idle until you add a key).
# Switch live in the TUI with `/model <id>`; list ids with `mge models`. Model
# ids drift over time, but the cascade self-heals past a stale one.
#
# PRIMARY = GitHub Models gpt-4.1 (FREE with a GitHub PAT) because it has reliable
# NATIVE tool-calling — the agent's tool loop just works. NIM/OpenRouter open
# models (Qwen/gpt-oss) are capable but speak tool calls as TEXT, which is less
# reliable; they're kept as fallbacks. No GitHub token? Set GITHUB_TOKEN via
# `mge setup`, or the cascade drops to NIM/OpenRouter/local automatically.

[models.main]            # primary coding loop — free + native tool-calling
provider = "github"
model = "openai/gpt-4.1-mini"
fallback = ["main_nim", "main_free", "local"]

[models.main_nim]        # NIM Qwen — capable, no daily wall (text-format tool calls)
provider = "nim"
model = "qwen/qwen3.5-122b-a10b"
fallback = ["main_free", "local"]

[models.main_free]       # OpenRouter free
provider = "openrouter"
model = "qwen/qwen3-coder:free"
fallback = ["local"]

[models.agent]           # subagents — free + native first
provider = "github"
model = "openai/gpt-4.1-mini"
fallback = ["agent_nim", "agent_free", "local"]

[models.agent_nim]
provider = "nim"
model = "moonshotai/kimi-k2.6"
fallback = ["agent_free", "local"]

[models.agent_free]
provider = "openrouter"
model = "openai/gpt-oss-120b:free"
fallback = ["local"]

[models.heavy]           # reasoning-heavy tier — stronger free native model
provider = "github"
model = "openai/gpt-4.1"
fallback = ["heavy_nim", "heavy_free", "local"]

[models.heavy_nim]
provider = "nim"
model = "moonshotai/kimi-k2.6"
fallback = ["heavy_free", "local"]

[models.heavy_free]
provider = "openrouter"
model = "openai/gpt-oss-120b:free"
fallback = ["local"]

[models.fast]            # light tasks — gpt-4.1-mini is free, fast & native
provider = "github"
model = "openai/gpt-4.1-mini"
fallback = ["main", "local"]

[models.local]           # ultimate fallback — run `llama-server` (docs/LOCAL_LLAMA.md)
provider = "local"
model = "local"
min_free_vram_mb = 3000

[models.openai]          # premium — add OPENAI_API_KEY, then /model openai
provider = "openai"
model = "gpt-4o-mini"
fallback = ["main"]

[models.claude]          # premium — add ANTHROPIC_API_KEY, then /model claude
provider = "anthropic"
model = "claude-3-5-sonnet-20241022"
fallback = ["main"]

# Multimodal: @image.png / @audio.mp3 in chat auto-route here (text models 400 on media).
[models.vision]          # images — NIM vision primary (no rate limits), free fallbacks
provider = "nim"
model = "meta/llama-3.2-11b-vision-instruct"
fallback = ["vision_or1", "vision_or2"]

[models.vision_or1]      # OpenRouter free vision fallback
provider = "openrouter"
model = "nvidia/nemotron-nano-12b-v2-vl:free"
fallback = ["vision_or2"]

[models.vision_or2]
provider = "openrouter"
model = "google/gemma-4-26b-a4b-it:free"

[models.audio]           # audio input (@voice.wav). NOTE: not free — OpenRouter needs ~$0.50 balance.
provider = "openrouter"
model = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"

# ── Permissions ──────────────────────────────────────────────────────────────
# [permissions]
# mode = "default"   # default | acceptEdits | plan | yolo
#                    #   default    : ask before bash/write/edit, allow reads
#                    #   acceptEdits: auto-allow edits, ask on bash
#                    #   plan       : READ-ONLY — blocks ALL bash & writes (even
#                    #                allow rules and `bash ls`); use for audits
#                    #   yolo       : allow everything (deny rules still apply)
# deny  = []         # deny ALWAYS wins, even in yolo. Use "bash:<pattern>" for
#                    #   shell commands, e.g. ["bash:rm -rf *", "bash:curl *| bash"]
# allow = []         # e.g. ["read_file", "grep", "glob"]
# ask   = []         # prompt on CLI; treated as allow in the TUI (no stdin)

# ── Checks (test/lint feedback loop) ─────────────────────────────────────────
# Opt-in. Runs after every successful write/edit; output is fed back to the model
# so it fixes failures in-session. Output is injected into the agent's context —
# use PROJECT-LOCAL binaries only (a compromised tool could inject instructions).
# [checks]
# enabled = false
# after_edit_cmd = "cargo check --message-format short"
# timeout_secs = 60

# ── External agents — use a Codex / Claude Code SUBSCRIPTION via its CLI ───────
# The `delegate` tool hands a subtask to these. They run under the CLI's OWN
# logged-in subscription (no API key, no token extraction) — install + log into
# each CLI first (`codex`, `claude`). Then the agent can `delegate` to them.
# [agents.codex]
# command = "codex"
# args = ["exec", "--skip-git-repo-check"]
# description = "OpenAI Codex (ChatGPT subscription)"
# [agents.claude]
# command = "claude"
# args = ["-p"]
# description = "Claude Code (Claude subscription)"

# ── Hooks — your shell commands fired around tool calls ───────────────────────
# A PreToolUse hook that exits non-zero BLOCKS the tool (its output is shown to
# the model); PostToolUse runs after success (e.g. auto-format). Each hook gets
# MGE_TOOL / MGE_TOOL_ARGS (pre) or MGE_TOOL_RESULT (post) in its env.
# [hooks]
# enabled = false
# timeout_secs = 30
# [[hooks.hooks]]
# event = "PostToolUse"
# matcher = "write_file"      # tool-name glob; "*" = all
# command = "cargo fmt"
# [[hooks.hooks]]
# event = "PreToolUse"
# matcher = "bash"
# command = '''case "$MGE_TOOL_ARGS" in *"rm -rf /"*) echo "refusing rm -rf /" ; exit 1 ;; esac'''

# ── LSP — real compiler/linter diagnostics via the `lsp_diagnostics` tool ──────
# Maps a file extension to the language server to spawn for it. Install the
# server yourself; MGE just talks LSP to it over stdio.
# [lsp]
# enabled = false
# timeout_secs = 30
# [lsp.servers]
# rs = ["rust-analyzer"]
# py = ["pyright-langserver", "--stdio"]
# ts = ["typescript-language-server", "--stdio"]
# go = ["gopls"]

# ── Semantic search (the `semantic_search` tool) ──────────────────────────────
# Optional: embed the codebase via an OpenAI-compatible embeddings endpoint and
# search by MEANING for conceptual queries. Off by default — lexical (grep /
# find_symbol / code_graph) works with no endpoint. GitHub Models is free.
# [rag]
# enabled = true
# endpoint = "https://models.github.ai/inference"
# model = "openai/text-embedding-3-small"
# api_key_env = "GITHUB_TOKEN"
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_tool_calls_defaults_on_and_starter_hardens_native() {
        // Default ON so open models (which emit tool calls as text) can use tools.
        let p = ProviderConfig {
            kind: ProviderKind::Openai,
            base_url: String::new(),
            api_key_env: String::new(),
            local: false,
            text_tool_calls: None,
        };
        assert!(p.parses_text_tool_calls(), "default must be true");
        // Explicit false (native-tool providers) is honored.
        let native = ProviderConfig {
            text_tool_calls: Some(false),
            ..p.clone()
        };
        assert!(!native.parses_text_tool_calls());
        // The shipped starter must harden OpenAI/Anthropic (they do native calls)…
        let cfg: Config = toml::from_str(STARTER_TOML).expect("STARTER_TOML must parse");
        assert!(!cfg.providers["openai"].parses_text_tool_calls());
        assert!(!cfg.providers["anthropic"].parses_text_tool_calls());
        // github carries all the default traffic — its hardening must not regress.
        assert!(!cfg.providers["github"].parses_text_tool_calls());
        // …while the open-model providers keep text parsing on (so they can act).
        assert!(cfg.providers["nim"].parses_text_tool_calls());
        assert!(cfg.providers["openrouter"].parses_text_tool_calls());
    }

    #[test]
    fn starter_toml_parses_into_cascade() {
        let cfg: Config = toml::from_str(STARTER_TOML).expect("STARTER_TOML must parse");
        for r in [
            "main",
            "main_free",
            "agent",
            "heavy",
            "fast",
            "local",
            "openai",
            "claude",
            "vision",
            "audio",
        ] {
            assert!(cfg.models.contains_key(r), "missing route {r}");
        }
        for p in [
            "openrouter",
            "nim",
            "openai",
            "anthropic",
            "github",
            "local",
        ] {
            assert!(cfg.providers.contains_key(p), "missing provider {p}");
        }
        // The cascade must terminate at local.
        assert!(cfg.models["main"].fallback.contains(&"local".to_string()));
    }
}

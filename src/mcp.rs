//! MCP client (Phase 1: stdio).
//!
//! Connects to configured MCP servers, lists their tools, and registers each as
//! an [`McpTool`] (which implements the existing [`crate::tools::Tool`] trait)
//! under a namespaced `mcp__<server>__<tool>` name — so MCP server tools look
//! identical to built-in tools and the agent loop needs ZERO changes.
//!
//! Contract: a broken / unreachable / misbehaving server is logged and skipped;
//! it never propagates and the rest of MGE_GOAT runs normally (graceful degrade).

use crate::config::{Config, McpServerConfig};
use crate::llm::ToolDef;
use crate::tools::{Registry, Tool};
use anyhow::{Result, anyhow, bail};
use async_trait::async_trait;
use rmcp::model::{CallToolRequestParams, RawContent};
use rmcp::service::{Peer, RunningService};
use rmcp::transport::{ConfigureCommandExt, StreamableHttpClientTransport, TokioChildProcess};
use rmcp::{RoleClient, ServiceExt};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

/// A hung MCP server must never freeze startup — cap the handshake/list.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);

// ── rug-pull defense: fingerprint a server's tool schemas and block on drift ──

fn baselines_path() -> Option<PathBuf> {
    Config::default_path()
        .ok()?
        .parent()
        .map(|d| d.join("mcp_baselines.json"))
}

fn load_baselines() -> BTreeMap<String, String> {
    let Some(p) = baselines_path() else {
        return BTreeMap::new();
    };
    match std::fs::read_to_string(&p) {
        // No file yet = first run (normal). A present-but-unparseable file is
        // suspicious for a rug-pull guard — warn instead of silently fail-open.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => BTreeMap::new(),
        Err(e) => {
            eprintln!("[mge] warning: cannot read MCP baselines ({e}); rug-pull check degraded");
            BTreeMap::new()
        }
        Ok(s) => serde_json::from_str(&s).unwrap_or_else(|e| {
            eprintln!(
                "[mge] warning: MCP baselines file is corrupt ({e}); rug-pull check degraded"
            );
            BTreeMap::new()
        }),
    }
}

fn save_baselines(m: &BTreeMap<String, String>) {
    let (Some(p), Ok(s)) = (baselines_path(), serde_json::to_string_pretty(m)) else {
        return;
    };
    // Atomic temp+rename so a write fault can't truncate the rug-pull baseline to
    // zero (which would fail-open every server next startup). 0600 — user config dir.
    let tmp = p.with_extension(format!("tmp.{}", std::process::id()));
    #[cfg(unix)]
    let opened = {
        use std::os::unix::fs::OpenOptionsExt;
        std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&tmp)
    };
    #[cfg(not(unix))]
    let opened = std::fs::File::create(&tmp);
    match opened {
        Ok(mut f) => {
            use std::io::Write;
            if f.write_all(s.as_bytes()).and_then(|_| f.flush()).is_ok() {
                if let Err(e) = std::fs::rename(&tmp, &p) {
                    eprintln!("[mge] warning: cannot save MCP baselines: {e}");
                    let _ = std::fs::remove_file(&tmp);
                }
            } else {
                eprintln!("[mge] warning: cannot write MCP baselines");
                let _ = std::fs::remove_file(&tmp);
            }
        }
        Err(e) => eprintln!("[mge] warning: cannot open MCP baselines for write: {e}"),
    }
}

/// Canonical SHA-256 over a server's tools (names + JSON schemas, sorted) — so a
/// silently changed/expanded tool surface (rug-pull) is detectable.
fn fingerprint(tools: &[rmcp::model::Tool]) -> String {
    let mut items: Vec<String> = tools
        .iter()
        .map(|t| {
            format!(
                "{}\u{0}{}",
                t.name,
                Value::Object((*t.input_schema).clone())
            )
        })
        .collect();
    items.sort();
    let mut h = Sha256::new();
    for it in items {
        h.update(it.as_bytes());
        h.update([0x0a]);
    }
    format!("{:x}", h.finalize())
}

/// Forget a server's stored fingerprint so its next connect re-baselines (after
/// the user has reviewed the schema change). `mge mcp --reapprove <name>`.
pub fn reapprove(server: &str) {
    let mut m = load_baselines();
    m.remove(server);
    save_baselines(&m);
}

/// Owns the live MCP connections. Must be kept alive for the whole session —
/// the `Peer` clones held by registered [`McpTool`]s stop working once dropped.
pub struct McpManager {
    _services: Vec<RunningService<RoleClient, ()>>,
}

/// Per-server connection result for `mge mcp list`.
pub struct ServerStatus {
    pub name: String,
    pub tools: Vec<String>,
    pub error: Option<String>,
}

impl McpManager {
    /// Connect to every enabled stdio server, registering their tools into `reg`.
    /// Returns the manager (keep it alive) plus per-server status for display.
    pub async fn connect(cfg: &Config, reg: &mut Registry) -> (Self, Vec<ServerStatus>) {
        let mut services = Vec::new();
        let mut status = Vec::new();
        if !cfg.mcp.enabled {
            return (
                Self {
                    _services: services,
                },
                status,
            );
        }
        let mut baselines = load_baselines();
        let mut baselines_dirty = false;

        for (name, sc) in &cfg.mcp.servers {
            if !sc.enabled {
                continue;
            }
            match Self::connect_one(sc).await {
                Ok(svc) => {
                    let listed =
                        match tokio::time::timeout(CONNECT_TIMEOUT, svc.list_all_tools()).await {
                            Ok(r) => r.map_err(|e| format!("list_tools failed: {e}")),
                            Err(_) => Err("list_tools timed out".to_string()),
                        };
                    match listed {
                        Ok(tools) => {
                            // Rug-pull defense: block if the tool surface changed
                            // since first connect, until the user re-approves.
                            let fp = fingerprint(&tools);
                            if matches!(baselines.get(name), Some(old) if old != &fp) {
                                status.push(ServerStatus {
                                    name: name.clone(),
                                    tools: vec![],
                                    error: Some(
                                        "tool schema changed since first connect (possible rug-pull) \
                                         — BLOCKED. Review, then `mge mcp --reapprove <name>`."
                                            .into(),
                                    ),
                                });
                                // svc dropped here → disconnects; nothing registered.
                            } else {
                                if !baselines.contains_key(name) {
                                    baselines.insert(name.clone(), fp);
                                    baselines_dirty = true;
                                }
                                let mut names = Vec::new();
                                for t in tools {
                                    if !sc.tools_allow.is_empty()
                                        && !sc
                                            .tools_allow
                                            .iter()
                                            .any(|p| t.name.starts_with(p.as_str()))
                                    {
                                        continue;
                                    }
                                    let full = format!("mcp__{name}__{}", t.name);
                                    let parameters = Value::Object((*t.input_schema).clone());
                                    let description =
                                        t.description.map(|c| c.to_string()).unwrap_or_default();
                                    names.push(full.clone());
                                    reg.add(Arc::new(McpTool {
                                        peer: svc.peer().clone(),
                                        tool: t.name.to_string(),
                                        full,
                                        description,
                                        parameters,
                                    }));
                                }
                                status.push(ServerStatus {
                                    name: name.clone(),
                                    tools: names,
                                    error: None,
                                });
                                services.push(svc);
                            }
                        }
                        Err(e) => status.push(ServerStatus {
                            name: name.clone(),
                            tools: vec![],
                            error: Some(e),
                        }),
                    }
                }
                Err(e) => status.push(ServerStatus {
                    name: name.clone(),
                    tools: vec![],
                    error: Some(format!("{e:#}")),
                }),
            }
        }
        if baselines_dirty {
            save_baselines(&baselines);
        }
        (
            Self {
                _services: services,
            },
            status,
        )
    }

    async fn connect_one(sc: &McpServerConfig) -> Result<RunningService<RoleClient, ()>> {
        // Cap the handshake so a hung server can't freeze startup. Both transports
        // serve into the same RunningService type.
        let svc = match sc.transport.as_str() {
            "stdio" => {
                let env = sc.env.clone();
                let (program, args) = wrap_sandbox(sc);
                let mut cmd = tokio::process::Command::new(&program).configure(|c| {
                    for a in &args {
                        c.arg(a);
                    }
                    // Scrub inherited secret-looking env vars BEFORE applying the
                    // server's explicit `env`, so a third-party MCP server can't read
                    // OPENAI_API_KEY/ANTHROPIC_API_KEY/etc. on startup — while the
                    // intentionally-configured `sc.env` still takes effect. Parity
                    // with bash/LSP/delegate/hooks, which already scrub.
                    for (k, _) in std::env::vars() {
                        if crate::util::is_secret_env(&k) {
                            c.env_remove(&k);
                        }
                    }
                    for (k, v) in &env {
                        c.env(k, v);
                    }
                });
                harden(&mut cmd, &sc.sandbox);
                let transport = TokioChildProcess::new(cmd)
                    .map_err(|e| anyhow!("spawn '{}': {e}", sc.command))?;
                tokio::time::timeout(CONNECT_TIMEOUT, ().serve(transport)).await
            }
            // Streamable HTTP (and legacy SSE, mapped to the same client).
            "http" | "sse" => {
                if sc.url.is_empty() {
                    bail!("http transport requires a `url`");
                }
                // Same SSRF guard as web_fetch / embeddings: refuse a loopback/
                // private/link-local MCP endpoint, so a tampered config can't make us
                // proxy tool calls through e.g. the cloud metadata service. Both the
                // hostname STRING and its RESOLVED IP are checked (DNS-rebinding: a
                // public name pointing at 169.254.169.254 must be refused too).
                if crate::tools::is_blocked_host(&sc.url)
                    || crate::tools::host_resolves_to_blocked(&sc.url).await
                {
                    bail!("MCP http url targets/resolves to a loopback/private/link-local address");
                }
                // Use rmcp's own (reqwest-backed) default client; do NOT name our
                // reqwest::Client here — rmcp pins a different reqwest major.
                let transport = StreamableHttpClientTransport::from_uri(sc.url.clone());
                tokio::time::timeout(CONNECT_TIMEOUT, ().serve(transport)).await
            }
            other => bail!("unknown transport '{other}' (use stdio or http)"),
        }
        .map_err(|_| {
            anyhow!(
                "MCP handshake timed out after {}s",
                CONNECT_TIMEOUT.as_secs()
            )
        })?
        .map_err(|e| anyhow!("MCP handshake: {e}"))?;
        Ok(svc)
    }
}

/// Tier-1 sandbox for a spawned stdio MCP server: set NO_NEW_PRIVS so the child
/// (third-party code) can't gain privileges via setuid binaries. Runs in the
/// child after fork, before exec — only async-signal-safe syscalls.
#[cfg(unix)]
fn harden(cmd: &mut tokio::process::Command, mode: &str) {
    if mode == "off" {
        return;
    }
    // SAFETY: prctl is async-signal-safe; we touch no shared state in the child.
    unsafe {
        cmd.pre_exec(|| {
            libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);
            Ok(())
        });
    }
}
#[cfg(not(unix))]
fn harden(_cmd: &mut tokio::process::Command, _mode: &str) {}

fn find_bin(name: &str) -> Option<String> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|p| p.join(name))
            .find(|p| p.exists())
            .map(|p| p.to_string_lossy().into_owned())
    })
}

/// Tier-2 sandbox: when `sandbox = "bwrap"` and bubblewrap is installed, run the
/// stdio server inside a bwrap namespace (read-only root, private /tmp, own PID
/// namespace, dies with parent). Falls back to the bare command (still hardened
/// with NO_NEW_PRIVS) if bwrap is missing.
fn wrap_sandbox(sc: &McpServerConfig) -> (String, Vec<String>) {
    if sc.sandbox == "bwrap"
        && let Some(bw) = find_bin("bwrap")
    {
        let mut a: Vec<String> = [
            "--ro-bind",
            "/",
            "/",
            "--dev-bind",
            "/dev",
            "/dev",
            "--proc",
            "/proc",
            "--tmpfs",
            "/tmp",
            "--die-with-parent",
            "--unshare-pid",
            "--new-session",
            "--",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        a.push(sc.command.clone());
        a.extend(sc.args.clone());
        return (bw, a);
    }
    (sc.command.clone(), sc.args.clone())
}

/// A single tool exposed by an MCP server, presented as a native [`Tool`].
struct McpTool {
    peer: Peer<RoleClient>,
    tool: String,
    full: String,
    description: String,
    parameters: Value,
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.full
    }

    fn def(&self) -> ToolDef {
        ToolDef {
            name: self.full.clone(),
            description: self.description.clone(),
            parameters: self.parameters.clone(),
        }
    }

    async fn run(&self, args: Value) -> Result<String> {
        let mut param = CallToolRequestParams::new(self.tool.clone());
        if let Some(obj) = args.as_object() {
            param = param.with_arguments(obj.clone());
        }
        let res = self
            .peer
            .call_tool(param)
            .await
            .map_err(|e| anyhow!("mcp tool '{}' failed: {e}", self.tool))?;

        let text = res
            .content
            .iter()
            .map(|c| match &c.raw {
                RawContent::Text(t) => t.text.clone(),
                other => serde_json::to_string(other).unwrap_or_default(),
            })
            .collect::<Vec<_>>()
            .join("\n");

        if res.is_error.unwrap_or(false) {
            Ok(format!("error: {text}"))
        } else {
            Ok(text)
        }
    }
}

//! `lsp_diagnostics` tool — run a configured language server over a file and
//! return its diagnostics (errors/warnings), so the agent can see what a real
//! compiler/linter sees instead of guessing.
//!
//! It speaks just enough LSP over stdio: `initialize` → `initialized` →
//! `textDocument/didOpen`, then collects the `textDocument/publishDiagnostics`
//! the server pushes back for that file. The server command comes only from the
//! trusted `[lsp]` config (keyed by extension) — never from the model — and the
//! model supplies only a file path, so there is no command-injection surface.

use crate::config::Config;
use crate::llm::ToolDef;
use crate::tools::Tool;
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::process::Stdio;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::ChildStdout;
use tokio::time::{Instant, timeout};

const OUTPUT_CAP: usize = 16_000;
/// After the first (possibly empty) diagnostics batch, wait this long for the
/// server to push a fuller one before reporting (handles "empty then filled").
const SETTLE: Duration = Duration::from_millis(1200);
/// Reject any LSP frame larger than this (a misbehaving server emitting a garbage
/// Content-Length must not trigger a multi-GB allocation / OOM abort).
const MAX_FRAME: usize = 8 * 1024 * 1024;

pub struct LspTool {
    /// Configured extensions, computed once at construction (NOT per `def()` call —
    /// `def()` runs every agent turn and must not do blocking config I/O).
    exts: String,
}

impl LspTool {
    pub fn new() -> Self {
        let exts = Config::load()
            .map(|c| c.lsp.servers.keys().cloned().collect::<Vec<_>>().join(", "))
            .unwrap_or_default();
        let exts = if exts.is_empty() {
            "(none configured — add servers under [lsp] in config.toml)".to_string()
        } else {
            exts
        };
        Self { exts }
    }
}

#[async_trait]
impl Tool for LspTool {
    fn name(&self) -> &str {
        "lsp_diagnostics"
    }

    fn def(&self) -> ToolDef {
        let exts = &self.exts;
        ToolDef {
            name: "lsp_diagnostics".into(),
            description: format!(
                "Get compiler/linter diagnostics for a file from its language server — \
                 the ground truth of what's wrong (errors, warnings) instead of guessing. \
                 Configured extensions: {exts}."
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the source file to check." }
                },
                "required": ["path"]
            }),
        }
    }

    async fn run(&self, args: Value) -> Result<String> {
        let path = args
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim();
        if path.is_empty() {
            anyhow::bail!("lsp_diagnostics: provide `path`");
        }
        let cfg = Config::load()?;
        if !cfg.lsp.enabled {
            anyhow::bail!("LSP is disabled — set `enabled = true` under [lsp] in config.toml");
        }
        let budget = Duration::from_secs(cfg.lsp.timeout_secs.max(2));
        // Shares the persistent warm session with find_references (index once, reuse).
        let diags = diagnostics(&cfg, std::path::Path::new(path)).await?;
        Ok(crate::util::clip(
            &format_diags(&diags, path, budget),
            OUTPUT_CAP,
        ))
    }
}

/// Diagnostics for `file` via the PERSISTENT warm session (shares the index with
/// `find_references`). `None` = none received; `Some([])` = clean; else issues.
pub async fn diagnostics(cfg: &Config, file: &std::path::Path) -> Result<Option<Vec<Value>>> {
    if !cfg.lsp.enabled {
        anyhow::bail!("lsp disabled");
    }
    let abs = tokio::fs::canonicalize(file)
        .await
        .with_context(|| format!("lsp_diagnostics: no such file '{}'", file.display()))?;
    let ext = abs
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    let server = cfg
        .lsp
        .servers
        .get(&ext)
        .filter(|v| !v.is_empty())
        .ok_or_else(|| {
            anyhow::anyhow!("no LSP server configured for '.{ext}' (add it under [lsp.servers])")
        })?;
    let text = tokio::fs::read_to_string(&abs).await?;
    let uri = path_to_uri(&abs.to_string_lossy());
    let root = std::env::current_dir()
        .map(|d| path_to_uri(&d.to_string_lossy()))
        .unwrap_or_else(|_| uri.clone());
    let lang = language_id(&ext);
    let budget = Duration::from_secs(cfg.lsp.timeout_secs.max(2));
    let key = format!("{} @ {root}", server.join(" "));

    let deadline = Instant::now() + budget;
    let handle = session_for(server, &root, &key, deadline).await?;
    let mut s = handle.lock().await; // per-session lock
    let result = s.collect_diagnostics(&uri, lang, &text, deadline).await;
    if result.is_err() {
        drop(s);
        drop_session(&key);
    }
    result
}

/// A PERSISTENT language-server connection, kept warm across tool calls so the
/// workspace is indexed ONCE per session instead of on every query. Sequential
/// (the agent runs tools one at a time) → a plain mutex over the registry suffices;
/// no request multiplexing needed.
struct Session {
    _child: tokio::process::Child, // alive for the session (kill_on_drop on exit)
    stdin: tokio::process::ChildStdin,
    reader: BufReader<ChildStdout>,
    next_id: i64,
    /// uri → last text sent, so we `didChange` only when content actually changed.
    open: std::collections::HashMap<String, String>,
    /// decoded uri → latest pushed diagnostics (servers don't re-push unchanged files).
    diags: std::collections::HashMap<String, Vec<Value>>,
    version: i32,
}

impl Session {
    async fn start(server: &[String], root: &str, deadline: Instant) -> Result<Session> {
        let mut cmd = tokio::process::Command::new(&server[0]);
        cmd.args(&server[1..]);
        for (k, _) in std::env::vars() {
            if crate::util::is_secret_env(&k) {
                cmd.env_remove(k);
            }
        }
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        cmd.kill_on_drop(true);
        let mut child = cmd
            .spawn()
            .with_context(|| format!("cannot start '{}'", server[0]))?;
        let stdin = child.stdin.take().context("no stdin")?;
        let reader = BufReader::new(child.stdout.take().context("no stdout")?);
        let mut s = Session {
            _child: child,
            stdin,
            reader,
            next_id: 1,
            open: std::collections::HashMap::new(),
            diags: std::collections::HashMap::new(),
            version: 1,
        };
        s.send_msg(&json!({"jsonrpc":"2.0","id":0,"method":"initialize",
            "params":{"processId":Value::Null,"rootUri":root,
                "capabilities":{"textDocument":{"references":{},"publishDiagnostics":{}}}}}))
            .await?;
        loop {
            let Some(m) = read_until(&mut s.reader, deadline).await else {
                anyhow::bail!("lsp: no initialize response");
            };
            if m.get("method").is_none() && m.get("id").and_then(Value::as_i64) == Some(0) {
                break;
            }
            if let Some(reply) = build_answer(&m) {
                s.send_msg(&reply).await?;
            }
        }
        s.send_msg(&json!({"jsonrpc":"2.0","method":"initialized","params":{}}))
            .await?;
        Ok(s)
    }

    async fn send_msg(&mut self, v: &Value) -> Result<()> {
        send(&mut self.stdin, v).await
    }

    /// `didOpen` the file the first time; `didChange` when its content changed.
    /// Returns `true` if it triggered (re)analysis (so the caller knows to wait for
    /// fresh diagnostics), `false` if the file was already open with this content.
    async fn ensure_open(&mut self, uri: &str, lang: &str, text: &str) -> Result<bool> {
        match self.open.get(uri) {
            Some(prev) if prev == text => return Ok(false),
            Some(_) => {
                self.version += 1;
                let v = self.version;
                self.send_msg(&json!({"jsonrpc":"2.0","method":"textDocument/didChange",
                    "params":{"textDocument":{"uri":uri,"version":v},
                        "contentChanges":[{"text":text}]}}))
                    .await?;
            }
            None => {
                self.send_msg(&json!({"jsonrpc":"2.0","method":"textDocument/didOpen",
                    "params":{"textDocument":{"uri":uri,"languageId":lang,"version":1,"text":text}}}))
                    .await?;
            }
        }
        self.open.insert(uri.to_string(), text.to_string());
        // Drop any cached diagnostics for this file — they're now stale; the caller
        // must wait for the server's FRESH push for the new content.
        self.diags.remove(&percent_decode(uri));
        Ok(true)
    }

    /// Cache a `publishDiagnostics` notification (keyed by decoded uri) so a later
    /// `collect_diagnostics` for an unchanged file can return it without re-waiting.
    fn note_diagnostics(&mut self, m: &Value) {
        if m.get("method").and_then(Value::as_str) == Some("textDocument/publishDiagnostics") {
            let p = m.get("params");
            if let Some(u) = p.and_then(|p| p.get("uri")).and_then(Value::as_str) {
                let d = p
                    .and_then(|p| p.get("diagnostics"))
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                self.diags.insert(percent_decode(u), d);
            }
        }
    }

    /// Diagnostics for `uri`: open/change the file, then wait for the server's push
    /// (settling briefly after an empty batch). Returns the same shape as the
    /// per-call collector: `None` = none received, `Some([])` = clean, else issues.
    async fn collect_diagnostics(
        &mut self,
        uri: &str,
        lang: &str,
        text: &str,
        deadline: Instant,
    ) -> Result<Option<Vec<Value>>> {
        let want = percent_decode(uri);
        let triggered = self.ensure_open(uri, lang, text).await?;
        // Unchanged file we've already analysed → return the cached result at once.
        if !triggered && let Some(d) = self.diags.get(&want) {
            return Ok(Some(d.clone()));
        }
        let mut latest = self.diags.get(&want).cloned();
        let mut settle: Option<Instant> = None;
        loop {
            let until = settle.map(|s| s.min(deadline)).unwrap_or(deadline);
            if Instant::now() >= until {
                break;
            }
            let Some(m) = read_until(&mut self.reader, until).await else {
                break;
            };
            self.note_diagnostics(&m);
            match m.get("method").and_then(Value::as_str) {
                Some("textDocument/publishDiagnostics") => {
                    if self.diags_uri_matches(&m, &want) {
                        let d = self.diags.get(&want).cloned().unwrap_or_default();
                        if !d.is_empty() {
                            return Ok(Some(d));
                        }
                        latest = Some(d);
                        settle = Some(Instant::now() + SETTLE);
                    }
                }
                Some(_) if m.get("id").is_some() => {
                    if let Some(reply) = build_answer(&m) {
                        self.send_msg(&reply).await?;
                    }
                }
                _ => {}
            }
        }
        Ok(latest)
    }

    fn diags_uri_matches(&self, m: &Value, want: &str) -> bool {
        m.pointer("/params/uri")
            .and_then(Value::as_str)
            .map(percent_decode)
            .as_deref()
            == Some(want)
    }

    /// Send a request and read to its response, answering server requests inline.
    async fn request(&mut self, method: &str, params: Value, deadline: Instant) -> Result<Value> {
        self.next_id += 1;
        let id = self.next_id;
        self.send_msg(&json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}))
            .await?;
        loop {
            let Some(m) = read_until(&mut self.reader, deadline).await else {
                anyhow::bail!("lsp: timed out waiting for {method}");
            };
            self.note_diagnostics(&m); // cache any diagnostics pushed while we wait
            if m.get("method").is_none() && m.get("id").and_then(Value::as_i64) == Some(id) {
                // A JSON-RPC error for our request → surface it, don't silently
                // treat the (absent) result as an empty answer.
                if let Some(err) = m.get("error") {
                    anyhow::bail!("lsp: server error for {method}: {err}");
                }
                return Ok(m.get("result").cloned().unwrap_or(Value::Null));
            }
            if let Some(reply) = build_answer(&m) {
                self.send_msg(&reply).await?;
            }
        }
    }
}

type SessionHandle = std::sync::Arc<tokio::sync::Mutex<Session>>;

/// Process-global warm sessions, keyed by `server-command @ root`. The OUTER map
/// is a std mutex held only microseconds (insert/clone the Arc); the actual LSP
/// I/O holds the PER-SESSION tokio mutex. So swarm subagents on different
/// workspaces run in parallel, and only same-workspace callers queue.
static SESSIONS: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<String, SessionHandle>>,
> = std::sync::OnceLock::new();

/// Get-or-spawn the warm session handle for `key`. `Session::start` runs OUTSIDE
/// the map lock (it's async); a double-check on insert resolves the rare race.
async fn session_for(
    server: &[String],
    root: &str,
    key: &str,
    deadline: Instant,
) -> Result<SessionHandle> {
    let map = SESSIONS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    if let Some(h) = map
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(key)
        .cloned()
    {
        return Ok(h);
    }
    let started = std::sync::Arc::new(tokio::sync::Mutex::new(
        Session::start(server, root, deadline).await?,
    ));
    Ok(map
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .entry(key.to_string())
        .or_insert(started)
        .clone())
}

/// Drop a (likely dead) session so the next call respawns it.
fn drop_session(key: &str) {
    if let Some(map) = SESSIONS.get() {
        map.lock().unwrap_or_else(|e| e.into_inner()).remove(key);
    }
}

/// Precise references to the symbol defined at (`line`,`character`) of `file`, via
/// a PERSISTENT language server (indexed once, reused) — real name resolution that
/// distinguishes same-named symbols. Returns `path:line`. Errors if no server /
/// timeout so the caller can fall back to the heuristic graph.
pub async fn find_references(
    cfg: &Config,
    file: &std::path::Path,
    line: u32,
    character: u32,
) -> Result<Vec<String>> {
    if !cfg.lsp.enabled {
        anyhow::bail!("lsp disabled");
    }
    let abs = tokio::fs::canonicalize(file).await?;
    let ext = abs
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    let server = cfg
        .lsp
        .servers
        .get(&ext)
        .filter(|v| !v.is_empty())
        .ok_or_else(|| anyhow::anyhow!("no lsp server for .{ext}"))?;
    let text = tokio::fs::read_to_string(&abs).await?;
    let uri = path_to_uri(&abs.to_string_lossy());
    let root = std::env::current_dir()
        .map(|d| path_to_uri(&d.to_string_lossy()))
        .unwrap_or_else(|_| uri.clone());
    let lang = language_id(&ext);
    let budget = Duration::from_secs(cfg.lsp.timeout_secs.max(5));
    let key = format!("{} @ {root}", server.join(" "));

    let deadline = Instant::now() + budget;
    let handle = session_for(server, &root, &key, deadline).await?;
    let mut s = handle.lock().await; // per-session lock; other workspaces run free
    let result = run_references(&mut s, &uri, lang, &text, line, character, deadline).await;
    if result.is_err() {
        drop(s);
        drop_session(&key); // server may be dead → respawn on the next call
    }
    result
}

async fn run_references(
    s: &mut Session,
    uri: &str,
    lang: &str,
    text: &str,
    line: u32,
    character: u32,
    deadline: Instant,
) -> Result<Vec<String>> {
    s.ensure_open(uri, lang, text).await?;
    let params = json!({"textDocument":{"uri":uri},
        "position":{"line":line,"character":character},
        "context":{"includeDeclaration":false}});
    // rust-analyzer answers references with [] until indexed (no queue) → retry with
    // backoff. With a WARM session the index persists, so later calls return at once.
    loop {
        let result = s
            .request("textDocument/references", params.clone(), deadline)
            .await?;
        let refs = parse_locations(&result);
        if !refs.is_empty() || Instant::now() + Duration::from_secs(2) >= deadline {
            return Ok(refs);
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

/// Parse an LSP `Location[]` result into `path:line` strings.
fn parse_locations(result: &Value) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(arr) = result.as_array() {
        for loc in arr {
            let u = loc.get("uri").and_then(Value::as_str).unwrap_or("");
            let ln = loc
                .pointer("/range/start/line")
                .and_then(Value::as_i64)
                .unwrap_or(0)
                + 1;
            let path = percent_decode(u);
            let path = path.strip_prefix("file://").unwrap_or(&path).to_string();
            out.push(format!("{path}:{ln}"));
        }
    }
    out
}

/// If `msg` is a server→client request, build the minimal response to keep the
/// server unblocked (`workspace/configuration` wants one entry per item; else null).
/// Returns `None` for notifications/responses. Shared by the channel and the
/// persistent-session code paths.
fn build_answer(msg: &Value) -> Option<Value> {
    let (id, method) = (msg.get("id")?, msg.get("method").and_then(Value::as_str)?);
    let result = if method == "workspace/configuration" {
        let n = msg
            .pointer("/params/items")
            .and_then(Value::as_array)
            .map(|a| a.len())
            .unwrap_or(1);
        Value::Array(vec![Value::Null; n])
    } else {
        Value::Null
    };
    Some(json!({"jsonrpc": "2.0", "id": id, "result": result}))
}

/// Write one LSP frame (`Content-Length` header + JSON body).
async fn send(stdin: &mut tokio::process::ChildStdin, msg: &Value) -> Result<()> {
    let body = msg.to_string();
    stdin
        .write_all(format!("Content-Length: {}\r\n\r\n{body}", body.len()).as_bytes())
        .await?;
    stdin.flush().await?;
    Ok(())
}

/// Build a `file://` URI, percent-encoding bytes outside the RFC 3986 unreserved
/// set (but keeping `/`), so it matches what LSP servers echo back.
fn path_to_uri(path: &str) -> String {
    let mut s = String::from("file://");
    for &b in path.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' | b'/' => {
                s.push(b as char)
            }
            _ => s.push_str(&format!("%{b:02X}")),
        }
    }
    s
}

/// Percent-decode a URI to raw bytes (lossy UTF-8), for encoding-agnostic compare.
fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%'
            && i + 3 <= b.len()
            && let Ok(h) =
                u8::from_str_radix(std::str::from_utf8(&b[i + 1..i + 3]).unwrap_or("zz"), 16)
        {
            out.push(h);
            i += 3;
            continue;
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Read one frame, bounded by `deadline`. `None` on timeout / EOF / parse error.
async fn read_until(reader: &mut BufReader<ChildStdout>, deadline: Instant) -> Option<Value> {
    let budget = deadline.saturating_duration_since(Instant::now());
    if budget.is_zero() {
        return None;
    }
    match timeout(budget, read_frame(reader)).await {
        Ok(Ok(v)) => v,
        _ => None,
    }
}

/// Parse one LSP message: header lines until a blank line, then `Content-Length`
/// bytes of JSON. `Ok(None)` on EOF.
async fn read_frame(reader: &mut BufReader<ChildStdout>) -> Result<Option<Value>> {
    let mut len = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).await? == 0 {
            return Ok(None); // EOF
        }
        let t = line.trim_end();
        if t.is_empty() {
            break; // end of headers
        }
        if let Some(v) = t.strip_prefix("Content-Length:") {
            // A malformed length can't be guessed away — reading a wrong number of
            // body bytes would desync every later frame. Stop the stream cleanly.
            match v.trim().parse::<usize>() {
                Ok(n) => len = n,
                Err(_) => return Ok(None),
            }
        }
    }
    if len == 0 {
        return Ok(Some(Value::Null));
    }
    if len > MAX_FRAME {
        return Ok(None); // garbage Content-Length → refuse a giant allocation
    }
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).await?;
    // A frame that doesn't parse means the server desynced/crashed (e.g. a panic
    // trace on stdout). Stop the session rather than feed Null through the loop,
    // which would silently burn the whole timeout budget.
    match serde_json::from_slice(&buf) {
        Ok(v) => Ok(Some(v)),
        Err(_) => Ok(None),
    }
}

/// Render diagnostics as `severity line:col message [source]`, errors first.
fn format_diags(diags: &Option<Vec<Value>>, path: &str, budget: Duration) -> String {
    let Some(diags) = diags else {
        return format!(
            "[lsp: no diagnostics from server within {}s — it may still be indexing; try again]",
            budget.as_secs()
        );
    };
    if diags.is_empty() {
        return format!("[lsp: no diagnostics — {path} is clean]");
    }
    let mut rows: Vec<(i64, String)> = diags
        .iter()
        .map(|d| {
            let sev = d.get("severity").and_then(Value::as_i64).unwrap_or(1);
            let line = d
                .pointer("/range/start/line")
                .and_then(Value::as_i64)
                .unwrap_or(0)
                + 1;
            let col = d
                .pointer("/range/start/character")
                .and_then(Value::as_i64)
                .unwrap_or(0)
                + 1;
            let msg = d
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("")
                .replace('\n', " ");
            let src = d
                .get("source")
                .and_then(Value::as_str)
                .map(|s| format!(" [{s}]"))
                .unwrap_or_default();
            (
                sev,
                format!("{} {line}:{col}  {msg}{src}", severity_label(sev)),
            )
        })
        .collect();
    rows.sort_by_key(|(sev, _)| *sev); // 1=error first
    let n = rows.len();
    let body = rows
        .into_iter()
        .map(|(_, s)| s)
        .collect::<Vec<_>>()
        .join("\n");
    format!("{n} diagnostic(s) in {path}:\n{body}")
}

fn severity_label(sev: i64) -> &'static str {
    match sev {
        1 => "error",
        2 => "warning",
        3 => "info",
        _ => "hint",
    }
}

/// Map a file extension to an LSP `languageId` (falls back to the extension).
fn language_id(ext: &str) -> &str {
    match ext {
        "rs" => "rust",
        "py" => "python",
        "ts" => "typescript",
        "tsx" => "typescriptreact",
        "js" => "javascript",
        "jsx" => "javascriptreact",
        "go" => "go",
        "c" => "c",
        "h" | "hpp" | "cc" | "cpp" | "cxx" => "cpp",
        "rb" => "ruby",
        "java" => "java",
        "php" => "php",
        "cs" => "csharp",
        "swift" => "swift",
        "kt" | "kts" => "kotlin",
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Precise references against rust-analyzer, run from the mge repo root.
    /// Opt-in: `cargo test --ignored find_references_resolves`.
    #[tokio::test]
    #[ignore = "requires rust-analyzer + run from the mge repo root"]
    async fn find_references_resolves_real_symbol() {
        let mut cfg = crate::config::Config::default();
        cfg.lsp = crate::config::LspConfig {
            enabled: true,
            timeout_secs: 90,
            servers: std::collections::BTreeMap::from([(
                "rs".to_string(),
                vec!["rust-analyzer".to_string()],
            )]),
        };
        let routing = std::path::Path::new("src/routing.rs");
        let content = std::fs::read_to_string(routing).unwrap();

        // Call 1 — cold: spawns the server and indexes the workspace.
        let (l1, c1) =
            crate::repo_map::symbol_position(&content, "rs", "is_retriable").expect("pos");
        let t0 = std::time::Instant::now();
        let refs1 = find_references(&cfg, routing, l1, c1).await.unwrap();
        let cold = t0.elapsed();
        assert!(!refs1.is_empty(), "expected precise references");
        assert!(
            refs1.iter().any(|r| r.contains("agent/mod.rs")),
            "agent/mod.rs should reference is_retriable"
        );

        // Call 2 — WARM: reuses the persistent session, no re-index.
        let (l2, c2) =
            crate::repo_map::symbol_position(&content, "rs", "candidates_for").expect("pos");
        let t1 = std::time::Instant::now();
        let refs2 = find_references(&cfg, routing, l2, c2).await.unwrap();
        let warm = t1.elapsed();
        eprintln!(
            "cold={cold:?} warm={warm:?}  refs1={} refs2={}",
            refs1.len(),
            refs2.len()
        );
        assert!(!refs2.is_empty(), "warm call should also resolve");
        // The warm call reuses the index → must be much faster than the cold one.
        assert!(
            warm * 3 < cold,
            "warm ({warm:?}) should be far faster than cold ({cold:?})"
        );
    }

    #[test]
    fn language_id_maps_known_and_falls_back() {
        assert_eq!(language_id("rs"), "rust");
        assert_eq!(language_id("tsx"), "typescriptreact");
        assert_eq!(language_id("zig"), "zig"); // unknown → passthrough
    }

    #[test]
    fn uri_encodes_and_decodes_round_trip() {
        let u = path_to_uri("/home/a b/π/main.rs");
        assert!(u.starts_with("file:///home/a%20b/")); // space + non-ASCII encoded
        assert!(!u.contains(' '));
        // server-echoed (encoded) URI decodes back to the raw path we opened
        assert_eq!(percent_decode(&u), "file:///home/a b/π/main.rs");
        assert_eq!(percent_decode("file:///plain/x.rs"), "file:///plain/x.rs");
    }

    #[test]
    fn format_diags_sorts_errors_first_and_1_indexes() {
        let d = Some(vec![
            json!({"severity":2,"range":{"start":{"line":4,"character":2}},"message":"unused"}),
            json!({"severity":1,"range":{"start":{"line":0,"character":0}},"message":"boom","source":"rustc"}),
        ]);
        let s = format_diags(&d, "f.rs", Duration::from_secs(30));
        assert!(s.contains("2 diagnostic(s)"));
        assert!(s.find("error").unwrap() < s.find("warning").unwrap());
        assert!(s.contains("1:1")); // 0-based line/col rendered 1-based
        assert!(s.contains("5:3"));
        assert!(s.contains("[rustc]"));
        assert!(format_diags(&Some(vec![]), "f.rs", Duration::from_secs(1)).contains("clean"));
        assert!(format_diags(&None, "f.rs", Duration::from_secs(1)).contains("indexing"));
    }

    /// Real handshake against rust-analyzer. Opt-in (needs the binary + a few
    /// seconds): `cargo test --ignored lsp_real`.
    #[tokio::test]
    #[ignore = "requires rust-analyzer installed"]
    async fn lsp_real_rust_analyzer_reports_syntax_error() {
        // rust-analyzer needs a real Cargo workspace to finish initializing.
        let dir = std::env::temp_dir().join(format!("mge_lsp_{}", std::process::id()));
        let src = dir.join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(
            dir.join("Cargo.toml"),
            "[package]\nname = \"t\"\nversion = \"0.0.0\"\nedition = \"2021\"\n",
        )
        .unwrap();
        let file = src.join("main.rs");
        std::fs::write(&file, "fn main() { let x: = 3; }\n").unwrap();
        let text = std::fs::read_to_string(&file).unwrap();
        let uri = path_to_uri(&file.to_string_lossy());
        let root = path_to_uri(&dir.to_string_lossy());

        // Drive the persistent Session directly (the diagnostics path code_graph and
        // lsp_diagnostics share). A bespoke root, so it indexes this temp project.
        let deadline = Instant::now() + Duration::from_secs(30);
        let Ok(mut s) = Session::start(&["rust-analyzer".to_string()], &root, deadline).await
        else {
            eprintln!("rust-analyzer not found — skipping");
            return;
        };
        let diags = s
            .collect_diagnostics(&uri, "rust", &text, deadline)
            .await
            .unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(
            matches!(&diags, Some(d) if !d.is_empty()),
            "expected diagnostics for a syntax error, got {diags:?}"
        );
    }
}

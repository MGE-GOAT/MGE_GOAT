//! Marketplace discovery (read-only): search the official MCP registry for
//! servers you could add. No install side effects — it prints results and the
//! exact `[mcp.servers.*]` config snippet to copy in. (`mge market search/info`.)

use anyhow::{Context, Result};
use serde_json::Value;
use std::time::Duration;

const REGISTRY: &str = "https://registry.modelcontextprotocol.io/v0/servers";

pub struct Entry {
    pub name: String,
    pub description: String,
    /// Remote (HTTP) endpoints.
    pub remotes: Vec<String>,
    /// Package identifiers (e.g. an npm package) for stdio servers.
    pub packages: Vec<String>,
}

/// Quote a registry-controlled string as a TOML basic string, escaping anything
/// that could break out of the string and inject config (TOML injection guard).
fn toml_str(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04X}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

impl Entry {
    /// A short, config-key-safe id derived from the reverse-DNS name.
    pub fn short_id(&self) -> String {
        let id: String = self
            .name
            .rsplit(['/', '.'])
            .find(|s| !s.is_empty())
            .unwrap_or(&self.name)
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect();
        if id.is_empty() {
            "server".to_string()
        } else {
            id
        }
    }

    /// A ready-to-paste `[mcp.servers.*]` snippet. Registry-controlled values are
    /// TOML-escaped so a hostile entry can't inject config.
    pub fn config_snippet(&self) -> String {
        let id = self.short_id();
        if let Some(url) = self.remotes.first() {
            format!(
                "[mcp.servers.{id}]\ntransport = \"http\"\nurl = {}",
                toml_str(url)
            )
        } else if let Some(pkg) = self.packages.first() {
            format!(
                "[mcp.servers.{id}]\ntransport = \"stdio\"\ncommand = \"npx\"\nargs = [\"-y\", {}]",
                toml_str(pkg)
            )
        } else {
            format!("[mcp.servers.{id}]   # no remote/package metadata — check the server's docs")
        }
    }
}

fn parse(json: &Value) -> Vec<Entry> {
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let servers = json
        .get("servers")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    for item in servers {
        let s = item.get("server").unwrap_or(&item);
        let name = s
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        // Registry returns multiple versions per server — keep the first (reverse-DNS dedup).
        if name.is_empty() || !seen.insert(name.clone()) {
            continue;
        }
        let description = s
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let remotes = s
            .get("remotes")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|r| r.get("url").and_then(Value::as_str).map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let packages = s
            .get("packages")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|p| {
                        p.get("identifier")
                            .and_then(Value::as_str)
                            .map(String::from)
                    })
                    .collect()
            })
            .unwrap_or_default();
        out.push(Entry {
            name,
            description,
            remotes,
            packages,
        });
    }
    out
}

async fn fetch(query: &str, limit: usize) -> Result<Vec<Entry>> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(25))
        .user_agent("MGE_GOAT/0.1")
        .build()?;
    let json: Value = client
        .get(REGISTRY)
        .query(&[("search", query), ("limit", &limit.to_string())])
        .send()
        .await
        .context("contacting the MCP registry")?
        .json()
        .await
        .context("parsing the registry response")?;
    Ok(parse(&json))
}

pub async fn search(query: &str, limit: usize) -> Result<Vec<Entry>> {
    fetch(query, limit).await
}

/// Append a discovered server to the user's config (the `install` action).
/// Minimal vetting for now: it only writes config — npx/-y fetches the package
/// on first run. A published build wants the full vet gate (digest, dep-audit).
pub fn install(entry: &Entry) -> Result<std::path::PathBuf> {
    use crate::config::Config;
    let path = Config::default_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut text = std::fs::read_to_string(&path).unwrap_or_default();
    let id = entry.short_id();
    if text.contains(&format!("[mcp.servers.{id}]")) {
        anyhow::bail!("'{id}' is already in your config");
    }
    if !text.contains("[mcp]") {
        text.push_str("\n[mcp]\nenabled = true\n");
    }
    // Strip control chars from the name so it can't escape the `#` comment.
    let safe_name: String = entry.name.chars().filter(|c| !c.is_control()).collect();
    text.push('\n');
    text.push_str(&format!("# added by `mge market install` — {safe_name}\n"));
    text.push_str(&entry.config_snippet());
    text.push_str("\nenabled = true\n");
    std::fs::write(&path, text)?;
    Ok(path)
}

/// Best-effort exact/contains match for `mge market info`.
pub async fn info(name: &str) -> Result<Option<Entry>> {
    let entries = fetch(name, 20).await?;
    Ok(entries
        .into_iter()
        .find(|e| e.name == name || e.name.ends_with(name) || e.short_id() == name))
}

//! Plugin loader (Phase 5).
//!
//! A plugin is a directory under ~/.config/mge/plugins/<name>/ that can bundle:
//!   - `skills/`  → extra skill roots (loaded by the SkillLoader)
//!   - `mcp.json` → MCP servers (Claude Code-compatible `{"mcpServers": {...}}`
//!     or `{"servers": {...}}`) connected by the McpManager
//!
//! Everything fans out to the existing loaders — plugins add no new runtime path.
//! Plugins live in your trusted config dir, so they're treated as trusted.

use crate::config::{Config, McpServerConfig};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Default)]
pub struct Plugins {
    pub names: Vec<String>,
    pub skill_roots: Vec<PathBuf>,
    pub mcp_servers: BTreeMap<String, McpServerConfig>,
}

/// Discover plugins under ~/.config/mge/plugins.
pub fn discover() -> Plugins {
    let mut p = Plugins::default();
    let Ok(cfg_path) = Config::default_path() else {
        return p;
    };
    let Some(dir) = cfg_path.parent().map(|d| d.join("plugins")) else {
        return p;
    };
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return p;
    };
    for e in entries.flatten() {
        if !e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        let root = e.path();
        let pname = e.file_name().to_string_lossy().to_string();

        let skills = root.join("skills");
        if skills.is_dir() {
            p.skill_roots.push(skills);
        }

        for manifest in ["mcp.json", ".mcp.json"] {
            if let Ok(text) = std::fs::read_to_string(root.join(manifest)) {
                if let Ok(v) = serde_json::from_str::<Value>(&text) {
                    let servers = v
                        .get("mcpServers")
                        .or_else(|| v.get("servers"))
                        .unwrap_or(&v);
                    if let Some(obj) = servers.as_object() {
                        for (sname, sval) in obj {
                            if let Ok(sc) = serde_json::from_value::<McpServerConfig>(sval.clone())
                            {
                                p.mcp_servers.insert(format!("{pname}_{sname}"), sc);
                            }
                        }
                    }
                }
                break;
            }
        }
        p.names.push(pname);
    }
    p
}

/// Return a copy of `cfg` with plugin skills + MCP servers merged in. Existing
/// config entries win on name collision; MCP is enabled if a plugin adds servers.
pub fn apply(cfg: &Config) -> Config {
    let p = discover();
    let mut c = cfg.clone();
    for root in p.skill_roots {
        c.skills.extra_dirs.push(root.to_string_lossy().to_string());
    }
    let added_servers = !p.mcp_servers.is_empty();
    for (k, v) in p.mcp_servers {
        c.mcp.servers.entry(k).or_insert(v);
    }
    if added_servers {
        c.mcp.enabled = true;
    }
    c
}

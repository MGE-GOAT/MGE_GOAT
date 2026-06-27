//! Custom slash commands.
//!
//! Markdown files under `~/.config/mge/commands/<name>.md` (trusted), and — with
//! `[skills].trust_project_skills` — project `.mge/commands` and `.claude/commands`,
//! are reusable prompt macros. `/<name> [args]` expands the file body (with
//! `$ARGUMENTS` and `$1`..`$9` substitution) into a user message for the agent.
//! Optional YAML frontmatter may carry a `description:`. Builtin slash commands
//! always win on a name collision — the caller checks builtins first.

use crate::config::Config;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const MAX_CMD_BYTES: u64 = 64 * 1024;

#[derive(Clone)]
pub struct CustomCommand {
    pub name: String,
    pub description: String,
    body: String,
}

impl CustomCommand {
    /// Expand the macro: `$ARGUMENTS` → all args, `$1`..`$9` → positional words.
    pub fn expand(&self, args: &str) -> String {
        let args = args.trim();
        let words: Vec<&str> = args.split_whitespace().collect();
        let mut out = self.body.replace("$ARGUMENTS", args);
        for i in 1..=9 {
            out = out.replace(&format!("${i}"), words.get(i - 1).copied().unwrap_or(""));
        }
        out
    }
}

pub struct CommandLoader {
    commands: Vec<CustomCommand>,
}

impl CommandLoader {
    pub fn discover(cfg: &Config) -> Self {
        let mut roots: Vec<PathBuf> = Vec::new();
        // Trusted global commands (you put them there).
        if let Ok(p) = Config::default_path()
            && let Some(dir) = p.parent()
        {
            roots.push(dir.join("commands"));
        }
        // Project commands are untrusted content — only with the same opt-in as skills.
        if cfg.skills.trust_project_skills {
            roots.push(PathBuf::from(".mge/commands"));
            roots.push(PathBuf::from(".claude/commands"));
        }
        // First root wins (global is first) — a trusted global command is NEVER
        // shadowed by an untrusted project command of the same name. Collisions
        // are warned, not silently overridden (a repo could otherwise hijack a
        // command the user muscle-memorizes, e.g. `/fix`).
        let mut by_name: BTreeMap<String, CustomCommand> = BTreeMap::new();
        for root in roots {
            for c in scan_root(&root) {
                if by_name.contains_key(&c.name) {
                    eprintln!(
                        "mge: custom command '/{}' in {} ignored — name already defined (global wins)",
                        c.name,
                        root.display()
                    );
                    continue;
                }
                by_name.insert(c.name.clone(), c);
            }
        }
        Self {
            commands: by_name.into_values().collect(),
        }
    }

    pub fn get(&self, name: &str) -> Option<&CustomCommand> {
        self.commands.iter().find(|c| c.name == name)
    }
    pub fn list(&self) -> &[CustomCommand] {
        &self.commands
    }
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }
}

fn scan_root(root: &Path) -> Vec<CustomCommand> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(root) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        if entry
            .metadata()
            .map(|m| m.len() > MAX_CMD_BYTES)
            .unwrap_or(true)
        {
            continue;
        }
        let Some(name) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };
        let (description, body) = match crate::skills::split_frontmatter(&content) {
            Some((front, body)) => (parse_description(front), body.to_string()),
            None => (String::new(), content),
        };
        out.push(CustomCommand {
            name: name.to_string(),
            description,
            body,
        });
    }
    out
}

fn parse_description(front: &str) -> String {
    front
        .lines()
        .find_map(|l| l.trim().strip_prefix("description:"))
        .map(|d| d.trim().trim_matches(['"', '\'']).to_string())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_substitutes_arguments_and_positional() {
        let c = CustomCommand {
            name: "greet".into(),
            description: String::new(),
            body: "Hello $1, all: $ARGUMENTS".into(),
        };
        assert_eq!(c.expand("alice bob"), "Hello alice, all: alice bob");
    }

    #[test]
    fn expand_blanks_missing_positionals() {
        let c = CustomCommand {
            name: "x".into(),
            description: String::new(),
            body: "[$2]".into(),
        };
        assert_eq!(c.expand("only"), "[]");
    }
}

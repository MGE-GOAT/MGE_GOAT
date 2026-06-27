//! Markdown skill loader (Claude Code-compatible `SKILL.md`).
//!
//! Progressive disclosure: every skill's `name` + `description` is injected into
//! the agent's system prompt, but a skill's full body is only loaded when the
//! model calls the `use_skill` tool — keeping context lean. A skill is a
//! `SKILL.md` with YAML-ish frontmatter (`name:`, `description:`) followed by the
//! instruction body.
//!
//! Discovery roots (later wins on name collision): the trusted global
//! ~/.config/mge/skills and any `[skills].extra_dirs`; project roots
//! (./.mge/skills, ./.claude/skills) are only scanned when
//! `[skills].trust_project_skills = true` (a repo's skills are untrusted content
//! that could carry prompt injection). Each root may hold `SKILL.md` directly or
//! `<skill-name>/SKILL.md` (the Claude Code layout).

use crate::config::Config;
use crate::llm::ToolDef;
use crate::tools::{Registry, Tool};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub body: String,
}

pub struct SkillLoader {
    skills: Arc<Vec<Skill>>,
}

impl SkillLoader {
    /// Discover skills from the standard roots (+ configured extras).
    pub fn discover(cfg: &Config) -> Self {
        if !cfg.skills.enabled {
            return Self {
                skills: Arc::new(vec![]),
            };
        }
        let mut roots: Vec<PathBuf> = Vec::new();
        // Trusted global skills (you put them there).
        if let Ok(p) = Config::default_path()
            && let Some(dir) = p.parent()
        {
            roots.push(dir.join("skills"));
        }
        roots.extend(cfg.skills.extra_dirs.iter().map(PathBuf::from));
        // Project skills are untrusted content — only with explicit opt-in.
        if cfg.skills.trust_project_skills {
            roots.push(PathBuf::from(".mge/skills"));
            roots.push(PathBuf::from(".claude/skills"));
        }

        // Later roots override earlier on name collision (project > global).
        let mut by_name: BTreeMap<String, Skill> = BTreeMap::new();
        for root in roots {
            for s in scan_root(&root) {
                by_name.insert(s.name.clone(), s);
            }
        }
        Self {
            skills: Arc::new(by_name.into_values().collect()),
        }
    }

    pub fn count(&self) -> usize {
        self.skills.len()
    }

    pub fn list(&self) -> &[Skill] {
        &self.skills
    }

    /// The system-prompt addendum listing available skills, or `None` if empty.
    pub fn system_addendum(&self) -> Option<String> {
        if self.skills.is_empty() {
            return None;
        }
        let mut out = String::from(
            "You have SKILLS available. When a task matches one, call the `use_skill` \
             tool with its name to load detailed instructions, then follow them. Skills:\n",
        );
        for s in self.skills.iter() {
            out.push_str(&format!("  - {}: {}\n", s.name, s.description));
        }
        Some(out)
    }

    /// Register the `use_skill` tool (no-op when there are no skills).
    pub fn register(&self, reg: &mut Registry) {
        if self.skills.is_empty() {
            return;
        }
        reg.add(Arc::new(UseSkill {
            skills: self.skills.clone(),
        }));
    }
}

/// Scan one root for `SKILL.md` (directly or one level down).
fn scan_root(root: &Path) -> Vec<Skill> {
    let mut found = Vec::new();
    let Ok(entries) = std::fs::read_dir(root) else {
        return found;
    };
    // SKILL.md directly in the root
    if let Some(s) = read_skill(&root.join("SKILL.md")) {
        found.push(s);
    }
    // <name>/SKILL.md
    for e in entries.flatten() {
        if e.file_type().map(|t| t.is_dir()).unwrap_or(false)
            && let Some(s) = read_skill(&e.path().join("SKILL.md"))
        {
            found.push(s);
        }
    }
    found
}

/// Skip absurdly large SKILL.md files (avoids loading runaway content at startup).
const MAX_SKILL_BYTES: u64 = 256 * 1024;

fn read_skill(path: &Path) -> Option<Skill> {
    if std::fs::metadata(path).ok()?.len() > MAX_SKILL_BYTES {
        return None;
    }
    let content = std::fs::read_to_string(path).ok()?;
    let (front, body) = split_frontmatter(&content)?;
    let mut name = None;
    let mut description = String::new();
    for line in front.lines() {
        let line = line.trim();
        if let Some(v) = line.strip_prefix("name:") {
            name = Some(v.trim().trim_matches(['"', '\'']).to_string());
        } else if let Some(v) = line.strip_prefix("description:") {
            description = v.trim().trim_matches(['"', '\'']).to_string();
        }
    }
    // Fall back to the parent directory name if no `name:`.
    let name = name.filter(|n| !n.is_empty()).or_else(|| {
        path.parent()
            .and_then(|p| p.file_name())
            .map(|s| s.to_string_lossy().to_string())
    })?;
    Some(Skill {
        name,
        description,
        body: body.trim().to_string(),
    })
}

/// Split `---\nfrontmatter\n---\nbody`. Returns `(frontmatter, body)`. The body
/// is everything after the closing fence line — leading `-` chars are preserved
/// (so a body starting with a `---` rule isn't mangled).
pub(crate) fn split_frontmatter(content: &str) -> Option<(&str, &str)> {
    let rest = content
        .strip_prefix("---")?
        .trim_start_matches(['\n', '\r']);
    let end = rest.find("\n---")?;
    let front = &rest[..end];
    // Skip the rest of the closing fence line, then the body starts after its \n.
    let after_fence = &rest[end + 1..];
    let body = match after_fence.find('\n') {
        Some(i) => &after_fence[i + 1..],
        None => "",
    };
    Some((front, body))
}

struct UseSkill {
    skills: Arc<Vec<Skill>>,
}

#[async_trait]
impl Tool for UseSkill {
    fn name(&self) -> &str {
        "use_skill"
    }
    fn def(&self) -> ToolDef {
        let names: Vec<&str> = self.skills.iter().map(|s| s.name.as_str()).collect();
        ToolDef {
            name: "use_skill".into(),
            description: format!(
                "Load a skill's full instructions by name, then follow them. Available: {}.",
                names.join(", ")
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "the skill name"},
                    "arguments": {"type": "string", "description": "optional args substituted for $ARGUMENTS"}
                },
                "required": ["name"]
            }),
        }
    }
    async fn run(&self, args: Value) -> Result<String> {
        let name = args.get("name").and_then(Value::as_str).unwrap_or("");
        let extra = args.get("arguments").and_then(Value::as_str).unwrap_or("");
        match self.skills.iter().find(|s| s.name == name) {
            Some(s) => Ok(s.body.replace("$ARGUMENTS", extra)),
            None => Ok(format!(
                "unknown skill '{name}'. available: {}",
                self.skills
                    .iter()
                    .map(|s| s.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_frontmatter_and_body() {
        let md = "---\nname: foo\ndescription: does foo\n---\n# Body\nstep one $ARGUMENTS\n";
        let (front, body) = split_frontmatter(md).expect("frontmatter");
        assert!(front.contains("name: foo"));
        assert!(front.contains("description: does foo"));
        assert!(body.trim_start().starts_with("# Body"));
        assert!(body.contains("$ARGUMENTS"));
    }

    #[test]
    fn no_frontmatter_returns_none() {
        assert!(split_frontmatter("# just a heading\n").is_none());
    }
}

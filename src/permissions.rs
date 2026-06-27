//! Tiered permission policy + per-tool allow/ask/deny rules.
//!
//! A single [`PermissionPolicy`] is consulted in [`crate::tools::Registry::dispatch`]
//! before any tool runs — not inside individual tools — so every tool (built-in
//! or MCP) is gated uniformly. Evaluation order, with **deny as the unconditional
//! security primitive**:
//!
//! 1. explicit `deny` rule       → `Deny` (fires even in `Yolo`)
//! 2. `Plan` mode + mutating tool → `Deny` (structural block, *before* allow)
//! 3. explicit `ask` rule        → `Ask`
//! 4. explicit `allow` rule      → `Allow`
//! 5. mode default
//!
//! `Ask` with `can_prompt == false` falls through to `Allow` (a raw-mode TUI or a
//! piped/headless run can't prompt on stdin); `Plan` mode is the explicit
//! read-only gate for those contexts.
//!
//! Rule syntax: a bare tool name (`"bash"`, `"write_file"`, `"mcp__server__tool"`),
//! a wildcard (`"mcp__*"`), or a bash-command pattern (`"bash:rm -rf *"`). `*`
//! matches any run of characters including `/`; `?` matches one. Patterns are
//! plain wildcards (never a regex/glob that could fail to compile), so a
//! misconfigured rule can't silently stop matching.

/// Permission mode — the broad posture, cycled with Shift+Tab in the TUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Ask before mutating tools (bash/write/edit); allow reads.
    Default,
    /// Auto-allow file edits; bash still asks (or allows when can't prompt).
    AcceptEdits,
    /// Read-only: structurally block every mutating tool, ignoring allow rules.
    Plan,
    /// Allow everything (explicit `deny` rules are still honored).
    Yolo,
}

impl From<&str> for Mode {
    fn from(s: &str) -> Self {
        match s.trim().to_lowercase().replace(['_', '-'], "").as_str() {
            "acceptedits" => Mode::AcceptEdits,
            "plan" => Mode::Plan,
            "yolo" => Mode::Yolo,
            _ => Mode::Default,
        }
    }
}

impl Mode {
    pub fn label(self) -> &'static str {
        match self {
            Mode::Default => "default",
            Mode::AcceptEdits => "accept-edits",
            Mode::Plan => "plan",
            Mode::Yolo => "yolo",
        }
    }
    /// Cycle order for Shift+Tab: default → accept-edits → plan → yolo → …
    pub fn next(self) -> Mode {
        match self {
            Mode::Default => Mode::AcceptEdits,
            Mode::AcceptEdits => Mode::Plan,
            Mode::Plan => Mode::Yolo,
            Mode::Yolo => Mode::Default,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Ask,
    Deny,
}

/// Per-tool allow / ask / deny rule lists (patterns as described in the module doc).
#[derive(Debug, Clone, Default)]
pub struct PermissionRules {
    pub allow: Vec<String>,
    pub ask: Vec<String>,
    pub deny: Vec<String>,
}

/// The live policy: mode + rules + whether the context can prompt on stdin.
#[derive(Debug, Clone)]
pub struct PermissionPolicy {
    pub mode: Mode,
    pub rules: PermissionRules,
    pub can_prompt: bool,
}

impl Default for PermissionPolicy {
    fn default() -> Self {
        Self::default_cli()
    }
}

impl PermissionPolicy {
    /// Interactive CLI default: ask on mutating tools (there's a TTY to prompt on).
    pub fn default_cli() -> Self {
        Self {
            mode: Mode::Default,
            rules: PermissionRules::default(),
            can_prompt: true,
        }
    }

    /// Build from `[permissions]` config. `can_prompt` is set by the caller per
    /// context (CLI true, TUI false).
    pub fn from_config(pc: &crate::config::PermissionsConfig, can_prompt: bool) -> Self {
        let mode = pc.mode.as_deref().map(Mode::from).unwrap_or(Mode::Default);
        Self {
            mode,
            rules: PermissionRules {
                allow: pc.allow.clone(),
                ask: pc.ask.clone(),
                deny: pc.deny.clone(),
            },
            can_prompt,
        }
    }

    /// Decide whether `tool` (with optional bash `command`) may run.
    pub fn decide(&self, tool: &str, bash_cmd: Option<&str>) -> Decision {
        // 1. deny is unconditional — fires even in Yolo.
        if self
            .rules
            .deny
            .iter()
            .any(|r| rule_matches(r, tool, bash_cmd))
        {
            return Decision::Deny;
        }
        // 2. Plan mode hard-blocks mutating tools, before any allow rule.
        if self.mode == Mode::Plan && is_mutating(tool) {
            return Decision::Deny;
        }
        // 3. explicit ask.
        if self
            .rules
            .ask
            .iter()
            .any(|r| rule_matches(r, tool, bash_cmd))
        {
            return self.ask_or_allow();
        }
        // 4. explicit allow.
        if self
            .rules
            .allow
            .iter()
            .any(|r| rule_matches(r, tool, bash_cmd))
        {
            return Decision::Allow;
        }
        // 5. mode default.
        match self.mode {
            Mode::Yolo => Decision::Allow,
            // Non-mutating tools in plan mode (mutating already denied at step 2).
            Mode::Plan => Decision::Allow,
            Mode::AcceptEdits => {
                // Edits auto-apply; bash and delegate (runs an external agent) still ask.
                if tool == "bash" || tool == "delegate" {
                    self.ask_or_allow()
                } else {
                    Decision::Allow
                }
            }
            Mode::Default => {
                if is_mutating(tool) {
                    self.ask_or_allow()
                } else {
                    Decision::Allow
                }
            }
        }
    }

    fn ask_or_allow(&self) -> Decision {
        if self.can_prompt {
            Decision::Ask
        } else {
            Decision::Allow
        }
    }
}

/// Tools that change the world — gated by `Default`/`Plan`. MCP tools are treated
/// as mutating (conservative: we can't know their side effects). `spawn_agent` is
/// included too: even though its children enforce the inherited policy, the
/// spawn itself should be visible/gated (and Plan mode must block it).
pub fn is_mutating(tool: &str) -> bool {
    // `delegate` and `spawn_agent` hand off to agents that can edit files / run
    // commands, so gate the invocation itself (the child also enforces its policy).
    matches!(
        tool,
        "write_file" | "edit_file" | "bash" | "delegate" | "spawn_agent"
    ) || tool.starts_with("mcp__")
}

/// True if `rule` matches the tool call. `bash:<pattern>` rules match the bash
/// command; all other rules wildcard-match the tool name.
///
/// A `bash:` pattern matches the whole command OR any `;`/`|`/`&`/newline-
/// separated segment, so a deny like `bash:rm *` still fires inside a chained
/// command such as `true; rm -rf /` (a deny rule must not be prefix-bypassable).
fn rule_matches(rule: &str, tool: &str, bash_cmd: Option<&str>) -> bool {
    if let Some(pat) = rule.strip_prefix("bash:") {
        let pat = pat.trim();
        return tool == "bash"
            && bash_cmd.is_some_and(|c| {
                wildcard_match(pat, c.trim())
                    || c.split([';', '|', '&', '\n'])
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .any(|seg| wildcard_match(pat, seg))
            });
    }
    wildcard_match(rule, tool)
}

/// Classic `*`/`?` wildcard match. `*` matches any run of chars (including `/`),
/// `?` matches exactly one. No regex, so it can never fail to compile.
pub(crate) fn wildcard_match(pat: &str, text: &str) -> bool {
    let (p, t): (Vec<char>, Vec<char>) = (pat.chars().collect(), text.chars().collect());
    let (mut pi, mut ti) = (0usize, 0usize);
    let (mut star, mut mark): (Option<usize>, usize) = (None, 0);
    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            mark = ti;
            pi += 1;
        } else if let Some(s) = star {
            pi = s + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pol(
        mode: Mode,
        allow: &[&str],
        ask: &[&str],
        deny: &[&str],
        can_prompt: bool,
    ) -> PermissionPolicy {
        PermissionPolicy {
            mode,
            rules: PermissionRules {
                allow: allow.iter().map(|s| s.to_string()).collect(),
                ask: ask.iter().map(|s| s.to_string()).collect(),
                deny: deny.iter().map(|s| s.to_string()).collect(),
            },
            can_prompt,
        }
    }

    #[test]
    fn deny_overrides_allow_even_in_yolo() {
        let p = pol(Mode::Yolo, &["bash"], &[], &["bash:rm *"], true);
        assert_eq!(p.decide("bash", Some("rm -rf /")), Decision::Deny);
        assert_eq!(p.decide("bash", Some("ls")), Decision::Allow);
    }

    #[test]
    fn deny_not_bypassable_by_command_prefix() {
        // A chained command must not slip a denied verb past the deny rule.
        let p = pol(Mode::Yolo, &["bash"], &[], &["bash:rm *"], true);
        assert_eq!(p.decide("bash", Some("true; rm -rf /")), Decision::Deny);
        assert_eq!(
            p.decide("bash", Some("echo hi && rm -rf x")),
            Decision::Deny
        );
        assert_eq!(p.decide("bash", Some("cat a | rm b")), Decision::Deny);
        assert_eq!(p.decide("bash", Some("echo rm")), Decision::Allow); // not actually running rm
    }

    #[test]
    fn plan_blocks_writes_ignoring_allow() {
        let p = pol(Mode::Plan, &["bash", "write_file"], &[], &[], true);
        assert_eq!(p.decide("bash", Some("echo hi")), Decision::Deny);
        assert_eq!(p.decide("write_file", None), Decision::Deny);
        // reads still allowed in plan mode
        assert_eq!(p.decide("read_file", None), Decision::Allow);
    }

    #[test]
    fn default_asks_on_mutating_allows_reads() {
        let p = pol(Mode::Default, &[], &[], &[], true);
        assert_eq!(p.decide("bash", Some("ls")), Decision::Ask);
        assert_eq!(p.decide("write_file", None), Decision::Ask);
        assert_eq!(p.decide("grep", None), Decision::Allow);
    }

    #[test]
    fn accept_edits_allows_edits_asks_bash_but_allows_when_cannot_prompt() {
        let asking = pol(Mode::AcceptEdits, &[], &[], &[], true);
        assert_eq!(asking.decide("write_file", None), Decision::Allow);
        assert_eq!(asking.decide("bash", Some("ls")), Decision::Ask);
        // TUI: can't prompt → bash falls through to Allow (preserves old behavior)
        let tui = pol(Mode::AcceptEdits, &[], &[], &[], false);
        assert_eq!(tui.decide("bash", Some("ls")), Decision::Allow);
    }

    #[test]
    fn bash_pattern_and_wildcards() {
        assert!(wildcard_match("rm -rf *", "rm -rf /tmp/x"));
        assert!(wildcard_match("mcp__*", "mcp__ruflo__agent_spawn"));
        assert!(!wildcard_match("rm *", "git status"));
        assert!(wildcard_match("a?c", "abc"));
        assert!(!wildcard_match("a?c", "ac"));
    }

    #[test]
    fn mode_parsing_and_cycle() {
        assert_eq!(Mode::from("acceptEdits"), Mode::AcceptEdits);
        assert_eq!(Mode::from("accept_edits"), Mode::AcceptEdits);
        assert_eq!(Mode::from("PLAN"), Mode::Plan);
        assert_eq!(Mode::from("nonsense"), Mode::Default);
        assert_eq!(Mode::Default.next().next().next().next(), Mode::Default);
    }
}

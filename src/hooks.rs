//! Lifecycle hooks — user shell commands fired around every tool call.
//!
//! A `PreToolUse` hook that exits non-zero **blocks** the tool (its output
//! becomes the block reason the model sees); `PostToolUse` hooks run after a
//! tool succeeds, for side effects like auto-formatting. Hooks are defined only
//! in the trusted `[hooks]` config — never by the model — so a hook command is
//! as trusted as the user's own shell.
//!
//! Wiring is zero-touch: [`runner`] lazily builds a process-global [`HookRunner`]
//! from config on first tool dispatch, so no call site has to thread it through.
//! `MGE_HOOK=1` is exported into every hook child; if that hook re-invokes mge,
//! the child sees the flag and fires no further hooks (no infinite recursion).

use crate::config::{Config, HookEntry};
use crate::permissions::wildcard_match;
use std::process::Stdio;
use std::sync::OnceLock;
use std::time::Duration;

/// Bytes kept from a hook's output (shown to the model on a block).
const HOOK_OUTPUT_CAP: usize = 4_096;
/// Cap for the MGE_TOOL_ARGS / MGE_TOOL_RESULT env vars passed to hooks. Well
/// under Linux ARG_MAX (~2 MB) so a large write/edit can't make `execve` fail
/// with E2BIG — which `run_hook` would otherwise misreport as a hook block.
const HOOK_ENV_CAP: usize = 96_000;

pub struct HookRunner {
    entries: Vec<HookEntry>,
    timeout: Duration,
}

static RUNNER: OnceLock<Option<HookRunner>> = OnceLock::new();

/// The process-global hook runner, or `None` when hooks are disabled, unconfigured,
/// or we are already running inside a hook (anti-recursion). Built once from config.
pub fn runner() -> Option<&'static HookRunner> {
    RUNNER.get_or_init(HookRunner::from_config).as_ref()
}

impl HookRunner {
    fn from_config() -> Option<Self> {
        // If we're already a child spawned by a hook, fire nothing (no recursion).
        if std::env::var_os("MGE_HOOK").is_some() {
            return None;
        }
        // A config-load failure (vs. hooks simply being absent) is worth a word —
        // otherwise a TOML typo silently disables every hook for the session.
        let cfg = match Config::load() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[mge] warning: hooks disabled — config failed to load: {e}");
                return None;
            }
        };
        if !cfg.hooks.enabled || cfg.hooks.hooks.is_empty() {
            return None;
        }
        Some(Self {
            entries: cfg.hooks.hooks,
            timeout: Duration::from_secs(cfg.hooks.timeout_secs.max(1)),
        })
    }

    fn matching<'a>(&'a self, event: &str, tool: &'a str) -> impl Iterator<Item = &'a HookEntry> {
        self.entries.iter().filter(move |h| {
            h.event.eq_ignore_ascii_case(event) && wildcard_match(&h.matcher, tool)
        })
    }

    /// Run `PreToolUse` hooks. `Err(reason)` blocks the tool; `Ok(())` proceeds.
    pub async fn pre_tool_use(&self, tool: &str, args: &str) -> Result<(), String> {
        let args = crate::util::clip(args, HOOK_ENV_CAP);
        for h in self.matching("PreToolUse", tool) {
            let env = [("MGE_TOOL", tool), ("MGE_TOOL_ARGS", args.as_str())];
            let (ok, out) = run_hook(&h.command, &env, self.timeout).await;
            if !ok {
                let why = if out.trim().is_empty() {
                    format!("PreToolUse hook `{}` blocked this call", h.command)
                } else {
                    format!("blocked by PreToolUse hook:\n{}", out.trim())
                };
                return Err(why);
            }
        }
        Ok(())
    }

    /// Run `PostToolUse` hooks for their side effects. Exit status is ignored —
    /// a failing formatter must not fail an already-completed tool call.
    pub async fn post_tool_use(&self, tool: &str, result: &str) {
        let result = crate::util::clip(result, HOOK_ENV_CAP);
        for h in self.matching("PostToolUse", tool) {
            let env = [("MGE_TOOL", tool), ("MGE_TOOL_RESULT", result.as_str())];
            // A failing post-hook must NOT fail the (already-done) tool call, but a
            // silently broken formatter/linter is worse than a noisy one — leave a trace.
            let (ok, out) = run_hook(&h.command, &env, self.timeout).await;
            if !ok && !out.trim().is_empty() {
                eprintln!(
                    "[mge] warning: PostToolUse hook failed ({}): {}",
                    h.command,
                    crate::util::clip(out.trim(), 200)
                );
            } else if !ok {
                eprintln!("[mge] warning: PostToolUse hook failed: {}", h.command);
            }
        }
    }
}

/// Run one hook command via `sh -c`, returning `(success, captured_output)`.
/// Secret-looking env vars are stripped; `MGE_HOOK=1` plus the supplied vars are
/// set; the child is killed on timeout (no orphan).
async fn run_hook(cmd: &str, env: &[(&str, &str)], timeout: Duration) -> (bool, String) {
    let mut command = tokio::process::Command::new("sh");
    command.arg("-c").arg(cmd);
    for (k, _) in std::env::vars() {
        if crate::util::is_secret_env(&k) {
            command.env_remove(k);
        }
    }
    command.env("MGE_HOOK", "1");
    for (k, v) in env {
        command.env(k, v);
    }
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    command.kill_on_drop(true);

    let mut child = match command.spawn() {
        Ok(c) => c,
        Err(e) => return (false, format!("[hook failed to start: {e}]")),
    };
    let out = child.stdout.take();
    let err = child.stderr.take();
    let work = async move {
        let (o, e) = match (out, err) {
            (Some(o), Some(e)) => tokio::join!(
                crate::util::drain_capped(o, HOOK_OUTPUT_CAP),
                crate::util::drain_capped(e, HOOK_OUTPUT_CAP)
            ),
            _ => (Vec::new(), Vec::new()),
        };
        (child.wait().await, o, e)
    };
    match tokio::time::timeout(timeout, work).await {
        Err(_) => (
            false,
            format!("[hook timed out after {}s]", timeout.as_secs()),
        ),
        Ok((status, o, e)) => {
            let mut s = String::from_utf8_lossy(&o).into_owned();
            s.push_str(&String::from_utf8_lossy(&e));
            (
                status.map(|st| st.success()).unwrap_or(false),
                crate::util::clip(s.trim(), HOOK_OUTPUT_CAP),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(event: &str, matcher: &str) -> HookEntry {
        HookEntry {
            event: event.into(),
            matcher: matcher.into(),
            command: "true".into(),
        }
    }

    #[test]
    fn matching_filters_by_event_and_glob() {
        let hr = HookRunner {
            entries: vec![entry("PreToolUse", "write_*"), entry("PostToolUse", "*")],
            timeout: Duration::from_secs(1),
        };
        assert_eq!(hr.matching("PreToolUse", "write_file").count(), 1);
        assert_eq!(hr.matching("pretooluse", "write_file").count(), 1); // case-insensitive
        assert_eq!(hr.matching("PreToolUse", "bash").count(), 0); // glob miss
        assert_eq!(hr.matching("PostToolUse", "bash").count(), 1);
    }

    #[tokio::test]
    async fn run_hook_reports_status_and_passes_env() {
        let (ok, out) = run_hook(
            r#"test "$MGE_TOOL" = bash && echo SEEN"#,
            &[("MGE_TOOL", "bash")],
            Duration::from_secs(10),
        )
        .await;
        assert!(ok);
        assert!(out.contains("SEEN"), "env not passed: {out:?}");

        let (ok2, _) = run_hook("exit 7", &[], Duration::from_secs(10)).await;
        assert!(!ok2, "non-zero exit must report failure");
    }

    #[tokio::test]
    async fn pre_tool_use_blocks_on_nonzero() {
        let hr = HookRunner {
            entries: vec![HookEntry {
                event: "PreToolUse".into(),
                matcher: "bash".into(),
                command: "echo nope >&2; exit 1".into(),
            }],
            timeout: Duration::from_secs(10),
        };
        let blocked = hr.pre_tool_use("bash", "{}").await;
        assert!(blocked.is_err());
        assert!(blocked.unwrap_err().contains("nope"));
        // non-matching tool is not blocked
        assert!(hr.pre_tool_use("read_file", "{}").await.is_ok());
    }
}

//! Subagents — the `spawn_agent` tool.
//!
//! The main agent calls this to delegate a scoped task to a *fresh* agent that
//! has its own context window and returns only its final answer (so the parent's
//! window stays lean — the key context-isolation primitive every top agent has).
//!
//! - **Roles** preset a system prompt + a model route, so each kind of subagent
//!   uses a model suited to its job (reviewers/security/planners → `heavy`,
//!   coders → `agent`, researchers → `main`).
//! - **Swarm**: pass `tasks: [...]` to run several subagents of the same role in
//!   parallel and get all their answers back.
//! - **Recursion is bounded structurally**: the child registry handed in here
//!   does NOT contain `spawn_agent`, so a subagent cannot spawn its own
//!   subagents (effective depth = 1), the same cap Codex uses by default.

use crate::agent::{Agent, AgentEvent};
use crate::config::Config;
use crate::llm::ToolDef;
use crate::routing::candidates_for;
use crate::tools::{Registry, Tool};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use futures_util::future::join_all;
use serde_json::{Value, json};

/// Max concurrent agents in one swarm call — protects free-tier rate limits and
/// the local GPU from a thundering herd.
const MAX_SWARM: usize = 6;

/// Tool that delegates work to fresh subagents (single or parallel swarm).
pub struct SpawnAgentTool {
    cfg: Config,
    /// Tool set handed to each child. Built WITHOUT `spawn_agent` → depth cap 1.
    child_tools: Registry,
}

impl SpawnAgentTool {
    pub fn new(cfg: Config, child_tools: Registry) -> Self {
        Self { cfg, child_tools }
    }

    /// (model route, system prompt) for a role. Unknown roles → general coder.
    fn role_spec(role: &str) -> (&'static str, &'static str) {
        match role {
            "reviewer" => (
                "heavy",
                "You are a meticulous senior code reviewer. Examine the code for correctness \
                 bugs, edge cases, error handling, and maintainability. Use your read/search \
                 tools to ground every claim in the actual code. Report findings by severity \
                 (CRITICAL/HIGH/MEDIUM/LOW) with file:line and a concrete fix. Be specific; do \
                 not invent issues. End with a one-line verdict.",
            ),
            "security" => (
                "heavy",
                "You are a security reviewer. Hunt for injection, secrets in code, path \
                 traversal, SSRF, unsafe deserialization, auth bypass, and unsafe shell/eval. \
                 Read the actual code with your tools before claiming a vulnerability. Report by \
                 severity with file:line, the attack scenario, and the fix. No speculation.",
            ),
            "researcher" | "explorer" => (
                "main",
                "You are a code researcher. Explore the codebase with your read/search tools and \
                 answer the question precisely, citing file paths and symbols. Do not modify \
                 files. Return a focused, factual summary — not a tour.",
            ),
            "planner" => (
                "heavy",
                "You are an implementation planner. Investigate the codebase, then produce a \
                 concrete, ordered, step-by-step plan: files to touch, functions to add/change, \
                 risks, and a test approach. Do not write the implementation — return the plan.",
            ),
            // coder / general
            _ => (
                "agent",
                "You are a capable coding subagent with full file and shell tools. Complete the \
                 delegated task end to end, verifying your work (build/tests where applicable). \
                 When done, reply with a concise summary of exactly what you changed and why.",
            ),
        }
    }

    /// Run one subagent to completion and return its final answer.
    async fn run_one(&self, role: &str, task: &str) -> String {
        let (route, sys) = Self::role_spec(role);
        // If the requested route is unavailable we fall back to the default — but
        // SAY SO, so a 'security'/'reviewer' answer that quietly ran on a weaker
        // model isn't mistaken for an authoritative one.
        let (candidates, downgraded) = match candidates_for(&self.cfg, route) {
            Ok(c) => (c, false),
            Err(_) => match candidates_for(&self.cfg, &self.cfg.default_route) {
                Ok(c) => (c, true),
                Err(e) => return format!("subagent error: {e:#}"),
            },
        };
        let mut agent = Agent::new(candidates, self.child_tools.clone(), sys);
        if let Err(e) = agent.run_turn(task, |_ev: AgentEvent| {}).await {
            return format!("subagent error: {e:#}");
        }
        let out = agent.last_text();
        let out = if out.trim().is_empty() {
            "(subagent produced no text answer)".to_string()
        } else {
            out
        };
        if downgraded {
            format!(
                "[note: '{route}' route not configured — ran on the default route instead; weigh accordingly]\n{out}"
            )
        } else {
            out
        }
    }
}

#[async_trait]
impl Tool for SpawnAgentTool {
    fn name(&self) -> &str {
        "spawn_agent"
    }

    fn def(&self) -> ToolDef {
        ToolDef {
            name: "spawn_agent".into(),
            description: "Delegate a scoped task to one or more fresh subagents, each with its \
                          own context window; returns their final answers. Use it to keep your \
                          own context lean, to get an independent review, or to parallelize. \
                          Roles: 'coder' (implements, full tools), 'researcher' (read-only \
                          exploration), 'reviewer' (code review), 'security' (security audit), \
                          'planner' (produces a plan). Give `task` for one subagent, or `tasks` \
                          (array) to run a parallel swarm of the same role."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": ["coder", "researcher", "reviewer", "security", "planner", "general"],
                        "description": "The kind of subagent (selects its model and system prompt)."
                    },
                    "task": { "type": "string", "description": "Task for a single subagent." },
                    "tasks": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Multiple tasks to run in parallel (same role). Max 6."
                    }
                },
                "required": ["role"]
            }),
        }
    }

    async fn run(&self, args: Value) -> Result<String> {
        let role = args
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("general");

        let mut tasks: Vec<String> = Vec::new();
        if let Some(arr) = args.get("tasks").and_then(Value::as_array) {
            tasks.extend(arr.iter().filter_map(Value::as_str).map(str::to_string));
        }
        if let Some(t) = args.get("task").and_then(Value::as_str) {
            tasks.push(t.to_string());
        }
        if tasks.is_empty() {
            return Err(anyhow!(
                "spawn_agent: provide `task` or a non-empty `tasks` array"
            ));
        }

        let dropped = tasks.len().saturating_sub(MAX_SWARM);
        if dropped > 0 {
            tasks.truncate(MAX_SWARM);
        }

        if tasks.len() == 1 {
            return Ok(self.run_one(role, &tasks[0]).await);
        }

        // Swarm: run all subagents concurrently, label each answer.
        let futs = tasks.iter().enumerate().map(|(i, t)| async move {
            let out = self.run_one(role, t).await;
            format!("── subagent #{} ({role}) ──\n{out}", i + 1)
        });
        let mut joined = join_all(futs).await.join("\n\n");
        if dropped > 0 {
            joined.push_str(&format!(
                "\n\n(note: {dropped} extra task(s) dropped — swarm cap is {MAX_SWARM})"
            ));
        }
        Ok(joined)
    }
}

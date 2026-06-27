# MGE_GOAT — Capability Gap Matrix (vs Claude Code, Codex, and the field)

> ⚠️ **Historical planning snapshot (2026-06).** This was the pre-build research
> matrix. Many items marked ❌/🎯 here are now **implemented** — including
> web_search, subagents/swarm, plan mode, reasoning effort, auto-compaction,
> session resume/fork, repo map, and goal loops. For the current, accurate feature
> set see the [README](../README.md). Kept for design context, not as a status board.

Researched 2026-06 against live docs for **Claude Code**, **OpenAI Codex CLI**, and 12+ others
(Aider, Cline, Roo, Cursor, Windsurf/Cascade, Gemini CLI, Qwen Code, Goose, Amp, opencode, Crush, OpenHands, Devin).

Status legend: ✅ have · ⚠️ partial · ❌ missing · 🎯 priority to build

---

## A. Core agent loop & tools
| Capability | Field | MGE_GOAT |
|---|---|---|
| read / write / edit / list / glob / grep / tree / bash | all | ✅ 9 built-in tools |
| web_fetch (URL → context) | all | ✅ |
| **web_search** (first-party search/grounding) | Codex, Gemini, Cursor, Cline | ❌ 🎯 |
| Agentic tool loop with round cap | all | ✅ 25 rounds |
| Streaming tool calls | all | ✅ SSE |
| **Parallel tool execution** (one turn, many calls) | Claude, Codex | ❌ 🎯 |
| Tool-result truncation w/ markers | all | ✅ 30KB cap |
| **Structured outputs** (JSON-schema-constrained) | Codex `--output-schema`, Claude `agent({schema})` | ❌ |
| **apply_patch / multi-strategy diff** (search-replace, unified, fuzzy) | Codex, Aider | ✅ `edit_file` 3-tier cascade: exact → trailing-ws → full-trim/indent |

## B. Multi-agent orchestration
| Capability | Field | MGE_GOAT |
|---|---|---|
| **Subagent spawning** (isolated context, returns summary) | Claude, Codex, opencode, Goose, Gemini, Amp, OpenHands | ❌ 🎯🎯 |
| **Parallel agents / swarm** | Claude (≤1000), Cursor (8), Cline Kanban, Goose (≤10) | ❌ 🎯🎯 |
| **Planner+executor / architect+editor split** | Aider, Cline Plan/Act, Roo | ❌ 🎯 |
| **Reviewer / critic agents** (code, security) | Claude `/simplify`+`/security-review`, Amp Oracle, Cursor Bugbot | ❌ 🎯🎯 |
| Named specialist agents (search, image, oracle) | Amp | ❌ |
| Orchestrator decomposition (Boomerang) | Roo, MultiDevin | ❌ |
| Nesting depth cap | Claude (5), Codex (`max_depth=1`) | ❌ (will cap at 1) |

## C. Reasoning control
| Capability | Field | MGE_GOAT |
|---|---|---|
| **Reasoning effort levels** (low/med/high/xhigh) | Claude `/effort`, Codex `model_reasoning_effort` | ❌ 🎯🎯 |
| **Plan mode** (read-only plan → approve → execute) | Claude, Codex, Gemini, Cline, Cursor, Devin | ❌ 🎯 |
| Extended thinking budget cap | Claude `MAX_THINKING_TOKENS` | ❌ |
| Reflection / self-critique on failure | OpenHands, Devin | ❌ |

## D. Context & memory management
| Capability | Field | MGE_GOAT |
|---|---|---|
| **Auto-compaction / summarization near limit** | Claude `/compact`, Codex auto-compact, Roo, opencode, Amp | ❌ 🎯🎯 |
| Microcompaction (cache-preserving) | Claude | ❌ |
| Context-window budgeting w/ real tokenizer | all | ⚠️ char/4 estimate only 🎯 |
| **Memory / rules file** (AGENTS.md / CLAUDE.md, hierarchical) | all (converging on AGENTS.md) | ❌ 🎯 |
| Auto-learned persistent memory | Cursor, Windsurf, Goose, Codex `[memories]` | ❌ |
| **Conversation persistence & resume** (`--resume`/`--continue`) | all | ❌ 🎯 |
| Fork / branch session | Claude `/branch`+`/fork`, Codex `fork`, opencode | ❌ |
| **Prompt-cache-aware prefix stability** | Claude, Qwen (80% savings) | ⚠️ not engineered 🎯 |
| Repo map (tree-sitter + PageRank) | Aider, Plandex | ❌ 🎯 |
| Embeddings/vector codebase index | Roo (Qdrant), Cursor | ❌ (deliberate: Continue *deprecated* this for agent mode) |

## E. Control loops & self-healing
| Capability | Field | MGE_GOAT |
|---|---|---|
| **Goal loop** (run until done) | Claude `/goal`, Codex `/goal`, Qwen `/loop` | ❌ 🎯🎯 |
| Done-detection (machine-checkable) | best practice | ❌ 🎯 |
| No-progress / cycle detection | best practice | ❌ |
| **Test / lint feedback loop** (run → feed errors back → fix) | Aider, Cline, Cursor, Windsurf | ❌ 🎯 |
| Auto-fix lint the agent introduced | Cursor, Windsurf | ❌ |
| Retry / model fallback on failure | Gemini, all | ✅ mid-stream fallback chain |

## F. Safety, permissions & sandboxing
| Capability | Field | MGE_GOAT |
|---|---|---|
| Secret scrubbing (env, transcripts) | all | ✅ bash env-scrub |
| SSRF / network guard | — | ✅ web_fetch host block |
| OS sandbox (seccomp/Landlock/Seatbelt/bwrap) | Codex, Gemini, OpenHands | ⚠️ NO_NEW_PRIVS + optional bwrap (MCP only) 🎯 |
| **Tiered approval modes** (suggest/auto-edit/full-auto/YOLO) | Codex, Claude, Cline, Roo | ⚠️ bash approve-all toggle only 🎯 |
| **Fine-grained per-tool/per-pattern permissions** | Codex execpolicy, Claude allow/ask/deny, Continue | ❌ 🎯 |
| Network-off-by-default in sandbox | Codex, Gemini | ❌ |
| MCP rug-pull / schema-drift defense | (MGE original) | ✅ SHA-256 fingerprint |
| Dry-run / diff preview before apply | Codex, plan mode | ❌ |

## G. Editing accuracy
| Capability | Field | MGE_GOAT |
|---|---|---|
| Fuzzy patch application (exact→ws→anchored) | Aider, Codex | ❌ 🎯 |
| **LSP integration** (diagnostics, rename, refs) | Crush, opencode, Qwen | ❌ (differentiator — most CLIs lack it) |
| Tree-sitter syntax-aware parse/symbols | Aider, Roo | ❌ |
| Syntax validation post-edit | best practice | ❌ |
| Auto-format after edit (hook) | all | ❌ (have hook infra concept) |

## H. Git, checkpoints & integration
| Capability | Field | MGE_GOAT |
|---|---|---|
| **Checkpoint / rewind / undo** (workspace snapshots) | Claude `/rewind`, Cline (per-tool-call), Gemini, Windsurf | ❌ 🎯 |
| Auto-commit each AI edit + generated message | Aider, opencode | ❌ |
| Git worktree isolation for parallel agents | Claude, Cursor, Cline, Gemini | ❌ |
| PR creation / issue→PR resolver | Codex, OpenHands, Cursor, Devin | ❌ |
| AI PR reviewer | Cursor Bugbot, Amp review, Codex | ❌ |
| **Headless / scriptable mode** (`-p`, JSON out, CI) | Claude `-p`, Codex `exec --json`, Continue, Gemini | ❌ 🎯 |
| GitHub Actions / Slack triggers | OpenHands, Codex, Claude | ❌ |

## I. Input / output / UX
| Capability | Field | MGE_GOAT |
|---|---|---|
| **Multimodal input** (image / PDF / screenshot) | Codex `-i`, Gemini, Goose, Cursor, Aider | ❌ 🎯 |
| Slash commands (built-in + custom) | all | ⚠️ 5 built-in, no custom 🎯 |
| @-mentions (file/symbol context) | Codex, Cursor, Cline, Continue | ❌ 🎯 |
| Output styles (terse/explanatory/json) | Claude | ❌ |
| **Cost / token tracking** (per-turn + cumulative) | all | ⚠️ tool-usage telemetry only 🎯 |
| Statusline (model/cwd/branch/tokens) | all | ⚠️ context meter in TUI |
| Voice input | Aider, Qwen, Cursor, Windsurf | ❌ |
| TODO / focus-chain task tracking | Codex `update_plan`, Cline, Cursor | ❌ 🎯 |
| Animated themed TUI | (MGE original — goat/ice-cream) | ✅ |

## J. Extensibility
| Capability | Field | MGE_GOAT |
|---|---|---|
| MCP client (stdio + HTTP) | all | ✅ |
| MCP resources & prompts (beyond tools) | Codex, Claude | ❌ |
| MCP marketplace / one-click install | Cline, Roo, Gemini, Goose | ✅ `mge market` |
| Skills (SKILL.md, hot-reload) | Claude, Codex, Crush, Goose, Cline | ✅ (no hot-reload) |
| Plugins (bundle skills+MCP) | Claude, Codex, opencode, Amp | ✅ |
| Hooks (pre/post tool lifecycle) | Claude (many events), Codex (10), Cline | ❌ 🎯 |
| Custom slash commands / modes | Roo, opencode, Continue | ❌ |

## K. Models & routing (MGE's strength)
| Capability | Field | MGE_GOAT |
|---|---|---|
| Model-agnostic, OpenAI-compatible | all open tools | ✅ |
| Local models (llama.cpp / Ollama) | all open tools | ✅ GPU-aware gate |
| **Per-task / per-role routing** (fast/main/heavy) | Aider, Goose, Amp, Cline | ✅ **tested, accurate** |
| Mid-session model switch (preserved context) | Crush, Claude `/model` | ✅ `/model` |
| Free-first (free APIs + local) | (MGE differentiator) | ✅ |
| Automatic fallback chain | Gemini, Claude `fallbackModel` | ✅ |

---

## Highest-leverage build order (recommended)

**Tier 1 — foundational, originally requested, unlocks the rest — ✅ SHIPPED & TESTED**
1. ✅ Subagents + parallel swarm + reviewer/critic agents (B) — `spawn_agent`, 5 roles, depth-1 cap. *Tested: researcher subagent located routing.rs.*
2. ✅ Reasoning effort levels (C) — `/effort` + `reasoning_effort` wire passthrough.
3. ✅ Auto-compaction + token budgeting (D) — 48k trigger, summarize-older at safe User-boundary split.
4. ✅ Goal loop with machine-checkable done-detection (E) — `mge goal … --until <cmd>`. *Tested: built a file, `--until` grep check passed in 1 iteration.*
5. ✅ (bonus) AGENTS.md / CLAUDE.md project-memory loading (D) — all 3 entry points.
6. ✅ (already existed) `edit_file` fuzzy 3-tier cascade (G).

**Tier 2 — correctness & safety multipliers — ✅ SHIPPED & TESTED**
5. ✅ (already existed) Fuzzy/multi-strategy patch application (G).
6. ✅ Tiered approval modes + per-tool permissions (F) — `src/permissions.rs`: modes default/acceptEdits/plan/yolo, allow/ask/deny with `bash:` patterns, deny>plan>ask>allow>default, consulted in `dispatch`. `/mode` + Shift+Tab + `--permission-mode`/`--yolo`. *Tested: plan mode blocked a write end-to-end.*
7. ✅ Test/lint feedback loop (E) — `[checks].after_edit_cmd` runs after each edit & feeds failures back; `mge fix "<cmd>"` iterates until green (stuck-detection, no double-check). `util::run_check_captured` (timeout + secret-scrub + 8KB cap).
8. ✅ Checkpoint/rewind (H) — `src/checkpoint.rs`: per-session 0600 JSONL journal snapshots prior content before write/edit (bash not tracked); `mge rewind [seq]` + `/rewind`. *Tested: write recorded, restore deleted the new file.* (Snapshot-based, not git — works in non-git dirs.)

**Tier 3 — reach & parity. Batch A ✅ SHIPPED & TESTED:**
9. ✅ Headless `-p`/JSON mode (`mge run … [--json]`, clean stdout). *Tested.*
10. ✅ AGENTS.md memory (earlier) + ✅ session resume/fork (`--resume`/`--continue`/`--fork`, `mge sessions`; path-guard + validate_and_truncate). *Tested: recalled "42".*
14. ✅ @-mentions (`src/mentions.rs`, email-safe) *Tested: `@fruit.txt`* + ✅ custom slash commands (`src/commands.rs`, `/commands`, `mge commands`).
17. ✅ Cost/token tracking (`/cost`, statusline chip, headless JSON, CLI summary).

**Tier 3 — Batch B. Partially shipped:**
- ✅ **Repo map** (`src/repo_map.rs`, `mge map`) — dependency-free heuristic (regex+walkdir, no tree-sitter): per-language symbol extraction ranked by cross-reference density, injected into the system prompt. *Tested + dogfooded.* (Design weighed tree-sitter: ~6 grammar crates + ~90s build for ~15% more accuracy — not worth it for free-first; documented upgrade path.)
- ✅ **Plan-research mode** (`mge plan "…"`) — read-only research (`Mode::Plan`) → drafts a plan → you approve → executes (acceptEdits/`--yolo`). Also extracted the shared `build_agent`/`cli_print_event` helpers.
- ✅ **web_search** (`src/tools/web_search.rs`) — DDG HTML backend, no API key, free-first; registered in `with_defaults` (read-only, allowed in all modes). Percent-encode/decode, range-matched snippets, control-char sanitization. *Verified: live HTML-structure match + E2E ("Tokio – https://tokio.rs/").*
- ⏳ **Remaining (designed + triaged, blueprints saved):** hooks (PreToolUse/PostToolUse — needs wiring at all 6 registry sites + dispatch pre/post), LSP diagnostics (JSON-RPC + rust-analyzer spawn — heavy), multimodal image (**blocked**: all current free models are text-only → needs a vision route the user adds).

**Deliberate non-goals (for now):** embeddings vector DB (Continue deprecated it for agent mode; repo-map is cheaper/stronger), cloud/VM per-task runtime, voice. Revisit later.

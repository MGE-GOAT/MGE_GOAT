<div align="center">

<img src="docs/mascot.gif" alt="MGE_GOAT mascot — an evil goat licking a crying, melting ice cream" width="320">

<sub>the actual TUI mascot — an evil goat licking a crying ice cream that blinks, bobs &amp; melts</sub>

# 🐐🍦 MGE_GOAT

### the **G**reatest **O**f **A**ll **T**ools

**A free, open-source, GPU-aware agentic coding CLI in the spirit of Claude Code — built to run on open models.**

[![CI](https://github.com/MGE-GOAT/MGE_GOAT/actions/workflows/ci.yml/badge.svg)](https://github.com/MGE-GOAT/MGE_GOAT/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-edition_2024-orange.svg)](https://www.rust-lang.org/)
[![TUI: ratatui](https://img.shields.io/badge/TUI-ratatui-blueviolet.svg)](https://ratatui.rs/)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#-contributing)

*An evil goat that runs on open models and licks a crying ice cream while it thinks.*

</div>

---

MGE_GOAT is a terminal coding agent — a full agentic tool-use loop with a real
TUI — designed from the ground up to run on **open-source models**: locally on
your GPU via [llama.cpp](https://github.com/ggml-org/llama.cpp), and on **free**
remote APIs like OpenRouter and NVIDIA NIM. Bring your own free keys; it picks a
suitable model per task, **falls back automatically** when a provider is throttled,
and ends every cascade at your local model so it keeps working offline. It speaks
**MCP**, loads **skills** and **plugins**, delegates to your **Codex/Claude
subscriptions**, and is **hardened** against prompt injection and the usual
agent-tool footguns.

> **Status:** personal-use software, reviewed by a multi-agent adversarial pass
> (security · concurrency · correctness) with all confirmed findings fixed. Audited,
> not formally pen-tested — read the [Security](#-security) section before pointing
> it at untrusted repos or servers.

## Table of contents

- [Why MGE_GOAT?](#why-mge_goat)
- [Features](#-features)
- [Installation](#-installation)
- [Getting a free API key](#-getting-a-free-api-key)
- [Quick start](#-quick-start)
- [Usage](#-usage)
  - [Commands](#commands)
  - [The TUI](#the-tui)
  - [Permission modes](#permission-modes)
- [Configuration](#-configuration)
- [Local models (llama.cpp)](#-local-models-llamacpp)
- [MCP, skills, plugins & the marketplace](#-mcp-skills-plugins--the-marketplace)
- [Delegating to Codex / Claude subscriptions](#-delegating-to-codex--claude-subscriptions)
- [Architecture](#-architecture)
- [Security](#-security)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## Why MGE_GOAT?

Most great coding agents assume a paid frontier API. MGE_GOAT flips that:

- **Free-first.** It defaults to free, tool-capable models (OpenRouter `:free`
  tiers, NVIDIA NIM) and your own local GPU — no subscription required to start.
- **Never stuck.** Every model route is a **fallback chain**. Rate-limited? 5xx?
  Stale model id? Out of credit? It cascades to the next candidate, ending at your
  local llama.cpp server. Routes whose key is missing are simply skipped.
- **GPU-aware.** It reads your VRAM and only prefers a local model when it actually
  fits, so a small GPU degrades gracefully to remote instead of OOM-ing.
- **Yours to read.** ~12k lines of Rust, no telemetry phone-home, MIT licensed.

It is not a wrapper around one API — it is a complete agent: tool loop, streaming
TUI, routing, RAG, subagents, MCP, skills, sessions, checkpoints, and hardening.

## ✨ Features

**Agent core**
- **Agentic loop** with a 25-round-per-turn cap and streaming SSE tool calls.
- **Built-in tools:** `read_file` · `write_file` · `edit_file` (3-tier fuzzy diff)
  · `ls` · `glob` · `grep` · `tree` · `find_symbol` · `code_graph` ·
  `semantic_search` · `bash` · `web_fetch` · `web_search` (DuckDuckGo, no key) ·
  `delegate` · `spawn_agent` · `lsp_diagnostics` · `use_skill`.
- **Subagents & swarms** — `spawn_agent` delegates to fresh agents (coder /
  researcher / reviewer / security / planner), single or parallel, each with its
  own context window and per-role model.
- **Multimodal** — drop `@screenshot.png` or `@voice.mp3` into a message; the turn
  auto-routes to a configured vision/audio model.

**Working in real codebases**
- **Repo map** — a dependency-free symbol map ranked by cross-reference density is
  injected per turn so the agent orients fast (`mge map` to preview).
- **Code graph & semantic search** — `code_graph` (definitions + references) and
  optional embedding-backed `semantic_search` on top of default lexical BM25.
- **LSP diagnostics** — `lsp_diagnostics` runs your real language server (e.g.
  `rust-analyzer`) from a warm, persistent session so the agent sees ground-truth
  compiler errors, not guesses.
- **Auto-compaction** — long sessions summarize older turns at a safe boundary,
  with **lossless BM25 archive recovery** (exact prior tool outputs are
  retrievable, not lost to a digest).

**Control & autonomy**
- **Plan mode** — `mge plan "…"` researches read-only, drafts a plan, waits for
  approval, then executes.
- **Goal loops** — `mge goal "…" --until <check>` runs until a machine-checkable
  condition passes.
- **Self-healing** — `mge fix "<cmd>"` iterates until a command goes green; optional
  `[checks]` run after every edit and feed failures back.
- **Reasoning effort** — `/effort low|medium|high|xhigh` for models that honor it.
- **Checkpoint / rewind** — every file edit is snapshotted; `mge rewind` / `/rewind`
  restores (works without git).

**Models & routing**
- **Any model, any provider** — OpenRouter · NVIDIA NIM · OpenAI · Anthropic
  (via its OpenAI-compatible endpoint) · GitHub Models · Hugging Face · local
  llama.cpp. `/model <id>` switches to *any* model mid-chat; `mge models` browses
  the live catalog.
- **Auto-cascade routing** — fallback on rate-limit / 5xx / unavailable-model /
  no-credit, ending at local.
- **Per-task auto-routing** — a free heuristic picks `fast` / `main` / `heavy` per
  prompt; **GPU-aware** local-vs-remote gating.

**Ecosystem**
- **MCP client** — connect any MCP server (stdio or HTTP); tools appear as
  `mcp__server__tool`, with rug-pull fingerprinting and optional sandboxing.
- **Skills / plugins / marketplace** — `SKILL.md` progressive disclosure, plugin
  bundles, and `mge market` against the official MCP registry.
- **Custom slash commands** — `~/.config/mge/commands/*.md` macros with
  `$ARGUMENTS` / `$1..$9`.
- **AGENTS.md / CLAUDE.md** — project + user instruction files loaded into context.
- **Delegate to subscriptions** — hand subtasks to **Codex** (ChatGPT) or
  **Claude Code** via their official CLIs (flat-rate, not per-token).

**Quality of life**
- **Animated TUI** — the goat idles and licks the crying ice cream while thinking;
  live diffs, an activity/plan pane, and an in-TUI approval prompt.
- **Session resume** — `--resume` / `--continue` / `--fork`, `mge sessions`.
- **Headless / CI** — `mge run "…" --json` prints only the answer to stdout.
- **Cost/token tracking** — `/cost`, the statusline, and headless JSON.

## 📦 Installation

### Prerequisites

| Requirement | Notes |
|---|---|
| **Rust** (stable, edition 2024 → toolchain **1.85+**) | Install via [rustup](https://rustup.rs/). |
| **git** | To clone. |
| *(optional)* **llama.cpp** `llama-server` | For local/offline models — see [Local models](#-local-models-llamacpp). |
| *(optional)* **NVIDIA GPU + drivers** | Enables VRAM-aware local routing. Degrades gracefully if absent. |
| *(optional)* **bubblewrap** (`bwrap`) | For tier-2 MCP sandboxing on Linux. |

### Build from source

```bash
git clone https://github.com/MGE-GOAT/MGE_GOAT.git mge && cd mge
cargo build --release
# binary is now ./target/release/mge
```

Optionally put it on your `PATH`:

```bash
cargo install --path .      # installs `mge` into ~/.cargo/bin
```

## 🔑 Getting a free API key

You need **at least one** key (or a local llama.cpp server). All are free to start:

| Provider | Where | Free tier |
|---|---|---|
| **OpenRouter** | <https://openrouter.ai/keys> | Many `:free` models (great default). |
| **NVIDIA NIM** | <https://build.nvidia.com> | Generous free dev tier, no daily wall. |
| **GitHub Models** | a GitHub PAT | Free chat + embeddings via `models.github.ai`. |
| **OpenAI / Anthropic** | their dashboards | Optional premium routes (`/model openai`, `/model claude`). |

`mge setup` stores keys in `~/.config/mge/secrets.env` (`chmod 600`, never
committed) and writes sensible task-tier routes.

## 🚀 Quick start

```bash
./target/release/mge setup     # paste your free key(s); detects GPU; writes config + routes
./target/release/mge tui       # launch the animated TUI
```

…or jump straight in headless:

```bash
mge run "explain what src/agent/mod.rs does"          # one-shot, clean stdout
mge chat                                               # line-mode REPL
mge goal "make cargo test pass" --until "cargo test"   # autonomous until green
```

> 💡 The TUI mascot likes a terminal **~30 rows tall**. `mge chat` is a lighter
> line-mode REPL with the same agent if your terminal is small.

## 🧭 Usage

### Commands

| Command | What it does |
|---|---|
| `mge setup` | Guided first-run: keys → GPU detect → task-tier routes. |
| `mge init` | Write a starter `config.toml` (no prompts). |
| `mge tui` | Full-screen animated TUI. |
| `mge chat [--resume\|--continue\|--fork] [--yolo] [--route <r>]` | Line-mode agentic REPL (resumable). |
| `mge run "<prompt>" [--json] [--image <f>]` | Headless one-shot for pipes/CI (clean stdout). |
| `mge plan "<task>"` | Research read-only → draft plan → approve → execute. |
| `mge goal "…" [--until <cmd>] [--max N]` | Autonomous goal loop until done. |
| `mge fix "<cmd>" [--max N]` | Iterate until a shell command passes. |
| `mge rewind [seq] [--force]` | List / restore file-edit checkpoints. |
| `mge map` | Print the repo map (codebase orientation). |
| `mge models [query]` | Browse the live model catalog. |
| `mge sessions` | List saved sessions (resume with `--resume <id>`). |
| `mge doctor` | Show resolved config, routes, key **presence** (never values). |
| `mge gpu` | Local GPU / VRAM status used for routing. |
| `mge mcp [--reapprove <server>]` | Connect to MCP servers and list their tools. |
| `mge skills` / `mge commands` | List discovered skills / custom slash commands. |
| `mge market search\|info\|install <q>` | Find & add MCP servers from the registry. |
| `mge stats` / `mge prune` | Tool-usage stats / never-used MCP tools. |
| `mge banner` / `mge splash` | Print / animate the goat. |

### The TUI

| Key | Action |
|---|---|
| `Enter` | Send message |
| `Esc` / `Ctrl-C` | Quit (Esc twice mid-turn) |
| `↑` / `↓` / `PgUp` / `PgDn` | Scroll the conversation |
| `Ctrl-P` / `Ctrl-N` | Recall previous / next input |
| `Ctrl-U` | Clear the input line |
| **`Shift+Tab`** | Cycle permission mode (default → accept-edits → plan → yolo) |
| `@path` | Inject a file's contents into your message |
| `y` / `n` | Answer a `bash`/`delegate` approval prompt |

**Slash commands:** `/help` · `/clear` · `/context` · `/cost` · `/model <id>` ·
`/auto` · `/effort <level>` · `/mode <mode>` · `/rewind [seq]` · `/commands` · `/quit`.

### Permission modes

Cycle with **Shift+Tab** or set `[permissions].mode`. **`deny` rules always win**,
even in `yolo`.

| Mode | Behavior |
|---|---|
| `default` | Ask before `bash`/`write`/`edit`; allow reads. |
| `acceptEdits` | Auto-apply edits; **`bash`/`delegate` prompt for `y/n`** in the TUI. |
| `plan` | **Read-only.** Blocks *all* `bash` and writes (even `bash ls`) — for audits. |
| `yolo` | Allow everything (`deny` rules still apply). |

Fine-grained rules live under `[permissions]`: `allow` / `ask` / `deny`, with
`bash:<pattern>` matching for shell commands (e.g. `"bash:rm -rf *"`).

## ⚙️ Configuration

Config: `~/.config/mge/config.toml` · Secrets: `~/.config/mge/secrets.env`
(env-var values, never stored in the TOML). `mge init` writes a fully-commented
starter. The essentials:

```toml
default_route = "main"
auto_route = true            # pick fast/main/heavy per task automatically

# ── Providers (all OpenAI-compatible; keys come from secrets.env) ──
[providers.openrouter]
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"

[providers.local]
base_url = "http://localhost:8080/v1"   # llama.cpp llama-server
api_key_env = "NONE"
local = true
# text_tool_calls = true                # set true only for models lacking native
                                         # tool-calling (see Security)

# ── Model routes — free-first AUTO-CASCADE ──
# Each route falls back (rate-limit / 5xx / timeout / unavailable / no-credit)
# down its chain, ending at `local`. Missing-key routes are skipped automatically.
[models.main]
provider = "nim"
model = "qwen/qwen3.5-122b-a10b"
fallback = ["main_free", "local"]

[models.main_free]
provider = "openrouter"
model = "qwen/qwen3-coder:free"
fallback = ["local"]

[models.local]               # ultimate fallback — run llama-server
provider = "local"
model = "local"
min_free_vram_mb = 3000      # GPU gate: only prefer local if this much VRAM is free
```

<details>
<summary><b>Optional sections</b> — permissions, checks, hooks, LSP, embeddings, MCP</summary>

```toml
# Permissions — deny ALWAYS wins, even in yolo.
[permissions]
mode = "default"             # default | acceptEdits | plan | yolo
deny  = ["bash:rm -rf *", "bash:curl *| bash"]
allow = ["read_file", "grep", "glob"]

# Checks — runs after every successful write/edit; output fed back to the model.
# Use PROJECT-LOCAL binaries only (output is injected into context).
[checks]
enabled = true
after_edit_cmd = "cargo check --message-format short"
timeout_secs = 60

# Hooks — shell commands fired around tool calls (PreToolUse / PostToolUse).
[[hooks.hooks]]
event = "PostToolUse"
matcher = "write_file|edit_file"
command = "cargo fmt"

# LSP — language servers the lsp_diagnostics tool may spawn, keyed by extension.
[lsp]
enabled = true
timeout_secs = 60
[lsp.servers]
rs = ["rust-analyzer"]

# RAG — optional SEMANTIC retrieval (lexical BM25 is the default, no setup needed).
[rag]
enabled = true
endpoint = "https://models.github.ai/inference"
model = "openai/text-embedding-3-small"
api_key_env = "GITHUB_TOKEN"

# MCP — connect any server; tools become mcp__<name>__<tool>.
[mcp]
enabled = true
[mcp.servers.filesystem]
transport = "stdio"          # or "http" with a `url`
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/path"]
sandbox = "process"          # off | process (NO_NEW_PRIVS) | bwrap
tools_allow = ["read_", "list_"]   # optional: only expose these tool prefixes
```

</details>

- **Multimodal routes** (`vision`, `audio`) are pre-wired; `@image.png` / `@audio.mp3`
  auto-route there.
- **Skills** live in `~/.config/mge/skills/<name>/SKILL.md` (trusted). Project skills
  (`./.mge/skills`) are opt-in via `[skills].trust_project_skills = true`.
- **Plugins** live in `~/.config/mge/plugins/<name>/` and may bundle `skills/` + an
  `mcp.json`.

Run `mge doctor` any time to see exactly what resolved (providers, routes, which
keys are present).

## 🖥️ Local models (llama.cpp)

Local models are the ultimate fallback and the offline path. In short:

```bash
# build llama.cpp, then serve an OpenAI-compatible endpoint on :8080
llama-server -m qwen2.5-coder-7b-instruct-q4_k_m.gguf -c 8192 --port 8080
```

The default config already has a `[providers.local]` + `[models.local]` route
pointing at `http://localhost:8080/v1`. `min_free_vram_mb` gates whether local is
*preferred* (so a small GPU won't be chosen for a model that won't fit). Full
build/serve instructions, model picks for various VRAM budgets, and the
`text_tool_calls` note are in **[`docs/LOCAL_LLAMA.md`](docs/LOCAL_LLAMA.md)**.

## 🔌 MCP, skills, plugins & the marketplace

- **MCP** — add any [Model Context Protocol](https://modelcontextprotocol.io/)
  server (stdio or HTTP) under `[mcp.servers]`. Its tools appear to the agent as
  `mcp__<server>__<tool>`. Schemas are SHA-256 fingerprinted and **blocked on
  drift** until you re-approve (`mge mcp --reapprove <server>`).
- **Marketplace** — `mge market search <q>` / `info` / `install` against the
  official MCP registry.
- **Skills** — drop a `SKILL.md` (progressive-disclosure instructions) in
  `~/.config/mge/skills/<name>/`; the agent loads it via `use_skill`.
- **Plugins** — bundle skills + an `mcp.json` under `~/.config/mge/plugins/<name>/`.

## 🤝 Delegating to Codex / Claude subscriptions

The `delegate` tool hands a subtask to **Codex** (ChatGPT) or **Claude Code**
through their **official CLIs**, so you spend your flat-rate subscription instead of
per-token API. It authenticates with *their* own credentials (`~/.codex`,
`~/.claude`) — MGE's provider keys are scrubbed from the child environment, and
there is **no token extraction** (it stays within each provider's terms).

## 🏗️ Architecture

```
src/
  main.rs        CLI (clap subcommands)
  config.rs      TOML config + env-var secrets + starter template
  llm/           provider trait + OpenAI-compatible SSE streaming + tool-call parsing
  routing.rs     fallback chains + per-task classifier + GPU gate
  agent/         the agentic tool-use loop (+ spawn.rs for subagents)
  tools/         built-in tools (read/write/edit/bash/glob/grep/tree/…)
                   delegate.rs  → Codex/Claude CLIs
                   lsp.rs       → persistent-session LSP diagnostics
                   web_search.rs→ DuckDuckGo
  mcp.rs         MCP client (stdio + HTTP), rug-pull fingerprinting, sandbox
  permissions.rs tiered modes + allow/ask/deny rule engine
  repo_map.rs    cross-reference-ranked symbol map (per-turn RAG)
  graph.rs       code knowledge graph (defs + references)
  embed.rs       optional embeddings + cosine semantic search
  session.rs     transcript persistence, resume, lossless archive
  checkpoint.rs  per-edit snapshots for rewind
  skills.rs      SKILL.md loader + use_skill tool
  plugins.rs     plugin loader (fans out to skills + mcp)
  market.rs      MCP registry search / install
  hooks.rs       PreToolUse / PostToolUse shell hooks
  gpu.rs         NVML VRAM probe for routing
  tui/           ratatui frontend
  sprite.rs      half-block pixel-art mascot + animation
  theme.rs       palette + the goat/ice-cream scene
```

Every tool — built-in, MCP, skill, or plugin — implements the same `Tool` trait,
so the agent loop is identical no matter where a tool comes from.

## 🔒 Security

The agent runs shell commands and connects to third-party MCP servers, so the
trust boundary is taken seriously:

- **Prompt-injection-resistant tool calls.** Models that lack native structured
  tool-calling (local Qwen/Hermes) have their calls parsed from text; models that
  *do* have it (cloud APIs) do **not** — so a cloud model quoting a malicious file
  containing `<function=bash>…` markup can't be tricked into executing it. Controlled
  per provider via `text_tool_calls`.
- **Approval prompts.** In the TUI, `bash` and `delegate` ask for `y/n` before
  running (outside `yolo`); `deny`/`plan` rules are always enforced.
- **Secrets are stripped** from every child process — `bash`, LSP servers, hooks,
  `delegate`, **and** MCP stdio servers — and config/secrets/checkpoints/embed
  caches are written **atomically at `0600`**.
- **SSRF guard with DNS-rebinding protection** on `web_fetch`, the embeddings
  endpoint, and MCP HTTP: loopback/private/link-local hosts are blocked, *and* the
  resolved IP is checked, so a public name pointing at `169.254.169.254` is refused.
- **MCP hardening** — tool schemas are SHA-256 fingerprinted and **blocked on
  drift** (rug-pull defense) until you re-approve; spawned servers get
  `NO_NEW_PRIVS` and optional `bwrap` namespace isolation.
- **Sensitive-path denylist** — `@mention` refuses to read `~/.ssh`, `~/.aws`,
  `~/.config/mge`, `/proc`, etc., so secrets can't be slurped into a prompt.
- Registry installs are TOML-injection-safe; project-repo skills/`CLAUDE.md` are
  untrusted by default.

It is personal-use software — audited by a multi-agent review, but not formally
pen-tested. Review the code before pointing it at untrusted repositories or servers.

## 🧯 Troubleshooting

| Symptom | Fix |
|---|---|
| `no usable candidates for route …` | The route's provider key is missing and so are its fallbacks. Add a key (`mge setup`) or point `default_route` at one you have. |
| Local route fails / times out | Start `llama-server` (see [Local models](#-local-models-llamacpp)), or rely on a remote route — local is only the *last* fallback. |
| A model id 404s | Model ids drift; the cascade self-heals to the next candidate. Run `mge models` for current ids and update the route. |
| No GPU detected | Fine — `mge gpu` will say so and routing just prefers remote. A GPU is optional. |
| TUI looks cramped / mascot clipped | Use a terminal **≥ ~30 rows**, or use `mge chat` (line mode). |
| Want to see what's configured | `mge doctor` (shows key presence, never values). |

## 👋 Contributing

PRs and issues welcome. Before submitting:

```bash
cargo build              # compiles clean
cargo clippy -- -D warnings   # zero warnings
cargo test               # all green
cargo fmt                # formatted
```

Keep functions small, handle errors explicitly, and add a test for non-trivial
logic. Security-sensitive changes (anything touching the tool trust boundary,
permissions, or child-process spawning) should call that out in the PR.

## 📜 License

[MIT](LICENSE). Use it, fork it, ship it. 🐐🍦

<div align="center">
<sub>Built with <a href="https://www.rust-lang.org/">Rust</a> ·
<a href="https://ratatui.rs/">ratatui</a> ·
<a href="https://modelcontextprotocol.io/">MCP</a> ·
<a href="https://github.com/ggml-org/llama.cpp">llama.cpp</a></sub>
</div>

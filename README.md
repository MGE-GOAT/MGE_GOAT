<div align="center">

# 🐐🍦 MGE_GOAT

### the **G**reatest **O**f **A**ll **T**ools — a free, open-source agentic coding CLI

*An evil goat that runs on open models and licks a crying ice cream while it thinks.*

`Rust` · `ratatui` · open models (local + free APIs) · MCP · skills · plugins

</div>

---

MGE_GOAT is a terminal coding agent in the spirit of Claude Code — full agentic
tool-use loop, a real TUI — but built to run on **open-source models** (locally on
your GPU via [llama.cpp](https://github.com/ggml-org/llama.cpp), and on **free**
remote APIs like OpenRouter and NVIDIA NIM). Bring your own free keys; it picks a
suitable model per task and falls back automatically. It speaks **MCP**, loads
**skills** and **plugins**, and has a **marketplace** to find and add more tools.

```
  ▄▄        ▄▄
 ▐  ▌      ▐  ▌        evil goat  ·  yellow slit eyes  ·  fanged grin
 ▐██▌◣    ◢▐██▌        gold ear piercing  ·  licking...
  ▝██████████▝            🍓 strawberry  +  🫐 blueberry  +  🍪 biscuit cone
   ▝▀▀▀▀▀▀▀▀                ...a small ice cream that cries, sweats & melts
```

## ✨ Features

- **Agentic loop** — read/write/edit/glob/grep/tree/bash/web_fetch/**web_search**, fed back round after round
- **Web search** — built-in `web_search` (DuckDuckGo, no API key) so the agent can find current docs/examples, then `web_fetch` to read them
- **Multimodal** — drop `@screenshot.png` or `@voice.mp3` into a chat message (or `mge run --image x.png`); the turn auto-routes to a configured vision/audio model (free ones exist on OpenRouter)
- **Subagents & swarms** — `spawn_agent` delegates to fresh agents (roles: coder/researcher/reviewer/security/planner), single or parallel, each with its own context window and per-role model
- **Repo map** — a dependency-free symbol map (ranked by cross-reference density) is injected into context so the agent orients in large codebases fast (`mge map` to preview)
- **Plan mode** — `mge plan "…"` researches read-only, drafts a plan, waits for your approval, then executes
- **Goal loops** — `mge goal "…" --until <check>` works autonomously until a machine-checkable condition passes
- **Self-healing** — `mge fix "<cmd>"` iterates until the command goes green; optional `[checks]` run after every edit and feed failures back
- **Reasoning effort** — `/effort low|medium|high|xhigh` (wired to models that honor it)
- **Auto-compaction** — long sessions summarize older turns automatically at a safe turn boundary
- **Checkpoint / rewind** — every file edit is snapshotted; `mge rewind` / `/rewind` restores (works without git)
- **Tiered permissions** — modes `default/acceptEdits/plan/yolo` + per-tool allow/ask/deny (`bash:` patterns); `plan` is read-only; deny always wins
- **@-mentions** — `@path` in a message injects that file's contents (email-safe, capped)
- **Custom slash commands** — `~/.config/mge/commands/*.md` macros with `$ARGUMENTS`/`$1..$9`
- **Session resume** — every chat/TUI session is saved; `--resume`/`--continue`/`--fork`, `mge sessions`
- **Headless / CI** — `mge run "…" --json` prints only the answer (or a JSON object) to stdout
- **Cost/token tracking** — per-session estimate via `/cost`, the statusline, and headless JSON
- **Animated pixel-art TUI** — the goat chuckles when idle, licks the crying ice cream while thinking; vibrant matching theme
- **Any model, any provider** — OpenRouter / NVIDIA NIM / OpenAI / Anthropic (Claude via its OpenAI-compat endpoint) / local llama.cpp; `mge models` browses the live catalog and `/model <id>` switches to **any** model mid-chat
- **Auto-cascade routing** — every route falls back on rate-limit / 5xx / unavailable-model / no-credit down a chain ending at local, so it keeps working when a provider is throttled or a key is unfunded
- **Use your subscriptions** — `delegate` hands subtasks to **Codex** (ChatGPT) or **Claude Code** (Claude) via their official CLIs, so you spend your flat-rate subscription instead of per-token API (no token extraction; stays within provider terms)
- **Open models, GPU-aware** — local llama.cpp **and** free remote APIs, with automatic fallback
- **Per-task auto-routing** — a free heuristic picks `fast` / `main` / `heavy` per prompt
- **MCP client** — connect to any MCP server (stdio or HTTP); tools appear as `mcp__server__tool`
- **Skills / plugins / marketplace** — `SKILL.md` progressive disclosure, plugin bundles, `mge market` against the official MCP registry
- **AGENTS.md / CLAUDE.md** — project + user instruction files loaded into context
- **Hardened** — prompt-injection-resistant tool calls (text-format parsing gated to models that need it, so a cloud model quoting a malicious file can't execute it), in-TUI approval prompt for `bash`/`delegate`, env-scrubbed child processes (bash, LSP, hooks, delegate **and** MCP servers), SSRF guard with DNS-rebinding protection, panic-safe text, NO_NEW_PRIVS + optional bwrap sandbox, MCP rug-pull fingerprinting, TOML-injection-safe installs, atomic 0600 secrets/checkpoints/caches

## 🚀 Quick start

```bash
git clone https://github.com/MGE-GOAT/MGE_GOAT.git mge && cd mge
cargo build --release
./target/release/mge setup     # paste your free API key(s), it writes the config
./target/release/mge tui       # launch the animated TUI
```

Get a free key from [OpenRouter](https://openrouter.ai/keys) and/or
[NVIDIA NIM](https://build.nvidia.com). `mge setup` stores them in
`~/.config/mge/secrets.env` (chmod 600 — never committed) and writes task-tier
routes with sensible free, tool-capable models.

> Tip: the TUI mascot likes a terminal ~30 rows tall. `mge chat` is a lighter
> line-mode REPL with the same agent.

## 🧰 Commands

| Command | What it does |
|---|---|
| `mge setup` | Guided first-run: keys + GPU detect + task-tier routes |
| `mge tui` | Full-screen animated TUI |
| `mge chat [--resume\|--continue\|--fork] [--yolo]` | Line-mode agentic REPL (resumable) |
| `mge run "<prompt>" [--json]` | Headless one-shot for pipes/CI (clean stdout) |
| `mge goal "…" [--until <cmd>] [--max N]` | Autonomous goal loop until done |
| `mge fix "<cmd>" [--max N]` | Iterate until a shell command passes |
| `mge rewind [seq] [--force]` | List / restore file-edit checkpoints |
| `mge plan "<task>"` | Research read-only → approve → execute |
| `mge map` | Print the repo map (codebase orientation) |
| `mge sessions` | List saved sessions (resume with `--resume <id>`) |
| `mge models [query]` | Browse the live model catalog (OpenRouter + NIM) |
| `mge commands` | List custom slash commands |
| `mge doctor` | Show resolved config, routes, key presence |
| `mge gpu` | Local GPU / VRAM (for local-vs-remote routing) |
| `mge mcp` | Connect to MCP servers and list their tools |
| `mge skills` | List discovered `SKILL.md` skills |
| `mge market search/info/install <q>` | Find & add MCP servers from the registry |
| `mge stats` | Tool-usage stats |
| `mge prune` | Report never-used MCP tools (trim candidates) |

In the TUI: `Enter` send · `Esc`/`Ctrl-C` quit · `↑/↓ PgUp/PgDn` scroll ·
`Ctrl-P/N` recall input · **Shift+Tab** cycles permission mode · `/help` for slash
commands (`/model` `/auto` `/effort` `/mode` `/rewind` `/context` `/clear`).

## ⚙️ Configuration

Config lives at `~/.config/mge/config.toml`; secrets at `~/.config/mge/secrets.env`.

```toml
default_route = "main"
auto_route = true            # pick fast/main/heavy per task automatically

[providers.openrouter]
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"

[models.main]                # the default coder
provider = "openrouter"
model = "qwen/qwen3-coder:free"
fallback = ["heavy", "fast"] # auto-retry on rate-limit / 5xx

[models.fast]   # trivial tasks      [models.heavy]  # reasoning-heavy
# ...                                # ...

# Connect any MCP server — its tools become mcp__<name>__<tool>
[mcp]
enabled = true
[mcp.servers.filesystem]
transport = "stdio"          # or "http" with a `url`
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/path"]
sandbox = "process"          # off | process (NO_NEW_PRIVS) | bwrap
tools_allow = ["read_", "list_"]   # optional: only expose these tool prefixes
```

- **Skills** live in `~/.config/mge/skills/<name>/SKILL.md` (trusted). Project
  skills (`./.mge/skills`) are opt-in via `[skills].trust_project_skills = true`.
- **Plugins** live in `~/.config/mge/plugins/<name>/` and may bundle `skills/`
  and an `mcp.json`.
- **Local GPU**: see [`docs/LOCAL_LLAMA.md`](docs/LOCAL_LLAMA.md) to build
  llama.cpp and add a `[models.local]` route.

## 🏗️ Architecture

```
src/
  main.rs        CLI (clap subcommands)
  config.rs      TOML + env-var secrets
  llm/           provider trait + OpenAI-compatible SSE streaming
  routing.rs     fallback chains + per-task classifier + GPU gate
  agent/         the agentic tool-use loop
  tools/         built-in tools (read/write/edit/bash/glob/grep/tree/web_fetch)
  mcp.rs         MCP client (stdio + HTTP), rug-pull fingerprinting, sandbox
  skills.rs      SKILL.md loader + use_skill tool
  plugins.rs     plugin loader (fans out to skills + mcp)
  market.rs      MCP registry search / install
  telemetry.rs   tool-usage log (informs prune)
  tui/           ratatui frontend
  sprite.rs      half-block pixel-art mascot + animation
```

MCP/skill/plugin tools all implement the same `Tool` trait, so the agent loop is
unchanged no matter where a tool comes from.

## 🔒 Security

The agent runs shell commands and connects to third-party MCP servers, so the
trust boundary is taken seriously:

- **Prompt-injection-resistant tool calls.** Models that lack native structured
  tool-calling (local Qwen/Hermes) have their calls parsed from text; models that
  *do* have it (cloud APIs) do not — so a cloud model quoting a malicious file
  containing `<function=bash>…` markup can't be tricked into executing it. Set
  per provider via `text_tool_calls`.
- **Approval prompts.** In the TUI, `bash` and `delegate` ask for `y/n` before
  running (outside `yolo` mode); `deny`/`plan` rules are always enforced.
- **Secrets are stripped** from every child process — `bash`, LSP servers, hooks,
  `delegate`, **and** MCP stdio servers — and config/secrets/checkpoints/embed
  caches are written atomically at `0600`.
- **SSRF guard with DNS-rebinding protection** on `web_fetch` and the embeddings
  endpoint: loopback/private/link-local hosts are blocked, *and* the resolved IP
  is checked, so a public name pointing at `169.254.169.254` is refused too.
- **MCP hardening** — tool schemas are SHA-256 fingerprinted and **blocked on
  drift** (rug-pull defense) until you re-approve; HTTP endpoints are SSRF-guarded;
  spawned servers get `NO_NEW_PRIVS` and optional `bwrap` namespace isolation.
- Registry installs are TOML-injection-safe; project-repo skills/`CLAUDE.md` are
  untrusted by default. Review the code before pointing it at untrusted repos or
  servers — it is personal-use software, audited but not formally pen-tested.

## 📜 License

MIT. PRs welcome. 🐐

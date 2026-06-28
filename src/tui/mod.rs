//! Full-screen ratatui frontend for MGE_GOAT — the "Editorial Goat" look.
//!
//! Layout: a masthead header with status chips, a hairline rule, a body band
//! with the conversation pane beside a right column tri-sected into the animated
//! goat, a thinking-status line, and a live plan/activity list, then an input box
//! and a key-hint footer.
//!
//! Palette is load-bearing, never decorative:
//!   WHITE  = primary signal (wordmark, your text, titles)
//!   GRAY   = structure (borders, rules, hints, settled items)
//!   BLUE   = the goat's voice / working state (spinner, model chip)
//!   PINK   = heat / motion / your turn (melt drips, active item, caret)
//!
//! The agent runs in a worker task and streams [`AgentEvent`]s over a channel,
//! so the UI stays responsive and the goat keeps melting during inference.

use crate::agent::{Agent, AgentEvent};
use crate::config::Config;
use crate::routing::{self, Candidate};
use crate::sprite;
use crate::tools::Registry;
use anyhow::Result;
use crossterm::event::{Event, EventStream, KeyCode, KeyEvent, KeyModifiers};
use crossterm::{execute, terminal};
use futures_util::StreamExt;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Wrap};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

// ── Palette — matches the pixel-art mascot, vibrant ──────────────────────────
const BLUE: Color = Color::Rgb(125, 114, 224); // blueberry — agent voice / working
const GRAY: Color = Color::Rgb(150, 160, 178); // structure / borders
const PINK: Color = Color::Rgb(245, 106, 138); // strawberry — hot / your turn
const WHITE: Color = Color::Rgb(236, 239, 247); // signal
const PISTACHIO: Color = Color::Rgb(224, 168, 102); // biscuit — "settled" / done
const YELLOW: Color = Color::Rgb(255, 225, 70); // goat-eye gold — spinner / highlight
const TEAR: Color = Color::Rgb(169, 208, 255); // tear blue — info
const ADD: Color = Color::Rgb(126, 199, 138); // diff + (added)
const DEL: Color = Color::Rgb(240, 138, 138); // diff − (removed)
/// Max diff lines shown per side in the TUI before eliding.
const DIFF_MAX: usize = 40;

const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

const HELP_LINES: &[&str] = &[
    "commands:",
    "  /help            this help",
    "  /clear           reset conversation (new context)",
    "  /context         show context size",
    "  /cost            show estimated token usage this session",
    "  /model <id>      switch to a route OR any model id (e.g. provider:model)",
    "  /auto            re-enable per-task auto-routing",
    "  /effort <level>  reasoning effort: low|medium|high|xhigh|off",
    "  /mode <mode>     permissions: default|acceptEdits|plan|yolo (Shift+Tab cycles)",
    "  /rewind [seq]    list file-edit checkpoints · /rewind <seq> restores one",
    "  @<path>          inject a file's contents into your message",
    "  /commands        list custom slash commands (~/.config/mge/commands)",
    "  /quit            exit",
    "keys: ↑/↓ pgup/pgdn scroll · ctrl-p/n recall input · ctrl-u clear · esc quit",
];

/// Rough token estimate (~4 chars/token) for the context-size indicator.
fn approx_tokens(s: &str) -> usize {
    s.chars().count() / 4 + 1
}

/// Compact count for chips: 1234 → "1.2k".
fn fmt_k(n: usize) -> String {
    if n < 1000 {
        n.to_string()
    } else {
        format!("{:.1}k", n as f64 / 1000.0)
    }
}

fn gray() -> Style {
    Style::default().fg(GRAY)
}
fn rounded(border: Color, title: &str) -> Block<'_> {
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border))
        .title(Span::styled(title.to_string(), Style::default().fg(WHITE)))
}

/// A plan/activity entry shown in the right column.
struct Act {
    done: bool,
    failed: bool,
    label: String,
}

enum WorkerMsg {
    Event(AgentEvent),
    Done {
        error: Option<String>,
        tokens_in: usize,
        tokens_out: usize,
    },
}

/// Control message from the UI to the agent worker.
enum Ctrl {
    Run {
        input: String,
        media: Vec<serde_json::Value>,
    },
    Reset,
    SetCandidates(Vec<Candidate>),
    SetEffort(Option<String>),
    SetMode(crate::permissions::Mode),
}

struct App {
    header: String,
    cfg: Config,
    route: String,
    auto_route: bool,                         // pick a tier per task
    manual_route: bool,                       // user pinned a route via /model
    mode: crate::permissions::Mode,           // permission posture (Shift+Tab / /mode)
    session_in: usize,                        // cumulative estimated input tokens
    session_out: usize,                       // cumulative estimated output tokens
    commands: crate::commands::CommandLoader, // custom /slash macros
    transcript: Vec<Line<'static>>,
    pending: String,
    pending_reasoning: String,
    input: String,
    thinking: bool,
    tick: usize,
    turn_start: Option<Instant>,
    activity: Vec<Act>,
    ctx_tokens: usize,   // rough token estimate of the live conversation
    inputs: Vec<String>, // submitted-prompt history
    hist_idx: Option<usize>,
    scroll: u16,
    follow: bool,
    last_bottom: u16,
    quit: bool,
    /// Set after the first mid-turn Esc; a second Esc then quits.
    confirm_quit: bool,
    /// A pending tool-approval prompt from the agent worker (bash/delegate in a
    /// gating mode). While `Some`, keystrokes drive the y/n overlay, not input.
    pending_approval: Option<crate::tools::ApprovalRequest>,
}

impl App {
    fn new(header: String, cfg: Config, route: String, mode: crate::permissions::Mode) -> Self {
        let auto_route = cfg.auto_route;
        let commands = crate::commands::CommandLoader::discover(&cfg);
        let mut app = Self {
            header,
            cfg,
            route,
            auto_route,
            manual_route: false,
            mode,
            session_in: 0,
            session_out: 0,
            commands,
            transcript: Vec::new(),
            pending: String::new(),
            pending_reasoning: String::new(),
            input: String::new(),
            thinking: false,
            tick: 0,
            turn_start: None,
            activity: Vec::new(),
            ctx_tokens: 0,
            inputs: Vec::new(),
            hist_idx: None,
            scroll: 0,
            follow: true,
            last_bottom: 0,
            quit: false,
            confirm_quit: false,
            pending_approval: None,
        };
        app.transcript.push(Line::from(Span::styled(
            "welcome — describe a task and press Enter · /help for commands · the goat is listening.",
            Style::default().fg(GRAY).add_modifier(Modifier::ITALIC),
        )));
        app
    }

    /// Push a dim info line (command output / notices).
    fn note(&mut self, msg: impl Into<String>) {
        self.transcript.push(Line::from(Span::styled(
            format!("  {}", msg.into()),
            Style::default().fg(TEAR),
        )));
        self.follow = true;
    }

    fn push_user(&mut self, text: &str) {
        self.transcript.push(Line::from(""));
        self.transcript.push(Line::from(Span::styled(
            "▍ you",
            Style::default().fg(WHITE).add_modifier(Modifier::BOLD),
        )));
        self.transcript.push(Line::from(Span::styled(
            format!("  {text}"),
            Style::default().fg(WHITE),
        )));
    }

    /// Push up to `DIFF_MAX` colored diff lines (`prefix` + content), eliding the rest.
    fn push_diff_lines(&mut self, text: &str, color: Color, prefix: &str) {
        let lines: Vec<&str> = text.lines().collect();
        let shown = lines.len().min(DIFF_MAX);
        for line in &lines[..shown] {
            self.transcript.push(Line::from(Span::styled(
                format!("{prefix}{line}"),
                Style::default().fg(color),
            )));
        }
        if lines.len() > shown {
            self.transcript.push(Line::from(Span::styled(
                format!("  … ({} more lines)", lines.len() - shown),
                Style::default().fg(GRAY).add_modifier(Modifier::DIM),
            )));
        }
    }

    /// Flush buffered reasoning as a dim "💭 thinking" block (distinct from the answer).
    fn flush_reasoning(&mut self) {
        if self.pending_reasoning.trim().is_empty() {
            self.pending_reasoning.clear();
            return;
        }
        let text = std::mem::take(&mut self.pending_reasoning);
        self.transcript.push(Line::from(Span::styled(
            "  💭 thinking",
            Style::default().fg(GRAY).add_modifier(Modifier::ITALIC),
        )));
        for line in text.split('\n').map(|l| l.trim_end_matches('\r')) {
            self.transcript.push(Line::from(Span::styled(
                format!("  {line}"),
                Style::default().fg(GRAY).add_modifier(Modifier::DIM),
            )));
        }
    }

    fn flush_pending(&mut self) {
        self.flush_reasoning();
        if self.pending.is_empty() {
            return;
        }
        let text = std::mem::take(&mut self.pending);
        self.ctx_tokens += approx_tokens(&text);
        self.transcript.push(Line::from(Span::styled(
            "◆ goat",
            Style::default().fg(BLUE).add_modifier(Modifier::BOLD),
        )));
        for line in text.split('\n').map(|l| l.trim_end_matches('\r')) {
            self.transcript.push(Line::from(Span::styled(
                format!("  {line}"),
                Style::default().fg(WHITE),
            )));
        }
    }

    fn on_msg(&mut self, msg: WorkerMsg) {
        match msg {
            WorkerMsg::Event(AgentEvent::Text(t)) => {
                self.flush_reasoning(); // reasoning precedes the answer
                self.pending.push_str(&t);
            }
            WorkerMsg::Event(AgentEvent::Reasoning(t)) => self.pending_reasoning.push_str(&t),
            WorkerMsg::Event(AgentEvent::ToolStart { name, args }) => {
                self.flush_pending();
                let (icon, verb) = crate::agent::tool_glyph(&name);
                let head = if verb.is_empty() { name.as_str() } else { verb };
                let short = crate::util::clip(&args, 50);
                self.activity.push(Act {
                    failed: false,
                    done: false,
                    label: format!("{head} {short}"),
                });
                self.transcript.push(Line::from(Span::styled(
                    format!("  {icon} {head} {short}"),
                    Style::default().fg(BLUE),
                )));
            }
            WorkerMsg::Event(AgentEvent::ToolEnd { name, preview }) => {
                let failed = preview
                    .trim_start()
                    .to_ascii_lowercase()
                    .starts_with("error");
                if let Some(a) = self.activity.iter_mut().rev().find(|a| !a.done) {
                    a.done = true;
                    a.failed = failed;
                }
                self.transcript.push(Line::from(Span::styled(
                    format!("  ↳ {name}: {preview}"),
                    Style::default()
                        .fg(if failed { PINK } else { GRAY })
                        .add_modifier(Modifier::DIM),
                )));
            }
            WorkerMsg::Event(AgentEvent::Diff { path, old, new }) => {
                self.flush_pending();
                self.activity.push(Act {
                    failed: false,
                    done: false,
                    label: format!("edit {path}"),
                });
                self.transcript.push(Line::from(Span::styled(
                    format!("  ✎ edit {path}"),
                    Style::default().fg(PISTACHIO),
                )));
                self.push_diff_lines(&old, DEL, "  - ");
                self.push_diff_lines(&new, ADD, "  + ");
            }
            WorkerMsg::Event(AgentEvent::Wrote { path, content }) => {
                self.flush_pending();
                let n = content.lines().count();
                self.activity.push(Act {
                    failed: false,
                    done: false,
                    label: format!("write {path}"),
                });
                self.transcript.push(Line::from(Span::styled(
                    format!("  ✎ write {path}  (+{n} lines)"),
                    Style::default().fg(PISTACHIO),
                )));
                self.push_diff_lines(&content, ADD, "  + ");
            }
            WorkerMsg::Event(AgentEvent::Notice(m)) => {
                self.transcript.push(Line::from(Span::styled(
                    format!("  ↻ {m}"),
                    Style::default().fg(TEAR),
                )));
            }
            WorkerMsg::Done {
                error,
                tokens_in,
                tokens_out,
            } => {
                // If the turn ended (esp. a crash) while a tool approval was pending,
                // dismiss the modal and deny the dead call — otherwise the overlay
                // sticks and y/n give misleading feedback on a receiver that's gone.
                if let Some(req) = self.pending_approval.take() {
                    let _ = req.reply.send(false);
                }
                self.flush_pending();
                if let Some(e) = error {
                    self.transcript.push(Line::from(Span::styled(
                        format!("  ✖ {e}"),
                        Style::default().fg(PINK).add_modifier(Modifier::BOLD),
                    )));
                }
                self.session_in = tokens_in;
                self.session_out = tokens_out;
                self.thinking = false;
                self.turn_start = None;
            }
        }
        // NOTE: do NOT force follow here. Forcing it on every streamed event yanked
        // the view to the bottom mid-inference, so scrolling up to read was
        // impossible. Follow now persists: true auto-tails, false (user scrolled up)
        // stays put; `submit`/`note` re-enable it when new content is user-initiated.
    }

    fn submit(&mut self, tx: &mpsc::UnboundedSender<Ctrl>) {
        let t = self.input.trim().to_string();
        if t.is_empty() {
            return;
        }
        // Don't clear the box until we know we'll act on the text — otherwise a
        // message typed mid-turn vanishes silently with the input lost.
        if self.thinking && !t.starts_with('/') {
            self.note("busy — let the current turn finish, then resend.");
            return;
        }
        self.input.clear();
        self.hist_idx = None;
        if let Some(cmd) = t.strip_prefix('/') {
            self.command(cmd, tx);
            return;
        }
        self.inputs.push(t.clone());
        // Expand @file mentions for the model; keep the original `t` for display.
        let exp = crate::mentions::expand(&t);
        for n in exp.notices {
            self.note(n);
        }
        self.ctx_tokens += approx_tokens(&exp.text);
        let (model_input, media) = (exp.text, exp.media);

        // Per-task auto-routing: classify the prompt and switch tier (unless the
        // user pinned a route with /model).
        if self.auto_route && !self.manual_route {
            let tier = routing::classify(&t);
            let route = if self.cfg.models.contains_key(tier) {
                tier
            } else {
                self.route.as_str()
            };
            if let Ok(c) = routing::candidates_for(&self.cfg, route) {
                self.header = format!("auto·{route} → {}", c[0].label);
                let _ = tx.send(Ctrl::SetCandidates(c));
            }
        }

        self.push_user(&t); // display the user's literal text
        self.activity.clear();
        self.thinking = true;
        self.turn_start = Some(Instant::now());
        self.follow = true;
        let _ = tx.send(Ctrl::Run {
            input: model_input,
            media,
        }); // model sees expanded @mentions
    }

    /// Handle a `/command`.
    fn command(&mut self, cmd: &str, tx: &mpsc::UnboundedSender<Ctrl>) {
        let mut it = cmd.split_whitespace();
        let name = it.next().unwrap_or("");
        let arg = it.collect::<Vec<_>>().join(" ");
        match name {
            "help" | "?" => {
                for l in HELP_LINES {
                    self.note(*l);
                }
            }
            "clear" | "reset" => {
                self.transcript.clear();
                self.activity.clear();
                self.pending.clear();
                self.ctx_tokens = 0;
                self.session_in = 0;
                self.session_out = 0;
                let _ = tx.send(Ctrl::Reset);
                self.note("context cleared — fresh conversation.");
            }
            "context" | "ctx" => {
                self.note(format!(
                    "context ≈ {} tokens · {} prompt(s) this session",
                    self.ctx_tokens,
                    self.inputs.len()
                ));
            }
            "cost" | "tokens" => {
                self.note(format!(
                    "~{} in / {} out tokens this session (estimated; free models cost $0)",
                    self.session_in, self.session_out
                ));
            }
            "model" | "route" => {
                if arg.is_empty() {
                    let routes: Vec<&str> = self.cfg.models.keys().map(String::as_str).collect();
                    self.note(format!(
                        "route: {} · usage /model <name> · available: {}",
                        self.route,
                        routes.join(", ")
                    ));
                } else {
                    // candidates_for resolves a named route OR an arbitrary model
                    // spec (`provider:id` / bare id), so any model works.
                    match routing::candidates_for(&self.cfg, &arg) {
                        Ok(c) => {
                            self.header = format!("{arg} → {}", c[0].label);
                            self.route = arg.clone();
                            self.manual_route = true;
                            let _ = tx.send(Ctrl::SetCandidates(c));
                            self.note(format!(
                                "switched to '{arg}' (auto-routing off — /auto to re-enable)."
                            ));
                        }
                        Err(e) => self.note(format!("can't switch: {e:#}")),
                    }
                }
            }
            "auto" => {
                if !self.auto_route {
                    self.note("auto-routing isn't configured (set auto_route = true in config).");
                } else {
                    self.manual_route = false;
                    self.note("auto-routing on — picks fast/main/heavy per task.");
                }
            }
            "effort" => {
                let lvl = arg.trim().to_lowercase();
                match lvl.as_str() {
                    "" => {
                        self.note("reasoning effort — usage: /effort <low|medium|high|xhigh|off>")
                    }
                    "off" | "none" | "clear" | "default" => {
                        let _ = tx.send(Ctrl::SetEffort(None));
                        self.note("reasoning effort cleared (model default).");
                    }
                    "low" | "medium" | "high" | "xhigh" => {
                        let _ = tx.send(Ctrl::SetEffort(Some(lvl.clone())));
                        self.note(format!(
                            "reasoning effort → '{lvl}' (applies to models that honor it)."
                        ));
                    }
                    _ => self.note("effort levels: low | medium | high | xhigh | off"),
                }
            }
            "rewind" => match crate::checkpoint::find_latest_journal() {
                None => self.note("no checkpoints yet this session."),
                Some(j) => {
                    let entries = crate::checkpoint::load_journal(&j);
                    let a = arg.trim();
                    if a.is_empty() {
                        self.note("checkpoints (newest last) — /rewind <seq> to restore:");
                        for e in entries
                            .iter()
                            .rev()
                            .take(12)
                            .collect::<Vec<_>>()
                            .into_iter()
                            .rev()
                        {
                            let st = if e.skipped {
                                "unrestorable".to_string()
                            } else if !e.existed {
                                "new→delete".to_string()
                            } else {
                                format!("{}b", e.prior.as_deref().map(str::len).unwrap_or(0))
                            };
                            self.note(format!("  #{} {} {} · {st}", e.seq, e.tool, e.path));
                        }
                    } else if let Ok(seq) = a.parse::<u64>() {
                        match entries.iter().find(|e| e.seq == seq) {
                            Some(e) => match crate::checkpoint::restore(e) {
                                Ok(m) => self.note(format!("✓ {m}")),
                                Err(err) => self.note(format!("rewind failed: {err:#}")),
                            },
                            None => self.note(format!("no snapshot #{seq}")),
                        }
                    } else {
                        self.note("usage: /rewind [seq]");
                    }
                }
            },
            "mode" => {
                let m = arg.trim();
                if m.is_empty() {
                    self.note(format!(
                        "permission mode: {} — usage: /mode <default|acceptEdits|plan|yolo> (or Shift+Tab to cycle)",
                        self.mode.label()
                    ));
                } else if !matches!(
                    m.to_lowercase().replace(['_', '-'], "").as_str(),
                    "default" | "acceptedits" | "plan" | "yolo"
                ) {
                    self.note(format!(
                        "unknown mode '{m}' — use default | acceptEdits | plan | yolo"
                    ));
                } else {
                    let new = crate::permissions::Mode::from(m);
                    self.mode = new;
                    let _ = tx.send(Ctrl::SetMode(new));
                    let extra = if new == crate::permissions::Mode::Plan {
                        " (read-only: blocks ALL writes & bash)"
                    } else {
                        ""
                    };
                    self.note(format!("permission mode → {}{extra}", new.label()));
                }
            }
            "quit" | "exit" | "q" => self.quit = true,
            "commands" => {
                if self.commands.is_empty() {
                    self.note("no custom commands. Drop a <name>.md in ~/.config/mge/commands/.");
                } else {
                    let lines: Vec<String> = self
                        .commands
                        .list()
                        .iter()
                        .map(|c| format!("  /{} {}", c.name, c.description))
                        .collect();
                    self.note("custom commands:");
                    for l in lines {
                        self.note(l);
                    }
                }
            }
            // Fall back to a custom slash command (builtins above always win).
            other => match self.commands.get(other).map(|c| c.expand(&arg)) {
                Some(_) if self.thinking => {
                    self.note("busy — let the current turn finish before running a command.");
                }
                Some(expanded) => {
                    // Expand any @file mentions in the command body too.
                    let exp = crate::mentions::expand(&expanded);
                    for n in exp.notices {
                        self.note(n);
                    }
                    self.push_user(&format!("/{other} {arg}"));
                    self.activity.clear();
                    self.thinking = true;
                    self.turn_start = Some(Instant::now());
                    self.follow = true;
                    let _ = tx.send(Ctrl::Run {
                        input: exp.text,
                        media: exp.media,
                    });
                }
                None => self.note(format!("unknown command '/{other}' — try /help")),
            },
        }
    }

    fn recall(&mut self, back: bool) {
        if self.inputs.is_empty() {
            return;
        }
        let idx = match (self.hist_idx, back) {
            (None, true) => self.inputs.len() - 1,
            (Some(i), true) => i.saturating_sub(1),
            (Some(i), false) if i + 1 < self.inputs.len() => i + 1,
            (Some(_), false) => {
                self.hist_idx = None;
                self.input.clear();
                return;
            }
            (None, false) => return,
        };
        self.hist_idx = Some(idx);
        self.input = self.inputs[idx].clone();
    }

    fn on_key(&mut self, key: KeyEvent, tx: &mpsc::UnboundedSender<Ctrl>) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        // A pending approval captures input: only y / n / esc answer it. Ctrl-C
        // still quits (handled below by dropping the reply → worker sees deny).
        if self.pending_approval.is_some()
            && !(ctrl && matches!(key.code, KeyCode::Char('c' | 'd')))
        {
            match key.code {
                KeyCode::Char('y' | 'Y') => {
                    if let Some(req) = self.pending_approval.take() {
                        let _ = req.reply.send(true);
                        self.note("✓ approved");
                    }
                }
                KeyCode::Char('n' | 'N') | KeyCode::Esc => {
                    if let Some(req) = self.pending_approval.take() {
                        let _ = req.reply.send(false);
                        self.note("✗ denied");
                    }
                }
                _ => {} // ignore anything else; keep the prompt up
            }
            return;
        }
        // Esc confirmation only persists for the immediately following keypress.
        let esc_armed = std::mem::take(&mut self.confirm_quit);
        match key.code {
            // Control combos first — but NEVER blanket-return, or we'd swallow
            // Backspace on terminals that send it as Ctrl-H.
            KeyCode::Char('c' | 'd') if ctrl => self.quit = true,
            KeyCode::Char('h') if ctrl => {
                self.input.pop(); // Ctrl-H == Backspace on many terminals
            }
            KeyCode::Char('u') if ctrl => self.input.clear(), // clear line
            KeyCode::Char('p') if ctrl => self.recall(true),  // previous input
            KeyCode::Char('n') if ctrl => self.recall(false), // next input
            KeyCode::Char('w') if ctrl => {
                // delete the last word — advance past the FULL whitespace char
                // (may be multi-byte, e.g. U+3000) so truncate lands on a boundary.
                let end = self.input.trim_end();
                let cut = end
                    .rfind(char::is_whitespace)
                    .map(|i| i + end[i..].chars().next().map_or(1, char::len_utf8));
                match cut {
                    Some(c) => self.input.truncate(c),
                    None => self.input.clear(),
                }
            }
            _ if ctrl => {} // ignore any other control combo
            KeyCode::BackTab => {
                // Shift+Tab cycles the permission mode.
                self.mode = self.mode.next();
                let _ = tx.send(Ctrl::SetMode(self.mode));
                self.note(format!("permission mode → {}", self.mode.label()));
            }
            KeyCode::Esc => {
                // Mid-turn, Esc is easy to hit by accident (dismiss autocomplete);
                // require a confirming second press so an in-flight edit isn't lost.
                if self.thinking && !esc_armed {
                    self.confirm_quit = true;
                    self.note("turn in progress — press Esc again to abort and quit.");
                } else {
                    self.quit = true;
                }
            }
            KeyCode::Enter => self.submit(tx),
            KeyCode::Backspace | KeyCode::Delete => {
                self.input.pop();
            }
            KeyCode::Char('\u{7f}') | KeyCode::Char('\u{8}') => {
                self.input.pop();
            }
            KeyCode::Char(c) => self.input.push(c),
            KeyCode::Up => {
                if self.follow {
                    self.scroll = self.last_bottom;
                    self.follow = false;
                }
                self.scroll = self.scroll.saturating_sub(1);
            }
            KeyCode::PageUp => {
                if self.follow {
                    self.scroll = self.last_bottom;
                    self.follow = false;
                }
                self.scroll = self.scroll.saturating_sub(10);
            }
            KeyCode::Down => {
                self.scroll = self.scroll.saturating_add(1);
                if self.scroll >= self.last_bottom {
                    self.follow = true;
                }
            }
            KeyCode::PageDown => {
                self.scroll = self.scroll.saturating_add(10);
                if self.scroll >= self.last_bottom {
                    self.follow = true;
                }
            }
            _ => {}
        }
    }

    /// Total logical lines in the conversation view (settled transcript + the live
    /// streaming "pending" block). Cheap — no allocation — for scroll math.
    fn display_total(&self) -> usize {
        let pending = if self.pending.is_empty() {
            0
        } else {
            1 + self.pending.split('\n').count() // "◆ goat" header + wrapped pending
        };
        self.transcript.len() + pending
    }

    /// The conversation view as `Line`s, BORROWING transcript span text instead of
    /// deep-copying every String each frame (the per-frame clone was the cost — for
    /// a long session it copied thousands of heap strings ~10×/s). Only the small
    /// live "pending" block allocates. Same line sequence as before, so the scroll
    /// math is unchanged. (ratatui 0.30 keeps `line_count` private, so a true
    /// O(viewport) slice can't land the wrapped bottom without re-implementing its
    /// word-wrap — borrowing is the safe, behavior-identical win.)
    fn display_lines(&self) -> Vec<Line<'_>> {
        let mut out: Vec<Line<'_>> = self
            .transcript
            .iter()
            .map(|line| {
                let spans: Vec<Span<'_>> = line
                    .spans
                    .iter()
                    .map(|s| Span::styled(s.content.as_ref(), s.style))
                    .collect();
                let mut l = Line::from(spans).style(line.style);
                if let Some(a) = line.alignment {
                    l = l.alignment(a);
                }
                l
            })
            .collect();
        if !self.pending.is_empty() {
            out.push(Line::from(Span::styled(
                "◆ goat",
                Style::default().fg(BLUE).add_modifier(Modifier::BOLD),
            )));
            for l in self.pending.split('\n').map(|x| x.trim_end_matches('\r')) {
                out.push(Line::from(Span::styled(
                    format!("  {l}"),
                    Style::default().fg(WHITE),
                )));
            }
        }
        out
    }

    fn elapsed(&self) -> u64 {
        self.turn_start.map(|t| t.elapsed().as_secs()).unwrap_or(0)
    }
}

fn ui(f: &mut ratatui::Frame, app: &mut App) {
    let area = f.area();
    let rows = Layout::vertical([
        Constraint::Length(1), // masthead
        Constraint::Length(1), // hairline rule
        Constraint::Min(0),    // body
        Constraint::Length(3), // input
        Constraint::Length(1), // footer hint
    ])
    .split(area);

    // ── masthead ───────────────────────────────────────────────────────────────
    let head = Layout::horizontal([Constraint::Min(16), Constraint::Length(80)]).split(rows[0]);
    let wordmark = Line::from(vec![
        Span::styled(
            "M G E · G O A T",
            Style::default().fg(WHITE).add_modifier(Modifier::BOLD),
        ),
        Span::styled("   the greatest of all tools", gray()),
    ]);
    f.render_widget(Paragraph::new(wordmark), head[0]);

    let spin = if app.thinking {
        SPINNER[app.tick % SPINNER.len()]
    } else {
        "○"
    };
    let status = if app.thinking {
        format!("{spin} thinking · {}s ", app.elapsed())
    } else {
        "○ ready ".to_string()
    };
    let ctx = if app.ctx_tokens >= 1000 {
        format!("{:.1}k", app.ctx_tokens as f32 / 1000.0)
    } else {
        app.ctx_tokens.to_string()
    };
    let mode_color = match app.mode {
        crate::permissions::Mode::Yolo => YELLOW,
        crate::permissions::Mode::Plan => BLUE,
        // Distinct from inert Default GRAY — accept-edits auto-applies edits.
        crate::permissions::Mode::AcceptEdits => PISTACHIO,
        crate::permissions::Mode::Default => GRAY,
    };
    let chips = Line::from(vec![
        Span::styled("◍ ", Style::default().fg(BLUE)),
        Span::styled(short_model(&app.header), Style::default().fg(GRAY)),
        Span::styled(
            format!("  ⛨ {}", app.mode.label()),
            Style::default().fg(mode_color),
        ),
        Span::styled(format!("  ◐ ctx {ctx}"), Style::default().fg(TEAR)),
        Span::styled(
            format!("  ⛁ {}", fmt_k(app.session_in + app.session_out)),
            Style::default().fg(GRAY),
        ),
        Span::styled(
            format!("  {status}"),
            Style::default().fg(if app.thinking { YELLOW } else { GRAY }),
        ),
    ]);
    f.render_widget(Paragraph::new(chips).alignment(Alignment::Right), head[1]);

    // ── hairline rule ────────────────────────────────────────────────────────────
    let rule = "─".repeat(area.width as usize);
    f.render_widget(Paragraph::new(Span::styled(rule, gray())), rows[1]);

    // ── body: conversation | right column ────────────────────────────────────────
    let body = Layout::horizontal([Constraint::Min(30), Constraint::Length(46)]).split(rows[2]);

    let viewport = body[0].height.saturating_sub(2);
    // Saturate: a very long session (>65535 lines) must not wrap to 0 and break
    // scrolling — clamp to u16::MAX so the newest lines stay reachable. Computed
    // BEFORE borrowing the lines so `last_bottom` can be set without a borrow clash.
    let total = app.display_total().min(u16::MAX as usize) as u16;
    let bottom = total.saturating_sub(viewport);
    app.last_bottom = bottom;
    let scroll_y = if app.follow {
        bottom
    } else {
        app.scroll.min(bottom)
    };
    let lines = app.display_lines();
    let convo = Paragraph::new(Text::from(lines))
        .block(rounded(GRAY, " conversation "))
        .wrap(Wrap { trim: false })
        .scroll((scroll_y, 0));
    f.render_widget(convo, body[0]);

    let right = Layout::vertical([
        Constraint::Length(sprite::rows() + 2),
        Constraint::Length(3),
        Constraint::Min(0),
    ])
    .split(body[1]);

    // mascot: continuous pixel-art animation; `thinking` switches idle↔active.
    let goat_lines = sprite::render(app.thinking, app.tick);
    f.render_widget(
        Paragraph::new(Text::from(goat_lines)).block(rounded(PINK, " 🐐 G.O.A.T 🍦 ")),
        right[0],
    );

    // status
    let status_line = if app.thinking {
        Line::from(vec![
            Span::styled(format!(" {spin} "), Style::default().fg(YELLOW)),
            Span::styled(
                format!("licking · {}s", app.elapsed()),
                Style::default().fg(PINK),
            ),
        ])
    } else {
        Line::from(Span::styled(
            " ready when you are",
            Style::default().fg(GRAY).add_modifier(Modifier::ITALIC),
        ))
    };
    f.render_widget(
        Paragraph::new(status_line).block(rounded(GRAY, " status ")),
        right[1],
    );

    // plan / activity
    let act_inner = right[2].height.saturating_sub(2) as usize;
    let mut act_lines: Vec<Line> = Vec::new();
    if app.activity.is_empty() {
        act_lines.push(Line::from(Span::styled(
            "  idle",
            Style::default().fg(GRAY).add_modifier(Modifier::DIM),
        )));
    } else {
        let start = app.activity.len().saturating_sub(act_inner.max(1));
        for a in &app.activity[start..] {
            if a.done && a.failed {
                act_lines.push(Line::from(vec![
                    Span::styled("  ✖ ", Style::default().fg(PINK)),
                    Span::styled(trunc(&a.label, 30), Style::default().fg(PINK)),
                ]));
            } else if a.done {
                act_lines.push(Line::from(vec![
                    Span::styled("  ▣ ", Style::default().fg(PISTACHIO)),
                    Span::styled(trunc(&a.label, 30), gray()),
                ]));
            } else {
                act_lines.push(Line::from(vec![
                    Span::styled("  ◐ ", Style::default().fg(PINK)),
                    Span::styled(trunc(&a.label, 24), Style::default().fg(WHITE)),
                    Span::styled(" ← now", Style::default().fg(PINK)),
                ]));
            }
        }
    }
    f.render_widget(
        Paragraph::new(Text::from(act_lines)).block(rounded(GRAY, " plan · activity ")),
        right[2],
    );

    // ── input ────────────────────────────────────────────────────────────────────
    let caret = if app.thinking { "" } else { "▌" };
    let input_border = if app.thinking { GRAY } else { PINK };
    // Horizontal scroll: show the TAIL that fits so the caret never clips off-screen
    // when the line is longer than the box (char-based; good enough for ASCII input).
    let inner = (rows[3].width.saturating_sub(2) as usize).saturating_sub(3); // borders+space+caret
    let nchars = app.input.chars().count();
    let shown: String = if inner > 0 && nchars > inner {
        let lead = "…";
        let tail: String = app.input.chars().skip(nchars - inner + 1).collect();
        format!("{lead}{tail}")
    } else {
        app.input.clone()
    };
    let input = Paragraph::new(Line::from(vec![
        Span::styled(format!(" {shown}"), Style::default().fg(WHITE)),
        Span::styled(caret, Style::default().fg(PINK)),
    ]))
    .block(rounded(input_border, " ▌ ask the goat "));
    f.render_widget(input, rows[3]);

    // ── footer hint ──────────────────────────────────────────────────────────────
    let footer = if app.pending_approval.is_some() {
        Span::styled(
            "  ⚠ approval needed — press [y] to run · [n]/esc to deny",
            Style::default().fg(YELLOW).add_modifier(Modifier::BOLD),
        )
    } else {
        Span::styled(
            "  enter send · esc/ctrl-c quit · ↑/↓ pgup/pgdn scroll",
            Style::default().fg(GRAY).add_modifier(Modifier::DIM),
        )
    };
    f.render_widget(Paragraph::new(footer), rows[4]);

    // ── approval overlay (drawn last, on top) ────────────────────────────────────
    if let Some(req) = &app.pending_approval {
        render_approval(f, area, req);
    }
}

/// Centered modal asking the user to approve one gated tool call (bash/delegate).
fn render_approval(
    f: &mut ratatui::Frame,
    area: ratatui::layout::Rect,
    req: &crate::tools::ApprovalRequest,
) {
    let what = match &req.command {
        Some(c) => format!("run shell command:\n\n  {}", trunc(c, 200)),
        None => format!("allow tool `{}`?", req.tool),
    };
    let w = area.width.saturating_sub(8).clamp(20, 72);
    let h = 9u16.min(area.height);
    let x = area.x + (area.width.saturating_sub(w)) / 2;
    let y = area.y + (area.height.saturating_sub(h)) / 2;
    let popup = ratatui::layout::Rect::new(x, y, w, h);
    f.render_widget(ratatui::widgets::Clear, popup); // clear underlying cells
    let body = Text::from(vec![
        Line::from(""),
        Line::from(Span::styled(what, Style::default().fg(WHITE))),
        Line::from(""),
        Line::from(Span::styled(
            "[y] yes    [n]/esc no",
            Style::default().fg(YELLOW).add_modifier(Modifier::BOLD),
        )),
    ]);
    f.render_widget(
        Paragraph::new(body)
            .wrap(Wrap { trim: false })
            .block(rounded(PINK, " 🐐 approve tool call? ")),
        popup,
    );
}

fn short_model(header: &str) -> String {
    // header looks like "main → main:openrouter/qwen/qwen3-coder:free"
    header.rsplit('/').next().unwrap_or(header).to_string()
}
fn trunc(s: &str, n: usize) -> String {
    if s.chars().count() > n {
        format!("{}…", s.chars().take(n).collect::<String>())
    } else {
        s.to_string()
    }
}

/// Restores the terminal on drop (panic/early-return safe).
struct TerminalGuard;
impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = terminal::disable_raw_mode();
        let _ = execute!(std::io::stdout(), terminal::LeaveAlternateScreen);
    }
}

/// Run the TUI event loop until the user quits. Builds the agent from config so
/// the UI can rebuild the candidate chain for `/model` switches.
pub async fn run(
    cfg: Config,
    route: String,
    system: String,
    resume_history: Vec<crate::llm::Message>,
    resume_path: Option<std::path::PathBuf>,
) -> Result<()> {
    // Merge in any installed plugins (extra skills + MCP servers).
    let cfg = crate::plugins::apply(&cfg);
    let candidates = routing::candidates_for(&cfg, &route)?;
    let header = format!("{route} → {}", candidates[0].label);
    let mut tools = Registry::with_defaults();
    // Build the policy with can_prompt=false; `set_approver()` below flips it to
    // true and routes Ask decisions to the in-TUI overlay (the raw-mode terminal
    // can't run the stdin prompt). `Plan`
    // and `deny` rules remain the gates. Default posture from config (AcceptEdits).
    tools.set_policy(crate::permissions::PermissionPolicy::from_config(
        &cfg.permissions,
        false,
    ));
    let tui_mode = cfg
        .permissions
        .mode
        .as_deref()
        .map(crate::permissions::Mode::from)
        .unwrap_or(crate::permissions::Mode::AcceptEdits);
    tools.set_mode(tui_mode);
    let store = std::sync::Arc::new(crate::checkpoint::CheckpointStore::new()?);
    tools.set_checkpoint(store.clone());
    if cfg.checks.enabled {
        tools.set_after_edit_cmd(cfg.checks.after_edit_cmd.clone(), cfg.checks.timeout_secs);
    }
    // Connect MCP servers (graceful skip on failure). `_mcp` must outlive the
    // worker so the registered MCP tools' peers stay connected.
    let (_mcp, mcp_status) = crate::mcp::McpManager::connect(&cfg, &mut tools).await;
    let mcp_tools: Vec<String> = mcp_status.iter().flat_map(|s| s.tools.clone()).collect();
    let market_cfg = cfg.marketplace.clone();
    // Load markdown skills: register `use_skill` and list them in the system prompt.
    let loader = crate::skills::SkillLoader::discover(&cfg);
    loader.register(&mut tools);
    // Subagents: built-in tools only (no `spawn_agent` → depth-1 cap). Inherit Plan
    // (read-only) from the parent; otherwise Yolo (children can't prompt).
    let mut child_tools = Registry::with_defaults();
    let child_mode = if tui_mode == crate::permissions::Mode::Plan {
        crate::permissions::Mode::Plan
    } else {
        crate::permissions::Mode::Yolo
    };
    child_tools.set_mode(child_mode);
    child_tools.set_can_prompt(false);
    child_tools.set_checkpoint(store.clone());
    // Share the child's policy so a runtime /mode switch (esp. → Plan) also
    // constrains spawned subagents — otherwise they keep their startup mode.
    let child_policy = child_tools.policy.clone();
    tools.add(std::sync::Arc::new(
        crate::agent::spawn::SpawnAgentTool::new(cfg.clone(), child_tools),
    ));
    let mut system = system;
    if let Some(mem) = crate::config::project_memory(&cfg) {
        system.push_str("\n\n");
        system.push_str(&mem);
    }
    if let Some(add) = loader.system_addendum() {
        system.push_str("\n\n");
        system.push_str(&add);
    }
    // Route Ask decisions (bash/delegate in a gating mode) to an in-TUI y/n overlay
    // instead of silently allowing — the terminal is owned by the alternate screen,
    // so the stdin prompt can't run here.
    let (approval_tx, mut approval_rx) = mpsc::unbounded_channel::<crate::tools::ApprovalRequest>();
    tools.set_approver(approval_tx);
    let base_candidates = candidates.clone(); // restore point after a media route-swap
    let mut agent = Agent::new(candidates, tools, system);
    // Repo map injected per-turn (query-scoped) by the agent, not statically.
    agent.set_repo_index(
        crate::repo_map::build_index(std::path::Path::new("."), &cfg.repo_map),
        cfg.repo_map.char_budget,
    );
    let resumed = resume_history.len();
    if resumed > 0 {
        agent.load_history(resume_history);
    }

    terminal::enable_raw_mode()?;
    let _guard = TerminalGuard;
    let mut stdout = std::io::stdout();
    execute!(stdout, terminal::EnterAlternateScreen)?;
    let mut term = Terminal::new(CrosstermBackend::new(stdout))?;

    // ponytail: unbounded is the right call here, not laziness. The producers are
    // the synchronous UI key handler (`on_key`, can't .await a bounded send) and a
    // single worker streaming events; volume is tiny (one in-flight turn) so there
    // is no unbounded-growth risk and no backpressure to apply.
    let (in_tx, mut in_rx) = mpsc::unbounded_channel::<Ctrl>();
    let (ui_tx, mut ui_rx) = mpsc::unbounded_channel::<WorkerMsg>();

    // Marketplace daemon (opt-in, notify-only): periodically remind about unused
    // MCP tools. It never installs or deletes anything — `mge prune` is manual.
    if market_cfg.enabled && !mcp_tools.is_empty() {
        let tx = ui_tx.clone();
        let period = Duration::from_secs(market_cfg.interval_mins.max(1) * 60);
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(period);
            loop {
                tick.tick().await;
                let stats = crate::telemetry::stats();
                let unused = mcp_tools
                    .iter()
                    .filter(|t| stats.get(*t).map(|(c, _)| c == &0).unwrap_or(true))
                    .count();
                if unused > 0 {
                    let _ = tx.send(WorkerMsg::Event(AgentEvent::Notice(format!(
                        "{unused} MCP tool(s) unused — `mge prune` to trim them"
                    ))));
                }
            }
        });
    }

    let worker_tx = ui_tx.clone();
    // Persist the TUI conversation so it can be resumed from `mge chat --resume`.
    // Resume into the same file so the TUI can continue a session it saved.
    let session_store = match resume_path {
        Some(p) => crate::session::SessionStore::at(p),
        None => crate::session::SessionStore::new()?,
    };
    // Persist the lossless compaction archive next to the session (loads existing
    // on resume) so recovered context spans sessions.
    agent.set_archive_path(session_store.archive_path());
    let worker_cfg = cfg.clone();
    let worker_route = route.clone();
    // The media route-swap restore point. Must track /model + auto-route changes
    // (updated in the SetCandidates arm), else a media turn would silently revert
    // a user's pin back to the startup route.
    let mut worker_base = base_candidates;
    let mut worker = tokio::spawn(async move {
        while let Some(cmd) = in_rx.recv().await {
            match cmd {
                Ctrl::Run { input, media } => {
                    let tx = worker_tx.clone();
                    // Media turn → swap to a capable (vision/audio) route, restore after.
                    let restore = if media.is_empty() {
                        None
                    } else {
                        let mr = crate::pick_route(&worker_cfg, &media);
                        match crate::routing::candidates_for(&worker_cfg, &mr) {
                            Ok(c) if mr != worker_route => {
                                let _ = tx.send(WorkerMsg::Event(AgentEvent::Notice(format!(
                                    "using '{mr}' for this media turn"
                                ))));
                                agent.set_candidates(c);
                                Some(worker_base.clone())
                            }
                            _ => None,
                        }
                    };
                    let res = agent
                        .run_turn_with_media(&input, media, |ev| {
                            let _ = tx.send(WorkerMsg::Event(ev));
                        })
                        .await;
                    if let Some(b) = restore {
                        agent.set_candidates(b); // back to the pinned route
                    }
                    // Persist only on success (an errored turn can leave a Tool-tail
                    // history that resume would strip back to a User message).
                    if res.is_ok() {
                        session_store.save(agent.history());
                    }
                    let (tokens_in, tokens_out) = agent.session_tokens();
                    let _ = worker_tx.send(WorkerMsg::Done {
                        error: res.err().map(|e| format!("{e:#}")),
                        tokens_in,
                        tokens_out,
                    });
                }
                Ctrl::Reset => agent.reset(),
                Ctrl::SetCandidates(c) => {
                    agent.set_candidates(c.clone());
                    worker_base = c; // keep the media-restore point current
                }
                Ctrl::SetEffort(e) => agent.set_effort(e),
                Ctrl::SetMode(m) => {
                    agent.set_mode(m);
                    // Keep subagents in lockstep — esp. so switching to Plan makes
                    // spawn_agent read-only too, not just the top-level agent.
                    child_policy.lock().unwrap_or_else(|e| e.into_inner()).mode = m;
                }
            }
        }
    });

    let mut app = App::new(header, cfg, route, tui_mode);
    if resumed > 0 {
        app.note(format!(
            "resumed {resumed} message(s) — the agent has prior context (transcript starts fresh)."
        ));
    }
    if tui_mode == crate::permissions::Mode::AcceptEdits {
        app.note("mode: accept-edits — edits auto-apply; bash and delegate ask for y/n approval in the TUI. /mode yolo to skip prompts, /mode plan for read-only, Shift+Tab to cycle.");
    }
    let mut reader = EventStream::new();
    let mut ticker = tokio::time::interval(Duration::from_millis(110));
    let mut worker_dead = false;
    // The agent owns the only approval_tx; when the worker dies it drops and
    // approval_rx.recv() returns None forever. Guard the arm so it doesn't win the
    // select! every iteration and busy-loop the CPU.
    let mut approval_closed = false;

    let result = loop {
        if let Err(e) = term.draw(|f| ui(f, &mut app)) {
            break Err(e.into());
        }
        if app.quit {
            break Ok(());
        }
        tokio::select! {
            // If the worker task ever exits/panics unexpectedly, surface it and
            // clear "thinking" instead of leaving the spinner spinning forever.
            res = &mut worker, if !worker_dead => {
                worker_dead = true;
                let err = match res {
                    Ok(()) => "agent worker exited unexpectedly".to_string(),
                    Err(e) => format!("agent worker crashed: {e}"),
                };
                app.on_msg(WorkerMsg::Done { error: Some(err), tokens_in: 0, tokens_out: 0 });
            }
            maybe = reader.next() => {
                if let Some(Ok(Event::Key(key))) = maybe
                    && key.kind != crossterm::event::KeyEventKind::Release {
                        app.on_key(key, &in_tx);
                    }
            }
            msg = ui_rx.recv() => {
                if let Some(msg) = msg { app.on_msg(msg); }
            }
            req = approval_rx.recv(), if !approval_closed => {
                match req {
                    Some(req) => {
                        app.pending_approval = Some(req);
                        app.follow = true;
                    }
                    None => approval_closed = true, // sender gone (worker exited)
                }
            }
            _ = ticker.tick() => { app.tick = app.tick.wrapping_add(1); } // always animate
        }
    };

    worker.abort();
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    /// Render the UI headless and confirm the mascot's half-blocks land in the
    /// goat panel (right side) and aren't clipped to nothing.
    #[test]
    fn mascot_renders_in_panel() {
        let backend = TestBackend::new(110, 32);
        let mut term = Terminal::new(backend).unwrap();
        let mut app = App::new(
            "main → qwen".to_string(),
            Config::default(),
            "main".to_string(),
            crate::permissions::Mode::AcceptEdits,
        );
        app.thinking = true;
        app.tick = 0;
        term.draw(|f| ui(f, &mut app)).unwrap();
        let buf = term.backend().buffer();
        let (mut blocks, mut right_blocks) = (0, 0);
        for y in 0..32u16 {
            for x in 0..110u16 {
                if buf[(x, y)].symbol() == "▀" {
                    blocks += 1;
                    if x > 60 {
                        right_blocks += 1;
                    }
                }
            }
        }
        assert!(
            blocks > 80,
            "expected many sprite half-blocks, got {blocks}"
        );
        assert!(
            right_blocks > 60,
            "sprite should be in the right panel, got {right_blocks}"
        );
    }

    /// The approval overlay renders its prompt and y/n affordance, and a y/n key
    /// answers the waiting oneshot (the in-TUI replacement for the stdin prompt).
    #[test]
    fn approval_overlay_renders_and_answers() {
        let backend = TestBackend::new(110, 32);
        let mut term = Terminal::new(backend).unwrap();
        let mut app = App::new(
            "main → qwen".to_string(),
            Config::default(),
            "main".to_string(),
            crate::permissions::Mode::AcceptEdits,
        );
        let (reply, rx) = tokio::sync::oneshot::channel::<bool>();
        app.pending_approval = Some(crate::tools::ApprovalRequest {
            tool: "bash".into(),
            command: Some("rm -rf /tmp/x".into()),
            reply,
        });
        term.draw(|f| ui(f, &mut app)).unwrap();
        let text: String = {
            let buf = term.backend().buffer();
            (0..32u16)
                .flat_map(|y| (0..110u16).map(move |x| (x, y)))
                .map(|(x, y)| buf[(x, y)].symbol().to_string())
                .collect()
        };
        assert!(text.contains("rm -rf /tmp/x"), "overlay shows the command");
        assert!(text.contains("[y]"), "overlay shows the y/n affordance");

        // Pressing 'y' answers true and clears the prompt.
        let (tx, _ui_rx) = mpsc::unbounded_channel::<Ctrl>();
        app.on_key(KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE), &tx);
        assert!(app.pending_approval.is_none(), "answer clears the prompt");
        assert_eq!(rx.blocking_recv().ok(), Some(true), "y → approved");
    }

    /// An `edit_file` Diff event renders a real diff in the conversation: the file
    /// path, the removed line in RED (DEL) and the added line in GREEN (ADD). This
    /// is the rich rendering the TUI shows — headless `mge run` prints only the
    /// final answer, so the diff only appears in `mge tui`/`mge chat`.
    #[test]
    fn diff_event_renders_red_removed_and_green_added() {
        let backend = TestBackend::new(110, 32);
        let mut term = Terminal::new(backend).unwrap();
        let mut app = App::new(
            "main → x".to_string(),
            Config::default(),
            "main".to_string(),
            crate::permissions::Mode::AcceptEdits,
        );
        // Same event the agent emits after a successful edit_file.
        app.on_msg(WorkerMsg::Event(AgentEvent::Diff {
            path: "greet.py".to_string(),
            old: "OLDLINEremoved".to_string(),
            new: "NEWLINEadded".to_string(),
        }));
        term.draw(|f| ui(f, &mut app)).unwrap();
        let buf = term.backend().buffer();

        let mut text = String::new();
        let (mut red_on_removed, mut green_on_added) = (false, false);
        for y in 0..32u16 {
            let mut row = String::new();
            for x in 0..110u16 {
                row.push_str(buf[(x, y)].symbol());
            }
            if row.contains("OLDLINEremoved") {
                red_on_removed = (0..110u16).any(|x| buf[(x, y)].fg == DEL);
            }
            if row.contains("NEWLINEadded") {
                green_on_added = (0..110u16).any(|x| buf[(x, y)].fg == ADD);
            }
            text.push_str(&row);
            text.push('\n');
        }
        assert!(text.contains("greet.py"), "shows the edited path");
        assert!(text.contains("OLDLINEremoved"), "shows the removed line");
        assert!(text.contains("NEWLINEadded"), "shows the added line");
        assert!(red_on_removed, "removed line renders in red (DEL)");
        assert!(green_on_added, "added line renders in green (ADD)");
    }
}

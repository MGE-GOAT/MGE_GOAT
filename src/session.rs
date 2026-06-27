//! Session persistence + resume.
//!
//! Each interactive session's conversation (the agent's message history, minus
//! the system prompt) is written to `~/.cache/mge/sessions/<id>.jsonl` so it can
//! be resumed with `mge chat --resume` / `--continue`. The store is intentionally
//! dumb: it serializes whatever slice of [`Message`]s the caller gives it and
//! reads it back — the caller owns the policy of what to persist and how to
//! rehydrate (keeping a fresh system prompt, ensuring the tail stays valid).
//!
//! The file is `0600` because a transcript can contain secrets.

use crate::llm::Message;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// `~/.cache/mge/sessions/`.
pub fn sessions_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("mge").join("sessions"))
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// A handle to one session's transcript file. Save overwrites the whole file
/// (the history is small and this guarantees a consistent, valid snapshot).
pub struct SessionStore {
    path: PathBuf,
}

impl SessionStore {
    /// Create a fresh session file (id = process-start unix seconds, incremented
    /// on collision so two sessions in the same second don't clobber each other).
    pub fn new() -> Result<Self> {
        let dir = sessions_dir().context("no cache dir available for sessions")?;
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        let mut id = now_secs();
        let mut path = dir.join(format!("{id}.jsonl"));
        while path.exists() {
            id += 1;
            path = dir.join(format!("{id}.jsonl"));
        }
        Ok(Self { path })
    }

    /// Reuse an existing transcript file (resume/fork target) for further writes.
    pub fn at(path: PathBuf) -> Self {
        Self { path }
    }

    /// Companion lossless-archive path (`<id>.archive.jsonl`) — where compaction-
    /// evicted messages are appended so retrieval survives across resume.
    pub fn archive_path(&self) -> PathBuf {
        self.path.with_extension("archive.jsonl")
    }

    /// Overwrite the transcript with `history` (one JSON [`Message`] per line),
    /// at `0600`. Best-effort: logs on error, never panics.
    pub fn save(&self, history: &[Message]) {
        let mut buf = String::new();
        for m in history {
            // Skip System messages: the session-start prompt is rebuilt fresh on
            // resume, and a System message spliced mid-conversation (e.g. a
            // compaction summary) is rejected by most providers.
            if m.role == crate::llm::Role::System {
                continue;
            }
            if let Ok(line) = serde_json::to_string(m) {
                buf.push_str(&line);
                buf.push('\n');
            }
        }
        if let Err(e) = write_private(&self.path, &buf) {
            eprintln!("session: cannot write transcript: {e}");
        }
    }
}

/// Write `contents` to `path` truncating, at `0600` on unix.
fn write_private(path: &Path, contents: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut opts = std::fs::OpenOptions::new();
    opts.create(true).write(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut f = opts.open(path)?;
    f.write_all(contents.as_bytes())
}

/// The most-recently-modified session transcript (target of `--continue`).
pub fn find_latest() -> Option<PathBuf> {
    let dir = sessions_dir()?;
    let mut newest: Option<(SystemTime, PathBuf)> = None;
    for entry in std::fs::read_dir(&dir).ok()?.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        if let Ok(mtime) = entry.metadata().and_then(|m| m.modified())
            && newest.as_ref().is_none_or(|(t, _)| mtime > *t)
        {
            newest = Some((mtime, path));
        }
    }
    newest.map(|(_, p)| p)
}

/// Resolve a resume target: an explicit numeric id if given, else the most
/// recent session. The id MUST be a bare numeric timestamp (path-traversal
/// guard) — `../../etc/passwd` and other non-numeric inputs are rejected.
pub fn resolve(id: Option<&str>) -> Option<PathBuf> {
    match id {
        None => find_latest(),
        Some(s) => {
            let stem = s.strip_suffix(".jsonl").unwrap_or(s);
            if stem.is_empty() || !stem.bytes().all(|b| b.is_ascii_digit()) {
                return None; // reject anything that isn't a numeric session id
            }
            let dir = sessions_dir()?;
            let cand = dir.join(format!("{stem}.jsonl"));
            cand.is_file().then_some(cand)
        }
    }
}

/// Trim a loaded transcript to a valid resume boundary: drop trailing System
/// messages (compaction summaries), orphaned tool results, and unanswered
/// assistant tool-calls — the kind of dangling tail a session killed mid-turn
/// leaves behind, which would otherwise make the provider reject the next call.
pub fn validate_and_truncate(mut history: Vec<Message>) -> Vec<Message> {
    use crate::llm::Role;
    while let Some(last) = history.last() {
        let dangling = last.role == Role::System
            || last.role == Role::Tool
            || (last.role == Role::Assistant && !last.tool_calls.is_empty());
        if dangling {
            history.pop();
        } else {
            break;
        }
    }
    history
}

/// Load a transcript, skipping any unparseable lines.
/// Append messages to a 0600 JSONL archive (append-only, never rewritten).
pub fn append_archive(path: &Path, msgs: &[Message]) {
    use std::io::Write;
    let mut buf = String::new();
    for m in msgs {
        if let Ok(line) = serde_json::to_string(m) {
            buf.push_str(&line);
            buf.push('\n');
        }
    }
    if buf.is_empty() {
        return;
    }
    #[cfg(unix)]
    let opened = {
        use std::os::unix::fs::OpenOptionsExt;
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .mode(0o600)
            .open(path)
    };
    #[cfg(not(unix))]
    let opened = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path);
    match opened {
        Ok(mut f) => {
            if let Err(e) = f.write_all(buf.as_bytes()) {
                eprintln!("[mge] warning: archive write failed: {e}");
            }
        }
        Err(e) => eprintln!("[mge] warning: cannot open archive {}: {e}", path.display()),
    }
}

/// Load a lossless archive JSONL (not validated/truncated). Empty if missing.
pub fn load_archive_file(path: &Path) -> Vec<Message> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    parse_jsonl(&text, "archive", path)
}

/// Parse JSONL into messages, warning (not silently dropping) on corrupt lines so
/// a truncated/garbled transcript is distinguishable from an empty one.
fn parse_jsonl(text: &str, kind: &str, path: &Path) -> Vec<Message> {
    let mut out = Vec::new();
    let mut bad = 0usize;
    for l in text.lines() {
        if l.trim().is_empty() {
            continue;
        }
        match serde_json::from_str(l) {
            Ok(m) => out.push(m),
            Err(_) => bad += 1,
        }
    }
    if bad > 0 {
        eprintln!(
            "[mge] warning: skipped {bad} corrupt line(s) in {kind} {}",
            path.display()
        );
    }
    out
}

pub fn load(path: &Path) -> Vec<Message> {
    match std::fs::read_to_string(path) {
        Ok(s) => parse_jsonl(&s, "session", path),
        Err(e) => {
            // A resolved session/archive path that won't read (perms/IO) must not
            // silently resume as an empty conversation — surface it.
            if e.kind() != std::io::ErrorKind::NotFound {
                eprintln!("session: cannot read {}: {e}", path.display());
            }
            Vec::new()
        }
    }
}

/// List recent sessions (path + message count), newest first.
pub fn list_recent(limit: usize) -> Vec<(PathBuf, usize)> {
    let Some(dir) = sessions_dir() else {
        return vec![];
    };
    let mut all: Vec<(SystemTime, PathBuf)> = std::fs::read_dir(&dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) != Some("jsonl") {
                return None;
            }
            let mtime = e.metadata().and_then(|m| m.modified()).ok()?;
            Some((mtime, p))
        })
        .collect();
    all.sort_by_key(|x| std::cmp::Reverse(x.0));
    all.into_iter()
        .take(limit)
        .map(|(_, p)| (p.clone(), load(&p).len()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{Message, Role};

    #[test]
    fn archive_append_and_load_roundtrip() {
        let store = SessionStore::new().unwrap();
        let p = store.archive_path();
        assert!(load_archive_file(&p).is_empty()); // none yet
        append_archive(&p, &[Message::user("first batch")]);
        append_archive(&p, &[Message::tool_result("1", "second batch")]);
        let loaded = load_archive_file(&p);
        assert_eq!(loaded.len(), 2); // append-only, both batches survive
        assert_eq!(loaded[0].content, "first batch");
        assert_eq!(loaded[1].content, "second batch");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn save_and_load_roundtrip() {
        let store = SessionStore::new().unwrap();
        let history = vec![
            Message::user("hello"),
            Message {
                role: Role::Assistant,
                content: "hi there".into(),
                tool_calls: vec![],
                tool_call_id: None,
                media: vec![],
            },
        ];
        store.save(&history);
        let loaded = load(&store.path);
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].content, "hello");
        assert_eq!(loaded[1].content, "hi there");
        let _ = std::fs::remove_file(&store.path);
    }
}

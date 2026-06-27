//! Checkpoint / rewind.
//!
//! Before every `write_file` / `edit_file` runs, [`CheckpointStore::snapshot`]
//! records the file's prior content to a per-session JSONL journal under
//! `~/.cache/mge/checkpoints/<session>.jsonl`, so the user can restore it with
//! `mge rewind` (a fresh process; finds the latest journal by mtime) or `/rewind`
//! in the TUI. The journal is created `0600` because it contains verbatim file
//! content (which may include secrets) — rotate credentials if one was edited.
//!
//! **Bash-tool writes are NOT tracked** — we can't know which paths a shell
//! command touches. This is surfaced in `mge rewind` output, not hidden. For
//! bash-heavy work, use git.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Files larger than this are recorded but not snapshotted (journal-size guard).
const MAX_SNAP_BYTES: u64 = 512 * 1024;

/// One recorded file mutation.
#[derive(Debug, Serialize, Deserialize)]
pub struct SnapshotEntry {
    pub seq: u64,
    pub ts_secs: u64,
    pub tool: String,
    pub path: String,
    /// Whether the file existed before the mutation (false → restore = delete).
    pub existed: bool,
    /// Prior UTF-8 content, if captured.
    pub prior: Option<String>,
    /// True if content was deliberately not captured (binary/too-large/unreadable).
    pub skipped: bool,
    pub skip_reason: Option<String>,
}

/// Per-session append-only snapshot journal. Shared by `Arc` across the main
/// agent and its subagents, so every file mutation is recoverable.
pub struct CheckpointStore {
    journal_path: PathBuf,
    seq: AtomicU64,
    lock: Mutex<()>,
}

/// `~/.cache/mge/checkpoints/`.
pub fn checkpoints_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("mge").join("checkpoints"))
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Open (creating if needed) the append-only journal at `0600` on unix — it holds
/// verbatim file content that may include secrets.
fn open_journal(path: &Path) -> std::io::Result<std::fs::File> {
    let mut opts = std::fs::OpenOptions::new();
    opts.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    opts.open(path)
}

impl CheckpointStore {
    /// Create a fresh per-session store (session id = process-start unix seconds).
    pub fn new() -> Result<Self> {
        let dir = checkpoints_dir().context("no cache dir available for checkpoints")?;
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        let journal_path = dir.join(format!("{}.jsonl", now_secs()));
        // Create the journal up front (0600) and surface the error if we can't.
        open_journal(&journal_path)
            .with_context(|| format!("creating checkpoint journal {}", journal_path.display()))?;
        Ok(Self {
            journal_path,
            seq: AtomicU64::new(0),
            lock: Mutex::new(()),
        })
    }

    /// Snapshot the prior state of `path` before `tool` mutates it. Infallible —
    /// logs on error and never panics. The (up to 512 KB) file read runs on the
    /// blocking pool so it can't stall the async runtime mid-stream.
    pub async fn snapshot(&self, tool: &str, path: &str) {
        let seq = self.seq.fetch_add(1, Ordering::SeqCst);
        let ts_secs = now_secs();
        let path_owned = path.to_string();
        let (existed, prior, skipped, skip_reason) =
            tokio::task::spawn_blocking(move || read_prior_state(&path_owned))
                .await
                .unwrap_or((true, None, true, Some("snapshot task failed".into())));
        let entry = SnapshotEntry {
            seq,
            ts_secs,
            tool: tool.to_string(),
            path: path.to_string(),
            existed,
            prior,
            skipped,
            skip_reason,
        };
        let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        if let Ok(line) = serde_json::to_string(&entry) {
            use std::io::Write;
            match open_journal(&self.journal_path) {
                Ok(mut f) => {
                    if let Err(e) = writeln!(f, "{line}") {
                        eprintln!("checkpoint: cannot append journal: {e}");
                    }
                }
                Err(e) => eprintln!("checkpoint: cannot write journal: {e}"),
            }
        }
    }
}

/// Read a file's prior state for a snapshot: `(existed, prior_text, skipped, reason)`.
/// Runs on the blocking pool. Skips files that are too large or binary.
fn read_prior_state(path: &str) -> (bool, Option<String>, bool, Option<String>) {
    let p = Path::new(path);
    match std::fs::metadata(p) {
        Err(_) => (false, None, false, None), // new file → restore = delete
        Ok(meta) if meta.len() > MAX_SNAP_BYTES => (
            true,
            None,
            true,
            Some(format!("too_large ({} bytes)", meta.len())),
        ),
        Ok(_) => match std::fs::read(p) {
            Ok(bytes) if bytes.contains(&0) => (true, None, true, Some("binary".into())),
            Ok(bytes) => match String::from_utf8(bytes) {
                Ok(s) => (true, Some(s), false, None),
                Err(_) => (true, None, true, Some("binary".into())),
            },
            Err(e) => (true, None, true, Some(format!("read error: {e}"))),
        },
    }
}

/// The most-recently-modified session journal (what a fresh `mge rewind` targets).
pub fn find_latest_journal() -> Option<PathBuf> {
    let dir = checkpoints_dir()?;
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

/// Read all entries from a journal, skipping any unparseable lines.
pub fn load_journal(path: &Path) -> Vec<SnapshotEntry> {
    std::fs::read_to_string(path)
        .map(|s| {
            s.lines()
                .filter_map(|l| serde_json::from_str(l).ok())
                .collect()
        })
        .unwrap_or_default()
}

/// Restore one snapshot. Returns a human-readable description of what changed.
pub fn restore(entry: &SnapshotEntry) -> Result<String> {
    if entry.skipped {
        anyhow::bail!(
            "snapshot #{} is not restorable ({})",
            entry.seq,
            entry.skip_reason.as_deref().unwrap_or("skipped")
        );
    }
    let p = Path::new(&entry.path);
    if !entry.existed {
        // The tool created this file → undo = remove it (if still present).
        if p.exists() {
            std::fs::remove_file(p).with_context(|| format!("removing {}", entry.path))?;
            return Ok(format!(
                "removed {} (it did not exist at checkpoint)",
                entry.path
            ));
        }
        return Ok(format!("{} already absent — nothing to do", entry.path));
    }
    let prior = entry.prior.as_deref().unwrap_or("");
    std::fs::write(p, prior).with_context(|| format!("restoring {}", entry.path))?;
    Ok(format!(
        "restored {} to its pre-edit state ({} bytes)",
        entry.path,
        prior.len()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restore_new_file_deletes_it() {
        let dir = std::env::temp_dir().join(format!("mge_ckpt_test_{}", now_secs()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("created.txt");
        std::fs::write(&f, "hello").unwrap();
        let entry = SnapshotEntry {
            seq: 0,
            ts_secs: 0,
            tool: "write_file".into(),
            path: f.to_string_lossy().into(),
            existed: false,
            prior: None,
            skipped: false,
            skip_reason: None,
        };
        restore(&entry).unwrap();
        assert!(
            !f.exists(),
            "restoring a new-file snapshot should delete it"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn restore_edited_file_brings_back_prior() {
        let dir = std::env::temp_dir().join(format!("mge_ckpt_test2_{}", now_secs()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("edited.txt");
        std::fs::write(&f, "NEW CONTENT").unwrap();
        let entry = SnapshotEntry {
            seq: 1,
            ts_secs: 0,
            tool: "edit_file".into(),
            path: f.to_string_lossy().into(),
            existed: true,
            prior: Some("OLD CONTENT".into()),
            skipped: false,
            skip_reason: None,
        };
        restore(&entry).unwrap();
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "OLD CONTENT");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

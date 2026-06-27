//! `@file` mentions.
//!
//! When a user's message references a file with `@<path>` (e.g.
//! "explain @src/main.rs" or "fix @screenshot.png"), [`expand`] resolves each to
//! a real file. **Text files** are read (size-capped) and appended to the message
//! text. **Image files** (png/jpg/gif/webp) are base64-encoded into `data:` URIs
//! and returned separately as `images`, so the caller attaches them as multimodal
//! input and routes the turn to a vision model — the human-visible text is left
//! intact either way.
//!
//! Email-safe: the `@` must start a token (be at the start of the input or
//! preceded by whitespace), so `data@peekage.com` is never treated as a path.

use serde_json::{Value, json};
use std::path::Path;

/// Per-file text cap, and the overall text cap across all mentions in one message.
const PER_FILE_CAP: usize = 20_000;
const TOTAL_CAP: usize = 80_000;
/// Per-media byte cap before base64.
const MAX_MEDIA_BYTES: usize = 16 * 1024 * 1024;
const IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg", "gif", "webp"];
const AUDIO_EXTS: &[&str] = &["mp3", "wav", "m4a", "ogg", "flac", "webm"];

/// Result of expanding `@mentions`: the (possibly augmented) message text, any
/// multimodal content parts (image/audio) to attach, and human-facing notices.
pub struct Expansion {
    pub text: String,
    pub media: Vec<Value>,
    pub notices: Vec<String>,
}

/// Expand `@file` mentions in `input`.
pub fn expand(input: &str) -> Expansion {
    let candidates = find_mentions(input);
    if candidates.is_empty() {
        return Expansion {
            text: input.to_string(),
            media: vec![],
            notices: vec![],
        };
    }

    let mut seen: Vec<String> = Vec::new();
    let mut notices: Vec<String> = Vec::new();
    let mut media: Vec<Value> = Vec::new();
    let mut block = String::new();
    let mut total = 0usize;
    let mut injected_any = false;

    for path in candidates {
        if seen.iter().any(|p| p == &path) {
            continue; // dedupe
        }
        seen.push(path.clone());
        let p = Path::new(&path);
        if !p.is_file() {
            notices.push(format!("@{path}: not found, skipped"));
            continue;
        }
        if is_sensitive_path(p) {
            notices.push(format!("@{path}: blocked (sensitive secret/system path)"));
            continue;
        }

        // Image / audio mention → encode as an OpenAI-compatible content part.
        let ext = p
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase);
        if let Some(ext) = ext.as_deref()
            && (IMAGE_EXTS.contains(&ext) || AUDIO_EXTS.contains(&ext))
        {
            match std::fs::read(p) {
                Ok(b) if b.len() > MAX_MEDIA_BYTES => {
                    notices.push(format!("@{path}: media >16MB, skipped"));
                }
                Ok(b) => {
                    let b64 = crate::util::base64_encode(&b);
                    if IMAGE_EXTS.contains(&ext) {
                        let url = format!("data:{};base64,{b64}", image_mime(ext));
                        media.push(json!({"type":"image_url","image_url":{"url": url}}));
                        notices.push(format!("@{path}: attached as image"));
                    } else {
                        media.push(json!({"type":"input_audio","input_audio":{"data": b64, "format": audio_format(ext)}}));
                        notices.push(format!("@{path}: attached as audio"));
                    }
                }
                Err(e) => notices.push(format!("@{path}: unreadable ({e})")),
            }
            continue;
        }

        // Text-file mention → inject content (capped).
        if total >= TOTAL_CAP {
            notices.push(format!(
                "@{path}: skipped (total @-mention size cap reached)"
            ));
            continue;
        }
        match std::fs::read(p) {
            Err(e) => notices.push(format!("@{path}: unreadable ({e})")),
            Ok(bytes) => match String::from_utf8(bytes) {
                Err(_) => notices.push(format!("@{path}: binary / non-UTF-8, skipped")),
                Ok(text) => {
                    let cap = PER_FILE_CAP.min(TOTAL_CAP - total);
                    let clipped = crate::util::clip(&text, cap);
                    total += clipped.len();
                    block.push_str(&format!("\n### {path}\n```\n{clipped}\n```\n"));
                    injected_any = true;
                    notices.push(format!("@{path}: injected ({} bytes)", clipped.len()));
                }
            },
        }
    }

    let text = if injected_any {
        format!("{input}\n\n--- Referenced files (from @mentions) ---{block}")
    } else {
        input.to_string()
    };
    Expansion {
        text,
        media,
        notices,
    }
}

fn image_mime(ext: &str) -> &'static str {
    match ext {
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        _ => "image/png",
    }
}

fn audio_format(ext: &str) -> &'static str {
    match ext {
        "m4a" => "mp4",
        "wav" => "wav",
        "ogg" => "ogg",
        "flac" => "flac",
        "webm" => "webm",
        _ => "mp3",
    }
}

/// Refuse @-mentions of obviously sensitive locations — the contents are
/// transmitted to a remote provider, so `@~/.config/mge/secrets.env` or
/// `@/proc/self/environ` would exfiltrate credentials. Checks the canonical path
/// (falling back to the literal) against known secret/system roots; project files
/// (including a repo's own `.env`) are intentionally left to the user.
fn is_sensitive_path(p: &Path) -> bool {
    let canon = std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    let s = canon.to_string_lossy();
    if s.starts_with("/proc/")
        || s.starts_with("/sys/")
        || s == "/etc/shadow"
        || s.starts_with("/etc/ssl/private")
    {
        return true;
    }
    if let Some(home) = std::env::var_os("HOME").map(std::path::PathBuf::from) {
        for sub in [
            ".ssh",
            ".config/mge",
            ".aws",
            ".gnupg",
            ".docker",
            ".kube",
            ".npmrc",
            ".netrc",
            ".config/gh",
            ".config/gcloud",
        ] {
            if canon.starts_with(home.join(sub)) {
                return true;
            }
        }
    }
    false
}

/// Find `@<path>` tokens: `@` must be at the start of the input or preceded by
/// whitespace (so emails like `a@b.com` are excluded). The path runs to the next
/// whitespace, minus trailing sentence punctuation.
fn find_mentions(input: &str) -> Vec<String> {
    let bytes = input.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'@' && (i == 0 || bytes[i - 1].is_ascii_whitespace()) {
            let start = i + 1;
            let mut j = start;
            while j < bytes.len() && !bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            if let Some(raw) = input.get(start..j) {
                let path =
                    raw.trim_end_matches([',', '.', ';', ':', '!', '?', ')', ']', '}', '"', '\'']);
                if !path.is_empty() {
                    out.push(path.to_string());
                }
            }
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn email_is_not_a_mention() {
        assert!(find_mentions("contact data@peekage.com about this").is_empty());
    }

    #[test]
    fn finds_paths_and_strips_trailing_punctuation() {
        assert_eq!(
            find_mentions("see @src/main.rs and @Cargo.toml."),
            vec!["src/main.rs", "Cargo.toml"]
        );
    }

    #[test]
    fn expand_injects_text_and_notes_missing() {
        let dir = std::env::temp_dir().join(format!("mge_mention_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("note.txt");
        std::fs::write(&f, "HELLO_MENTION").unwrap();
        let input = format!(
            "look at @{} and @{}/missing.txt",
            f.display(),
            dir.display()
        );
        let e = expand(&input);
        assert!(e.text.contains("HELLO_MENTION"));
        assert!(e.media.is_empty());
        assert!(e.notices.iter().any(|n| n.contains("injected")));
        assert!(e.notices.iter().any(|n| n.contains("not found")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn expand_attaches_image_part_not_text() {
        let dir = std::env::temp_dir().join(format!("mge_mention_img_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("pic.png");
        std::fs::write(&f, [0x89, 0x50, 0x4e, 0x47, 1, 2, 3]).unwrap(); // PNG-ish bytes
        let e = expand(&format!("what is in @{}", f.display()));
        assert_eq!(e.media.len(), 1);
        assert_eq!(e.media[0]["type"], "image_url");
        assert!(
            e.media[0]["image_url"]["url"]
                .as_str()
                .unwrap()
                .starts_with("data:image/png;base64,")
        );
        assert!(
            !e.text.contains("Referenced files"),
            "image must not be injected as text"
        );
        assert!(e.notices.iter().any(|n| n.contains("attached as image")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn expand_attaches_audio_part() {
        let dir = std::env::temp_dir().join(format!("mge_mention_aud_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("clip.mp3");
        std::fs::write(&f, [0x49, 0x44, 0x33, 1, 2, 3]).unwrap();
        let e = expand(&format!("transcribe @{}", f.display()));
        assert_eq!(e.media.len(), 1);
        assert_eq!(e.media[0]["type"], "input_audio");
        assert_eq!(e.media[0]["input_audio"]["format"], "mp3");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_mentions_returns_unchanged() {
        let e = expand("just a normal message");
        assert_eq!(e.text, "just a normal message");
        assert!(e.media.is_empty() && e.notices.is_empty());
    }
}

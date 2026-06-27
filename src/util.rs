//! Small shared helpers.

use std::time::Duration;

/// Max bytes captured from a check command's combined output (keeps context sane).
const CHECK_OUTPUT_CAP: usize = 8_192;

/// Heuristic: does this env var name look like a secret we shouldn't leak into a
/// child process? Shared by the bash tool, check runner, hooks, and LSP spawns.
/// Non-exhaustive on purpose — extend the needle list as new patterns show up.
pub(crate) fn is_secret_env(name: &str) -> bool {
    // $PWD/$OLDPWD contain "PWD" but are not secrets — stripping them from every
    // child would drop the working directory shells and tools rely on.
    if name == "PWD" || name == "OLDPWD" {
        return false;
    }
    let n = name.to_uppercase();
    // Unambiguous markers: substring match (these don't occur in benign names).
    const STRONG: &[&str] = &[
        "SECRET",
        "PASSWORD",
        "PASSWD",
        "TOKEN",
        "CREDENTIAL",
        "BEARER",
        "PRIVATE",
        "APIKEY",
    ];
    if STRONG.iter().any(|s| n.contains(s)) {
        return true;
    }
    // Ambiguous/short markers: match only as a whole `_`/`-`-delimited word, so
    // KEY/PASS/AUTH/URL/URI/DSN catch API_KEY, DB_PASS, GITHUB_AUTH, DATABASE_URL,
    // MONGODB_URI, REDIS_URL, CONN_STRING — but NOT MONKEY, COMPASS, AUTHOR, CURL.
    // Connection-string vars (URL/URI/DSN/CONN) routinely embed user:pass@host.
    const WORD: &[&str] = &[
        "KEY", "PASS", "PWD", "AUTH", "CERT", "URL", "URI", "DSN", "CONN",
    ];
    n.split(['_', '-']).any(|part| WORD.contains(&part))
}

/// Drain an async reader, keeping at most `cap` bytes but continuing to read (and
/// discard) the rest so the child never blocks on a full pipe.
pub(crate) async fn drain_capped<R: tokio::io::AsyncRead + Unpin>(mut r: R, cap: usize) -> Vec<u8> {
    use tokio::io::AsyncReadExt;
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        match r.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                if buf.len() < cap {
                    let take = n.min(cap - buf.len());
                    buf.extend_from_slice(&chunk[..take]);
                }
            }
        }
    }
    buf
}

/// Run a shell check command with a timeout, returning `(passed, output)` where
/// `output` is the combined stdout+stderr, secret-env-scrubbed and clipped to
/// `CHECK_OUTPUT_CAP`. On timeout the child is killed (no orphan) and returns
/// `(false, "[timed out after Ns]")`. Powers the test/lint loop.
pub(crate) async fn run_check_captured(cmd: &str, timeout_secs: u64) -> (bool, String) {
    use std::process::Stdio;
    let mut command = tokio::process::Command::new("sh");
    command.arg("-c").arg(cmd);
    // Strip secret-looking env vars so a check can't exfiltrate keys.
    for (k, _) in std::env::vars() {
        if is_secret_env(&k) {
            command.env_remove(&k);
        }
    }
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    command.kill_on_drop(true); // timeout → future dropped → child reaped, no orphan

    let mut child = match command.spawn() {
        Ok(c) => c,
        Err(e) => return (false, format!("[check failed to start: {e}]")),
    };
    let out = child.stdout.take();
    let err = child.stderr.take();
    let work = async move {
        let (o, e) = match (out, err) {
            (Some(o), Some(e)) => {
                tokio::join!(
                    drain_capped(o, CHECK_OUTPUT_CAP),
                    drain_capped(e, CHECK_OUTPUT_CAP)
                )
            }
            _ => (Vec::new(), Vec::new()),
        };
        let status = child.wait().await;
        (status, o, e)
    };

    match tokio::time::timeout(Duration::from_secs(timeout_secs.max(1)), work).await {
        Err(_) => (false, format!("[timed out after {timeout_secs}s]")),
        Ok((status, o, e)) => {
            let mut combined = String::from_utf8_lossy(&o).into_owned();
            combined.push_str(&String::from_utf8_lossy(&e));
            let passed = status.map(|s| s.success()).unwrap_or(false);
            (passed, clip(combined.trim(), CHECK_OUTPUT_CAP))
        }
    }
}

/// Standard-alphabet base64 encode (for `data:` image URIs). Hand-rolled to
/// avoid a new dependency.
pub(crate) fn base64_encode(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(T[(n >> 18 & 63) as usize] as char);
        out.push(T[(n >> 12 & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            T[(n >> 6 & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            T[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

/// Truncate `s` to at most `max` bytes on a UTF-8 char boundary (never panics),
/// appending an ellipsis when shortened. Use anywhere model/tool text — which
/// may contain multi-byte characters — is clipped for display.
pub fn clip(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_known_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn secret_env_catches_creds_and_connection_strings() {
        for v in [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "DB_PASSWORD",
            "GITHUB_AUTH",
            "DATABASE_URL",
            "REDIS_URL",
            "MONGODB_URI",
            "CONN_STRING",
            "AWS_SECRET_ACCESS_KEY",
        ] {
            assert!(is_secret_env(v), "{v} should be treated as secret");
        }
        // Not secrets — must NOT be over-scrubbed.
        for v in ["PWD", "OLDPWD", "AUTHOR", "COMPASS", "PATH", "HOME", "TERM"] {
            assert!(!is_secret_env(v), "{v} should NOT be treated as secret");
        }
    }

    #[test]
    fn clip_is_char_boundary_safe() {
        // multi-byte chars straddling the cut must not panic.
        let s = "🐐🍦🐐🍦"; // each emoji is 4 bytes
        let out = clip(s, 6); // 6 is mid-emoji
        assert!(out.ends_with('…'));
        assert!(s.starts_with(out.trim_end_matches('…')));
        assert_eq!(clip("short", 99), "short");
    }
}

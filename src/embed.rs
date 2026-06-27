//! Optional semantic retrieval: embed text via an OpenAI-compatible embeddings
//! endpoint, cache the codebase vectors on disk, and cosine-rank for conceptual
//! queries. This is the opt-in SEMANTIC layer ([rag] config); lexical BM25 stays
//! the default so normal/offline operation needs no endpoint and no new runtime.

use crate::config::RagConfig;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Max texts per embeddings request (keeps each HTTP body reasonable).
const BATCH: usize = 64;

/// Cosine similarity of two equal-length vectors (0 if degenerate).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

#[derive(Deserialize)]
struct EmbedResp {
    data: Vec<EmbedItem>,
}
#[derive(Deserialize)]
struct EmbedItem {
    embedding: Vec<f32>,
}

/// Embed `texts` via `<endpoint>/embeddings`, batched. Errors if the endpoint is
/// missing/unreachable so the caller can fall back to lexical.
/// One shared HTTP client (connection/TLS reuse) with real timeouts.
fn http() -> &'static reqwest::Client {
    static CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(15))
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .unwrap_or_else(|e| {
                eprintln!("[mge/embed] warning: HTTP client build failed ({e}); timeouts inactive");
                reqwest::Client::new()
            })
    })
}

pub async fn embed(cfg: &RagConfig, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    let endpoint = cfg
        .endpoint
        .as_deref()
        .context("[rag] endpoint not set")?
        .trim_end_matches('/');
    // The endpoint receives codebase-derived text + the bearer token — refuse a
    // loopback/private host so a misconfig can't exfiltrate to an internal service.
    if crate::tools::is_blocked_host(endpoint) {
        anyhow::bail!(
            "[rag] endpoint points at a loopback/private host — refusing to send code+token"
        );
    }
    let key = std::env::var(&cfg.api_key_env).unwrap_or_default();
    let client = http();

    let mut out: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(BATCH) {
        let body = serde_json::json!({ "model": cfg.model, "input": chunk });
        let mut req = client.post(format!("{endpoint}/embeddings")).json(&body);
        if !key.is_empty() {
            req = req.bearer_auth(&key);
        }
        let resp = req.send().await.context("embeddings request failed")?;
        if !resp.status().is_success() {
            let s = resp.status();
            let t = resp.text().await.unwrap_or_default();
            anyhow::bail!("embeddings HTTP {s}: {}", crate::util::clip(&t, 300));
        }
        let parsed: EmbedResp = resp.json().await.context("bad embeddings response")?;
        // Must be exactly one vector per input, or downstream index-by-position is wrong.
        if parsed.data.len() != chunk.len() {
            anyhow::bail!(
                "embeddings count mismatch: sent {}, got {}",
                chunk.len(),
                parsed.data.len()
            );
        }
        out.extend(parsed.data.into_iter().map(|d| d.embedding));
    }
    Ok(out)
}

#[derive(Serialize, Deserialize)]
struct VectorCache {
    key: u64,
    vectors: Vec<Vec<f32>>,
}

/// FNV-1a over the joined docs — a stable content key for cache invalidation.
fn docs_key(docs: &[String]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for d in docs {
        for b in d.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h ^= b'\n' as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// A SINGLE cache file (overwritten in place) — only one codebase state is ever
/// relevant, and the `key` check rejects stale content, so per-hash files would
/// just accumulate forever.
fn cache_path() -> Option<PathBuf> {
    let dir = directories::ProjectDirs::from("dev", "mge", "mge")?
        .cache_dir()
        .to_path_buf();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        let _ = std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(&dir);
    }
    #[cfg(not(unix))]
    let _ = std::fs::create_dir_all(&dir);
    Some(dir.join("embed-cache.json"))
}

/// Get embeddings for `docs`, using the on-disk cache when the doc set is
/// unchanged (so we only pay the embedding cost when the codebase actually
/// changes). The query is embedded separately by the caller.
pub async fn embed_docs_cached(cfg: &RagConfig, docs: &[String]) -> Result<Vec<Vec<f32>>> {
    let key = docs_key(docs);
    if let Some(path) = cache_path()
        && let Ok(text) = std::fs::read_to_string(&path)
        && let Ok(c) = serde_json::from_str::<VectorCache>(&text)
        && c.key == key
        && c.vectors.len() == docs.len()
    {
        return Ok(c.vectors);
    }
    let vectors = embed(cfg, docs).await?;
    // Only write if serialization succeeds (don't truncate the cache to garbage on
    // a NaN/Inf vector), and 0600 — it derives from your source tree.
    if let Some(path) = cache_path()
        && let Ok(json) = serde_json::to_string(&VectorCache {
            key,
            vectors: vectors.clone(),
        })
    {
        write_private(&path, json.as_bytes());
    }
    Ok(vectors)
}

/// Write a file 0600 on Unix (best-effort).
fn write_private(path: &std::path::Path, data: &[u8]) {
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        // Atomic: write a fresh 0600 temp file, fsync, then rename over the target.
        // Avoids (a) truncate-then-fail leaving a corrupt cache, and (b) inheriting a
        // pre-existing file's 0644 mode (O_CREAT won't re-chmod). Temp name is pid-
        // scoped (no RNG here); rename is atomic on the same filesystem.
        let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
        let opened = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&tmp);
        if let Ok(mut f) = opened {
            if f.write_all(data).is_ok() && f.flush().is_ok() {
                let _ = std::fs::rename(&tmp, path);
            } else {
                let _ = std::fs::remove_file(&tmp);
            }
        }
    }
    #[cfg(not(unix))]
    let _ = std::fs::write(path, data);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_basics() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }

    #[test]
    fn docs_key_is_stable_and_sensitive() {
        let a = vec!["foo".to_string(), "bar".to_string()];
        let b = vec!["foo".to_string(), "baz".to_string()];
        assert_eq!(docs_key(&a), docs_key(&a));
        assert_ne!(docs_key(&a), docs_key(&b));
    }
}

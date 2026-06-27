//! Tiny dependency-free lexical retrieval (BM25) — the shared scorer behind MGE's
//! RAG: query-scoped repo map, delegate context packs, memory-section selection.
//!
//! BM25 over identifier tokens beats embeddings for a coding agent because ~80% of
//! queries name exact symbols (`stream_round`, `Registry`); it's also offline and
//! adds no dependencies. Semantic/vector search is a deferred opt-in, not the base.

/// Split text into lowercased identifier tokens, ALSO splitting `snake_case` and
/// `camelCase` so `maybe_compact` and `maybeCompact` both yield `[maybe, compact]`
/// — a query for either should match the other.
pub fn tokenize(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut prev_is_lower = false;
    for ch in s.chars() {
        if ch.is_alphanumeric() {
            // camelCase boundary: a lowercase/digit run followed by an uppercase.
            if ch.is_uppercase() && prev_is_lower && !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            cur.extend(ch.to_lowercase());
            prev_is_lower = ch.is_lowercase() || ch.is_numeric();
        } else {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            prev_is_lower = false;
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out.retain(|t| t.len() >= 2);
    out
}

/// BM25-rank `docs` against `query`; returns up to `k` doc indices, best first.
/// Docs with zero query-term overlap are excluded (so a poor query returns few
/// or no results rather than padding with irrelevant docs).
pub fn bm25_rank(query: &str, docs: &[String], k: usize) -> Vec<usize> {
    use std::collections::{HashMap, HashSet};
    let q = tokenize(query);
    if q.is_empty() || docs.is_empty() {
        return Vec::new();
    }
    let qset: HashSet<&String> = q.iter().collect();
    let toks: Vec<Vec<String>> = docs.iter().map(|d| tokenize(d)).collect();
    let n = docs.len() as f64;
    let total_len: usize = toks.iter().map(Vec::len).sum();
    let avgdl = (total_len as f64 / n).max(1.0);

    // Document frequency, restricted to query terms.
    let mut df: HashMap<&str, usize> = HashMap::new();
    for t in &toks {
        let seen: HashSet<&str> = t
            .iter()
            .filter(|w| qset.contains(*w))
            .map(String::as_str)
            .collect();
        for w in seen {
            *df.entry(w).or_insert(0) += 1;
        }
    }

    const K1: f64 = 1.5;
    const B: f64 = 0.75;
    let mut scored: Vec<(usize, f64)> = Vec::new();
    for (i, t) in toks.iter().enumerate() {
        let dl = t.len() as f64;
        let mut tf: HashMap<&str, usize> = HashMap::new();
        for w in t {
            if qset.contains(w) {
                *tf.entry(w.as_str()).or_insert(0) += 1;
            }
        }
        if tf.is_empty() {
            continue; // no overlap → not a candidate
        }
        let mut s = 0.0;
        for (term, &f) in &tf {
            let dfi = *df.get(term).unwrap_or(&0) as f64;
            let idf = ((n - dfi + 0.5) / (dfi + 0.5) + 1.0).ln();
            let f = f as f64;
            s += idf * (f * (K1 + 1.0)) / (f + K1 * (1.0 - B + B * dl / avgdl));
        }
        scored.push((i, s));
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_splits_snake_and_camel() {
        assert_eq!(tokenize("maybe_compact"), vec!["maybe", "compact"]);
        assert_eq!(tokenize("maybeCompact"), vec!["maybe", "compact"]);
        assert!(tokenize("a b c").is_empty()); // single chars dropped
        assert_eq!(tokenize("read_file x"), vec!["read", "file"]); // "x" dropped
    }

    #[test]
    fn bm25_ranks_relevant_doc_first() {
        let docs = vec![
            "src/routing.rs: candidates_for, is_retriable, default_route".to_string(),
            "src/checkpoint.rs: CheckpointStore, snapshot, rewind, restore".to_string(),
            "src/tui/mod.rs: App, render, push_diff_lines, spinner".to_string(),
        ];
        let r = bm25_rank("fix the checkpoint rewind bug", &docs, 3);
        assert_eq!(r.first(), Some(&1)); // checkpoint doc ranked top
        // a query with no overlap returns nothing
        assert!(bm25_rank("zzz nonexistent", &docs, 3).is_empty());
    }
}

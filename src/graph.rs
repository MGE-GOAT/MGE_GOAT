//! Code knowledge graph — definition and reference edges across the repo, with
//! neighborhood retrieval ("everything related to X"). This is the graph-RAG
//! layer: given a symbol, return where it's defined AND every file that
//! references it (its callers/users), so the agent navigates a large codebase by
//! structure instead of sweeping greps. Dependency-free (regex + walkdir),
//! reusing the repo-map language patterns and exclude rules.

use crate::config::RepoMapConfig;
use regex::Regex;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;
use walkdir::WalkDir;

/// A resolved code graph: symbol definitions plus reverse references.
pub struct CodeGraph {
    /// symbol → files that DEFINE it.
    defs: BTreeMap<String, Vec<String>>,
    /// symbol → files that REFERENCE it (excluding its own defining file).
    referenced_by: BTreeMap<String, BTreeSet<String>>,
}

impl CodeGraph {
    /// Neighborhood of `symbol`: where it's defined + who references it. Falls
    /// back to substring matches when there's no exact symbol. `None` if unknown.
    pub fn neighborhood(&self, symbol: &str) -> Option<String> {
        if let Some(defined) = self.defs.get(symbol) {
            return Some(self.format_one(symbol, defined));
        }
        let low = symbol.to_lowercase();
        let matches: Vec<&String> = self
            .defs
            .keys()
            .filter(|k| k.to_lowercase().contains(&low))
            .take(8)
            .collect();
        if matches.is_empty() {
            return None;
        }
        let mut out = format!("no exact symbol `{symbol}`; related defined symbols:\n");
        for m in matches {
            out.push_str(&self.format_one(m, &self.defs[m]));
        }
        Some(out)
    }

    fn format_one(&self, symbol: &str, defined: &[String]) -> String {
        let mut out = format!("`{symbol}` defined in: {}\n", defined.join(", "));
        match self.referenced_by.get(symbol) {
            Some(r) if !r.is_empty() => {
                let list: Vec<&str> = r.iter().take(25).map(String::as_str).collect();
                out.push_str(&format!(
                    "  referenced by ({}): {}\n",
                    r.len(),
                    list.join(", ")
                ));
            }
            _ => out.push_str("  referenced by: (no in-repo references found)\n"),
        }
        out
    }
}

/// Build the code graph for `root`. One walk collects defs + identifier use per
/// file; references are resolved against the global symbol set afterward.
pub fn build(root: &Path, cfg: &RepoMapConfig) -> CodeGraph {
    let mut graph = CodeGraph {
        defs: BTreeMap::new(),
        referenced_by: BTreeMap::new(),
    };
    if !cfg.enabled {
        return graph;
    }

    // Pass 1: per file → (path, defined symbols, identifier tokens it uses).
    let mut per_file: Vec<(String, Vec<String>, BTreeSet<String>)> = Vec::new();
    let mut compiled: HashMap<String, Regex> = HashMap::new();
    let walker = WalkDir::new(root).into_iter().filter_entry(|e| {
        let name = e.file_name().to_string_lossy();
        if e.file_type().is_dir() {
            !crate::repo_map::EXCLUDE_DIRS.contains(&name.as_ref())
                && !(name.starts_with('.') && name != ".")
        } else {
            true
        }
    });
    for entry in walker.flatten() {
        if !entry.file_type().is_file() {
            continue;
        }
        let ext = entry
            .path()
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        let Some(pat) = crate::repo_map::lang_pattern(ext) else {
            continue;
        };
        if entry
            .metadata()
            .map(|m| m.len() > crate::repo_map::MAX_FILE_BYTES)
            .unwrap_or(true)
        {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        let re = compiled
            .entry(ext.to_string())
            .or_insert_with(|| Regex::new(pat).expect("BUG: invalid lang_pattern"));
        let mut defs: Vec<String> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for cap in re.captures_iter(&content) {
            if let Some(m) = cap.get(1) {
                let s = m.as_str();
                // O(1) dedup — symbol-dense files made the old Vec::contains O(n²).
                if seen.insert(s.to_string()) {
                    defs.push(s.to_string());
                }
            }
        }
        let rel = entry
            .path()
            .strip_prefix(root)
            .unwrap_or(entry.path())
            .to_string_lossy()
            .replace(['\n', '\r', '\0'], "");
        // Count references from CODE only — strip comments/strings so a symbol
        // named in a comment or a string literal isn't counted as a real caller.
        per_file.push((rel, defs, identifiers(&strip_noise(&content, ext))));
    }

    // Definitions: symbol → defining files.
    for (path, dn, _) in &per_file {
        for s in dn {
            let e = graph.defs.entry(s.clone()).or_default();
            if !e.contains(path) {
                e.push(path.clone());
            }
        }
    }
    let all_defs: BTreeSet<&String> = graph.defs.keys().collect();
    // References: a file references symbol S if S (defined elsewhere) appears in it.
    for (path, own, idents) in &per_file {
        let owns: BTreeSet<&String> = own.iter().collect();
        for tok in idents {
            if all_defs.contains(tok) && !owns.contains(tok) {
                graph
                    .referenced_by
                    .entry(tok.clone())
                    .or_default()
                    .insert(path.clone());
            }
        }
    }
    graph
}

/// Remove comments and string-literal contents so reference detection sees only
/// real code identifiers. Heuristic (no AST) but kills the common false positives
/// (a symbol named in a `// comment` or `"string"`). Handles `//` + `/* */` for
/// C-family, `#` for Python/Ruby/shell/etc, and `"` `'` `` ` `` strings w/ escapes.
fn strip_noise(content: &str, ext: &str) -> String {
    let hash = matches!(ext, "py" | "rb" | "sh" | "bash" | "yaml" | "yml" | "toml");
    let rust = ext == "rs";
    let mut out = String::with_capacity(content.len());
    let mut chars = content.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            // Rust: a `'` may be a LIFETIME (`'a`), not a char literal — keep its
            // identifier chars instead of skipping to a nonexistent closing quote
            // (which would eat the rest of the line, dropping real references).
            '\'' if rust => match chars.peek().copied() {
                Some(n) if n.is_alphabetic() || n == '_' => {
                    chars.next();
                    if chars.peek() == Some(&'\'') {
                        chars.next(); // char literal 'a' → blank
                        out.push(' ');
                    } else {
                        out.push('\''); // lifetime → keep the identifier
                        out.push(n);
                    }
                }
                _ => {
                    // char literal like '\n' / '5' → skip to the closing quote
                    while let Some(d) = chars.next() {
                        if d == '\\' {
                            chars.next();
                            continue;
                        }
                        if d == '\'' || d == '\n' {
                            break;
                        }
                    }
                    out.push(' ');
                }
            },
            '/' if !hash && chars.peek() == Some(&'/') => {
                for d in chars.by_ref() {
                    if d == '\n' {
                        out.push('\n');
                        break;
                    }
                }
            }
            '/' if !hash && chars.peek() == Some(&'*') => {
                chars.next(); // consume '*'
                let mut prev = ' ';
                for d in chars.by_ref() {
                    if prev == '*' && d == '/' {
                        break;
                    }
                    prev = d;
                }
                out.push(' ');
            }
            '#' if hash => {
                for d in chars.by_ref() {
                    if d == '\n' {
                        out.push('\n');
                        break;
                    }
                }
            }
            '"' | '\'' | '`' => {
                let q = c;
                while let Some(d) = chars.next() {
                    if d == '\\' {
                        chars.next(); // skip escaped char
                        continue;
                    }
                    if d == q || d == '\n' {
                        break;
                    }
                }
                out.push(' ');
            }
            _ => out.push(c),
        }
    }
    out
}

/// Whole identifiers (alnum + `_`) as a set, to match against defined symbol names
/// (NOT split on camelCase — symbols are matched verbatim).
fn identifiers(content: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut cur = String::new();
    for ch in content.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            cur.push(ch);
        } else if !cur.is_empty() {
            out.insert(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.insert(cur);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighborhood_links_definition_and_callers() {
        let dir = std::env::temp_dir().join(format!("mge_graph_{}", std::process::id()));
        let src = dir.join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("lib.rs"), "pub fn target_fn() {}\nfn other() {}\n").unwrap();
        std::fs::write(
            src.join("user.rs"),
            "fn caller() { target_fn(); target_fn(); }\n",
        )
        .unwrap();
        // This file MENTIONS target_fn only in a comment and a string — must NOT
        // be counted as a reference (precision: code only).
        std::fs::write(
            src.join("noise.rs"),
            "fn unrelated() {\n    // target_fn is mentioned here\n    let s = \"call target_fn\";\n}\n",
        )
        .unwrap();
        let cfg = RepoMapConfig {
            enabled: true,
            ..RepoMapConfig::default()
        };
        let g = build(&dir, &cfg);
        let n = g.neighborhood("target_fn").expect("known symbol");
        assert!(n.contains("lib.rs")); // defined here
        assert!(n.contains("user.rs")); // real reference
        assert!(!n.contains("noise.rs")); // comment/string mention NOT counted
        assert!(g.neighborhood("nonexistent_zzz").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn strip_noise_keeps_rust_lifetimes() {
        // A lifetime must NOT be treated as a string start (which would eat `bar`).
        let cleaned = strip_noise("fn f<'a>(x: &'a Bar) -> &'a Bar { bar(x) }", "rs");
        assert!(cleaned.contains("Bar"));
        assert!(cleaned.contains("bar"));
        // A real char literal IS blanked.
        assert!(!strip_noise("let c = 'x'; foo();", "rs").contains("'x'"));
    }
}

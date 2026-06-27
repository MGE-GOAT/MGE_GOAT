//! Repo map — cheap whole-codebase orientation injected into the system prompt.
//!
//! Dependency-free (regex + walkdir, both already deps): walk the project, pull
//! top-level symbol *definitions* per file with a per-language regex, rank files
//! by cross-reference density (how often their symbols appear across the repo),
//! and emit a compact `file: sym, sym, …` map. It's orientation, not ground
//! truth — the model still uses read_file/grep/glob for precision. Tree-sitter
//! would add ~6 grammar crates and a minute of build time for ~15% more accuracy;
//! not worth it for a free-first tool (revisit if users hit real gaps).

use crate::config::RepoMapConfig;
use regex::Regex;
use std::collections::HashMap;
use std::path::Path;
use walkdir::WalkDir;

pub(crate) const MAX_FILE_BYTES: u64 = 256 * 1024;
pub(crate) const EXCLUDE_DIRS: &[&str] = &[
    "target",
    "node_modules",
    ".git",
    "dist",
    "build",
    "__pycache__",
    ".next",
    "vendor",
    "coverage",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    "out",
];

/// Per-extension regex (multiline) capturing a defined symbol name in group 1.
pub(crate) fn lang_pattern(ext: &str) -> Option<&'static str> {
    Some(match ext {
        "rs" => {
            r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?(?:fn|struct|enum|trait|impl|type|const|static|macro_rules!|mod)\s+(\w+)"
        }
        "py" => r"(?m)^\s*(?:async\s+)?(?:def|class)\s+(\w+)",
        "ts" | "tsx" | "js" | "jsx" | "mjs" => {
            r"(?m)^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function|class|const|type|interface|enum)\s+(\w+)"
        }
        "go" => r"(?m)^\s*func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)",
        "java" | "kt" | "cs" | "swift" => {
            r"(?m)^\s*(?:[\w@]+\s+)*?(?:class|interface|enum|record|fun|func|def|struct)\s+(\w+)"
        }
        "rb" => r"(?m)^\s*(?:def|class|module)\s+(\w+)",
        "c" | "h" | "cpp" | "hpp" | "cc" => r"(?m)^\s*[\w\*]+\s+(\w+)\s*\(",
        _ => return None,
    })
}

struct FileSymbols {
    path: String,
    symbols: Vec<String>,
}

/// Pull `[A-Za-z0-9_]` identifier tokens, accumulating a global frequency map.
fn tally_tokens(content: &str, freq: &mut HashMap<String, u32>) {
    let mut cur = String::new();
    for ch in content.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            cur.push(ch);
        } else if !cur.is_empty() {
            *freq.entry(std::mem::take(&mut cur)).or_default() += 1;
        }
    }
    if !cur.is_empty() {
        *freq.entry(cur).or_default() += 1;
    }
}

/// A per-file searchable document for the lexical index: a `doc` string
/// (`path: symbol, symbol, …`) that BM25 scores against a query.
pub struct FileDoc {
    pub doc: String,
}

/// Build a query-scoped relevant-files block: BM25-rank the cached index against
/// `query`, emit the top files (path: symbols) within `char_budget`. `None` when
/// the index is empty or nothing matches the query.
pub fn render_scoped(
    index: &[FileDoc],
    query: &str,
    char_budget: usize,
    top: usize,
) -> Option<String> {
    if index.is_empty() {
        return None;
    }
    let docs: Vec<String> = index.iter().map(|d| d.doc.clone()).collect();
    let ranked = crate::rag::bm25_rank(query, &docs, top);
    if ranked.is_empty() {
        return None;
    }
    let header =
        "# Relevant files for this request (orientation only; use read_file/grep for detail)\n";
    let mut out = String::from(header);
    for &i in &ranked {
        let line = format!("{}\n", index[i].doc);
        if out.len() + line.len() > char_budget {
            break;
        }
        out.push_str(&line);
    }
    (out.len() > header.len()).then_some(out)
}

/// Build the per-file index (one [`FileDoc`] per source file). Built once and
/// cached on the agent; cheap to re-query per turn with [`render_scoped`].
pub fn build_index(root: &Path, cfg: &RepoMapConfig) -> Vec<FileDoc> {
    if !cfg.enabled {
        return Vec::new();
    }
    collect_files(root)
        .into_iter()
        .map(|f| {
            let syms = f
                .symbols
                .iter()
                .take(cfg.top_symbols)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            FileDoc {
                doc: format!("{}: {}", f.path, syms),
            }
        })
        .collect()
}

/// Locate where `symbol` is DEFINED in `content`: `(line, character)`, both
/// 0-based (LSP convention), for handing a precise position to a language server.
pub fn symbol_position(content: &str, ext: &str, symbol: &str) -> Option<(u32, u32)> {
    let pat = lang_pattern(ext)?;
    let re = Regex::new(pat).ok()?;
    for (i, line) in content.lines().enumerate() {
        if let Some(m) = re.captures(line).and_then(|c| c.get(1))
            && m.as_str() == symbol
        {
            let ch = line[..m.start()].chars().count() as u32;
            return Some((i as u32, ch));
        }
    }
    None
}

/// Structural index: symbol name → files that DEFINE it (the graph behind the
/// `find_symbol` tool, replacing the expensive `grep 'fn foo'` locate pattern).
pub fn build_symbol_index(
    root: &Path,
    cfg: &RepoMapConfig,
) -> std::collections::BTreeMap<String, Vec<String>> {
    let mut idx: std::collections::BTreeMap<String, Vec<String>> =
        std::collections::BTreeMap::new();
    if !cfg.enabled {
        return idx;
    }
    for f in collect_files(root) {
        for sym in &f.symbols {
            let entry = idx.entry(sym.clone()).or_default();
            if !entry.contains(&f.path) {
                entry.push(f.path.clone());
            }
        }
    }
    idx
}

/// Walk `root`, extracting per-file symbol definitions plus a global identifier
/// frequency map (for cross-reference ranking). Shared by [`build_map`]/[`build_index`].
fn collect_files(root: &Path) -> Vec<FileSymbols> {
    collect_with_freq(root).0
}

fn collect_with_freq(root: &Path) -> (Vec<FileSymbols>, HashMap<String, u32>) {
    let mut files: Vec<FileSymbols> = Vec::new();
    let mut freq: HashMap<String, u32> = HashMap::new();
    let mut compiled: HashMap<String, Regex> = HashMap::new();

    let walker = WalkDir::new(root).into_iter().filter_entry(|e| {
        // Skip excluded directories (and hidden dirs) by name.
        let name = e.file_name().to_string_lossy();
        if e.file_type().is_dir() {
            !EXCLUDE_DIRS.contains(&name.as_ref()) && !(name.starts_with('.') && name != ".")
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
        let Some(pat) = lang_pattern(ext) else {
            continue;
        };
        if entry
            .metadata()
            .map(|m| m.len() > MAX_FILE_BYTES)
            .unwrap_or(true)
        {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        tally_tokens(&content, &mut freq);

        let re = compiled.entry(ext.to_string()).or_insert_with(|| {
            Regex::new(pat).expect("BUG: lang_pattern returned an invalid regex literal")
        });
        let mut symbols: Vec<String> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for cap in re.captures_iter(&content) {
            if let Some(m) = cap.get(1) {
                let s = m.as_str();
                // O(1) dedup — a symbol-dense file made the old Vec::contains O(n²).
                if seen.insert(s.to_string()) {
                    symbols.push(s.to_string());
                }
            }
        }
        if symbols.is_empty() {
            continue;
        }
        let rel = entry.path().strip_prefix(root).unwrap_or(entry.path());
        files.push(FileSymbols {
            // Strip control chars: a filename with an embedded newline could
            // otherwise inject a line into the (trusted) system prompt.
            path: rel
                .to_string_lossy()
                .replace('\\', "/")
                .replace(['\n', '\r', '\0'], ""),
            symbols,
        });
    }

    (files, freq)
}

/// Build the cross-reference-ranked map string for `root` (the query-less view,
/// e.g. `mge map`). Returns `None` when disabled or nothing found.
pub fn build_map(root: &Path, cfg: &RepoMapConfig) -> Option<String> {
    if !cfg.enabled {
        return None;
    }
    let (mut files, freq) = collect_with_freq(root);
    if files.is_empty() {
        return None;
    }

    // Score each file by cross-reference density: how often its defined symbols
    // appear across the whole repo (definition + every reference).
    let score = |f: &FileSymbols| -> u32 {
        f.symbols
            .iter()
            .map(|s| freq.get(s).copied().unwrap_or(0))
            .sum()
    };
    files.sort_by_key(|f| std::cmp::Reverse(score(f)));

    let header = "# Repo map — most cross-referenced files (orientation only; use read_file/grep for detail)\n";
    let mut out = String::from(header);
    for f in files.iter().take(cfg.top_files) {
        let syms = f
            .symbols
            .iter()
            .take(cfg.top_symbols)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        let line = format!("{}: {}\n", f.path, syms);
        if out.len() + line.len() > cfg.char_budget {
            break;
        }
        out.push_str(&line);
    }
    // Nothing made it past the header (e.g. top_files=0) → no useful map.
    if out.len() == header.len() {
        return None;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbol_position_points_at_the_name() {
        let content = "// comment\npub fn is_retriable(e: Error) -> bool {\n    false\n}\n";
        let (line, ch) = symbol_position(content, "rs", "is_retriable").unwrap();
        assert_eq!(line, 1); // second line (0-based)
        assert_eq!(ch, 7); // after "pub fn "
        assert!(symbol_position(content, "rs", "nonexistent").is_none());
    }

    fn cfg() -> RepoMapConfig {
        RepoMapConfig {
            enabled: true,
            char_budget: 16000,
            top_files: 30,
            top_symbols: 15,
        }
    }

    #[test]
    fn extracts_symbols_and_skips_excluded() {
        let dir = std::env::temp_dir().join(format!("mge_repomap_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::create_dir_all(dir.join("target")).unwrap();
        std::fs::write(
            dir.join("src/lib.rs"),
            "pub fn alpha() {}\nstruct Beta;\nfn alpha_helper(){ alpha(); alpha(); }",
        )
        .unwrap();
        std::fs::write(dir.join("target/junk.rs"), "fn should_not_appear() {}").unwrap();
        let map = build_map(&dir, &cfg()).expect("map");
        assert!(map.contains("src/lib.rs"));
        assert!(map.contains("alpha"));
        assert!(map.contains("Beta"));
        assert!(
            !map.contains("should_not_appear"),
            "excluded dir must be skipped"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn zero_top_files_returns_none() {
        let dir = std::env::temp_dir().join(format!("mge_repomap_zero_{}", std::process::id()));
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(dir.join("src/a.rs"), "fn x() {}").unwrap();
        let c = RepoMapConfig {
            top_files: 0,
            ..cfg()
        };
        assert!(
            build_map(&dir, &c).is_none(),
            "header-only map must be None"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn disabled_returns_none() {
        let c = RepoMapConfig {
            enabled: false,
            ..cfg()
        };
        assert!(build_map(Path::new("."), &c).is_none());
    }

    #[test]
    fn empty_dir_returns_none() {
        let dir = std::env::temp_dir().join(format!("mge_repomap_empty_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(build_map(&dir, &cfg()).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}

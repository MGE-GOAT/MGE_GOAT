//! `web_search` tool — free, no-API-key web search for the agent.
//!
//! Backend: DuckDuckGo's HTML endpoint (`html.duckduckgo.com/html/`), parsed
//! with regex. No key, free-first. Works from residential IPs; datacenter/VPN
//! IPs may get a bot-challenge page, in which case the tool returns "no results"
//! rather than erroring. The agent typically follows up with `web_fetch` to read
//! a result. Output is control-char-sanitized and labelled as external data.

use crate::llm::ToolDef;
use crate::tools::Tool;
use anyhow::Result;
use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::OnceLock;
use std::time::Duration;

const MAX_RESULTS: usize = 8;

pub struct WebSearch;

#[async_trait]
impl Tool for WebSearch {
    fn name(&self) -> &str {
        "web_search"
    }

    fn def(&self) -> ToolDef {
        ToolDef {
            name: "web_search".into(),
            description:
                "Search the web (DuckDuckGo) and return result titles, URLs, and snippets. \
                          Use it to find current information, library docs, or examples, then call \
                          web_fetch on a result URL to read the page."
                    .into(),
            parameters: json!({
                "type": "object",
                "properties": { "query": { "type": "string", "description": "The search query." } },
                "required": ["query"]
            }),
        }
    }

    async fn run(&self, args: Value) -> Result<String> {
        let query = args
            .get("query")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim();
        if query.is_empty() {
            anyhow::bail!("web_search: empty query");
        }
        let hits = search_ddg(query).await?;
        if hits.is_empty() {
            return Ok(
                "(no results — DuckDuckGo may have served a bot-challenge page; \
                       try rephrasing, or use web_fetch on a known URL)"
                    .into(),
            );
        }
        let mut out = String::new();
        for (i, h) in hits.iter().enumerate() {
            out.push_str(&format!("{}. {}\n   {}\n", i + 1, h.title, h.url));
            if !h.snippet.is_empty() {
                out.push_str(&format!("   {}\n", h.snippet));
            }
        }
        Ok(out)
    }
}

struct Hit {
    title: String,
    url: String,
    snippet: String,
}

async fn search_ddg(query: &str) -> Result<Vec<Hit>> {
    let url = format!(
        "https://html.duckduckgo.com/html/?q={}",
        percent_encode(query)
    );
    // Reuse one client (connection pool + TLS session reuse) across searches.
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    let client = CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(20))
            .user_agent("Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0")
            .build()
            .expect("web_search: failed to build HTTP client")
    });
    let body = client
        .get(&url)
        .send()
        .await?
        .text()
        .await
        .unwrap_or_default();

    static LINK: OnceLock<regex::Regex> = OnceLock::new();
    static SNIP: OnceLock<regex::Regex> = OnceLock::new();
    let link = LINK.get_or_init(|| {
        regex::Regex::new(r#"(?s)class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>"#).unwrap()
    });
    let snip = SNIP.get_or_init(|| {
        regex::Regex::new(r#"(?s)class="result__snippet"[^>]*>(.*?)</a>"#).unwrap()
    });

    // Collect links and snippets with byte offsets, then attach each snippet to
    // the link it actually follows — positional zip would desync if a result
    // (e.g. an ad) has a link but no snippet.
    let links: Vec<(usize, String, String)> = link
        .captures_iter(&body)
        .filter_map(|c| {
            let end = c.get(0)?.end();
            Some((end, decode_ddg_url(&c[1]), sanitize(&strip_tags(&c[2]))))
        })
        .collect();
    let snips: Vec<(usize, String)> = snip
        .captures_iter(&body)
        .filter_map(|c| Some((c.get(0)?.start(), sanitize(&strip_tags(&c[1])))))
        .collect();

    let mut hits = Vec::new();
    for (i, (link_end, url, title)) in links.iter().enumerate().take(MAX_RESULTS) {
        if title.is_empty() {
            continue;
        }
        let next = links.get(i + 1).map(|l| l.0).unwrap_or(usize::MAX);
        let snippet = snips
            .iter()
            .find(|(pos, _)| pos > link_end && *pos < next)
            .map(|(_, s)| s.clone())
            .unwrap_or_default();
        hits.push(Hit {
            title: title.clone(),
            url: url.clone(),
            snippet,
        });
    }
    Ok(hits)
}

/// Percent-encode a query (RFC 3986 unreserved set passes through).
fn percent_encode(s: &str) -> String {
    let mut o = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            o.push(b as char);
        } else {
            o.push_str(&format!("%{b:02X}"));
        }
    }
    o
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            // Byte-based hex parse — never slice the &str (would panic on a
            // multibyte char boundary after a stray '%').
            b'%' if i + 2 < bytes.len()
                && bytes[i + 1].is_ascii_hexdigit()
                && bytes[i + 2].is_ascii_hexdigit() =>
            {
                let hi = (bytes[i + 1] as char).to_digit(16).unwrap();
                let lo = (bytes[i + 2] as char).to_digit(16).unwrap();
                out.push((hi * 16 + lo) as u8);
                i += 3;
                continue;
            }
            b'+' => out.push(b' '),
            b => out.push(b),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// DDG wraps result links as `//duckduckgo.com/l/?uddg=<encoded-url>&…`.
fn decode_ddg_url(href: &str) -> String {
    if let Some(idx) = href.find("uddg=") {
        let rest = &href[idx + 5..];
        let enc = rest.split('&').next().unwrap_or(rest);
        return percent_decode(enc);
    }
    if let Some(stripped) = href.strip_prefix("//") {
        format!("https://{stripped}")
    } else {
        href.to_string()
    }
}

fn strip_tags(s: &str) -> String {
    let mut o = String::new();
    let mut in_tag = false;
    for ch in s.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            c if !in_tag => o.push(c),
            _ => {}
        }
    }
    o.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#x27;", "'")
        .replace("&#39;", "'")
}

/// Collapse all control chars and whitespace runs to single spaces — keeps each
/// result field strictly single-line so an adversarial snippet can't inject a
/// forged "2. result\n  https://evil…" line into the model's context.
pub(crate) fn sanitize(s: &str) -> String {
    let spaced: String = s
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    spaced.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_and_decodes_roundtrip_ascii() {
        assert_eq!(percent_encode("a b&c"), "a%20b%26c");
        assert_eq!(percent_decode("a%20b%26c"), "a b&c");
    }

    #[test]
    fn decodes_ddg_redirect() {
        let href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fx&rut=abc";
        assert_eq!(decode_ddg_url(href), "https://example.com/x");
    }

    #[test]
    fn strips_tags_and_entities() {
        assert_eq!(strip_tags("a <b>B</b> &amp; c"), "a B & c");
    }

    #[test]
    fn sanitize_collapses_controls_to_single_line() {
        assert_eq!(sanitize("hi\u{7}\u{1b}there"), "hi there");
        assert_eq!(sanitize("forged\n2. evil\n  url"), "forged 2. evil url");
    }
}

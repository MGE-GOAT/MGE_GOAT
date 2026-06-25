//! MGE_GOAT 🐐🍦 — an open-source, GPU-aware agentic coding CLI.
//!
//! This is the early foundation: config + provider abstraction + a streaming
//! `chat` REPL used to validate a real provider before the full TUI and tool
//! loop are layered on.

mod config;
mod llm;
mod theme;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use config::{Config, ProviderConfig};
use futures_util::StreamExt;
use llm::openai_compat::OpenAiCompat;
use llm::{ChatRequest, LlmProvider, Message, StreamEvent};
use std::io::Write;
use std::sync::Arc;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser)]
#[command(name = "mge", version = VERSION, about = "MGE_GOAT — Greatest Of All Tools 🐐🍦")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Print the goat + ice-cream banner.
    Banner,
    /// Write a starter config to ~/.config/mge/config.toml
    Init,
    /// Show resolved config (providers, routes, key presence) for debugging.
    Doctor,
    /// Simple streaming chat REPL against a configured route (no tools yet).
    Chat {
        /// Logical model route to use (defaults to config's default_route).
        #[arg(short, long)]
        route: Option<String>,
    },
}

/// Build a concrete provider from its config entry.
fn build_provider(name: &str, pc: &ProviderConfig) -> Result<Arc<dyn LlmProvider>> {
    let mut client = OpenAiCompat::new(name, pc.base_url.clone(), pc.api_key());
    // OpenRouter ranks/labels traffic by these optional headers.
    if name == "openrouter" {
        client = client
            .with_header("HTTP-Referer", "https://github.com/mge-goat")
            .with_header("X-Title", "MGE_GOAT");
    }
    Ok(Arc::new(client))
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Some(Command::Banner) => {
            print!("{}", theme::banner(VERSION));
            Ok(())
        }
        None => {
            print!("{}", theme::banner(VERSION));
            println!("\nRun `mge init` to create a config, then `mge chat` to talk to a model.");
            Ok(())
        }
        Some(Command::Init) => {
            let path = Config::write_starter()?;
            println!("Wrote starter config to {}", path.display());
            println!("Edit it to add your model ids, then export your API keys.");
            Ok(())
        }
        Some(Command::Doctor) => doctor(),
        Some(Command::Chat { route }) => chat(route).await,
    }
}

/// Print resolved config and whether each provider's key is present.
fn doctor() -> Result<()> {
    let cfg = Config::load()?;
    print!("{}", theme::banner(VERSION));
    println!("config path: {}", Config::default_path()?.display());
    println!("default route: {}\n", cfg.default_route);

    println!("Providers:");
    for (name, pc) in &cfg.providers {
        let key = if pc.api_key().is_some() {
            "key: present"
        } else if pc.api_key_env.eq_ignore_ascii_case("none") || pc.api_key_env.is_empty() {
            "no key needed"
        } else {
            "key: MISSING"
        };
        let loc = if pc.local { " [local]" } else { "" };
        println!("  - {name}{loc}: {} ({key})", pc.base_url);
    }

    println!("\nModel routes:");
    if cfg.models.is_empty() {
        println!("  (none configured — add [models.*] entries in your config)");
    }
    for (name, mr) in &cfg.models {
        let model = if mr.model.is_empty() { "<unset>" } else { mr.model.as_str() };
        println!("  - {name} -> provider '{}', model '{}'", mr.provider, model);
    }
    Ok(())
}

/// Minimal streaming REPL to verify a provider/route works end to end.
async fn chat(route: Option<String>) -> Result<()> {
    let cfg = Config::load()?;
    let route = route.unwrap_or_else(|| cfg.default_route.clone());
    let (pc, mr) = cfg
        .resolve(&route)
        .with_context(|| format!("resolving route '{route}'"))?;

    if mr.model.trim().is_empty() {
        bail!(
            "route '{route}' has no model id set. Edit {} and add a model.",
            Config::default_path()?.display()
        );
    }
    if pc.api_key().is_none() && !pc.local && !pc.api_key_env.eq_ignore_ascii_case("none") {
        bail!(
            "provider '{}' needs env var {} but it is not set.",
            mr.provider,
            pc.api_key_env
        );
    }

    let provider = build_provider(&mr.provider, pc)?;
    print!("{}", theme::banner(VERSION));
    println!(
        "Chatting via route '{route}' (provider '{}', model '{}'). Ctrl-D to exit.\n",
        mr.provider, mr.model
    );

    let mut history: Vec<Message> = vec![Message::system(
        "You are MGE_GOAT, a helpful, concise coding assistant.",
    )];

    let stdin = std::io::stdin();
    loop {
        print!("{} you ▸ {}", theme::ansi::SKY, theme::ansi::RESET);
        std::io::stdout().flush().ok();

        let mut line = String::new();
        let n = stdin.read_line(&mut line)?;
        if n == 0 {
            println!("\n{}bye 🐐{}", theme::ansi::PISTACHIO, theme::ansi::RESET);
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        history.push(Message::user(line));
        let req = ChatRequest::new(mr.model.as_str(), history.clone());

        print!("{}{} mge ▸ {}", theme::ansi::STRAWBERRY, theme::MARK, theme::ansi::RESET);
        std::io::stdout().flush().ok();

        let mut stream = provider.stream_chat(req).await?;
        let mut reply = String::new();
        while let Some(ev) = stream.next().await {
            match ev? {
                StreamEvent::TextDelta(t) => {
                    reply.push_str(&t);
                    print!("{t}");
                    std::io::stdout().flush().ok();
                }
                StreamEvent::ToolCallDelta { .. } => { /* tools come in a later milestone */ }
                StreamEvent::Done { .. } => break,
            }
        }
        println!();
        history.push(Message::assistant(reply));
    }
    Ok(())
}

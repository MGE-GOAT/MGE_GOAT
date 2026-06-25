//! MGE_GOAT visual identity: goat 🐐 + ice cream 🍦 ASCII art and a small color palette.
//!
//! Kept dependency-light: colors are raw ANSI escapes so this module works in both
//! the simple line REPL and (later) inside the ratatui TUI banner.

/// ANSI color helpers. We keep these as associated constants rather than a crate
/// so the banner renders identically regardless of which frontend prints it.
pub mod ansi {
    pub const RESET: &str = "\x1b[0m";
    pub const BOLD: &str = "\x1b[1m";
    pub const DIM: &str = "\x1b[2m";

    // Ice-cream palette: strawberry, vanilla, pistachio, cone-brown, sky.
    pub const STRAWBERRY: &str = "\x1b[38;5;211m";
    pub const VANILLA: &str = "\x1b[38;5;230m";
    pub const PISTACHIO: &str = "\x1b[38;5;151m";
    pub const CONE: &str = "\x1b[38;5;179m";
    pub const SKY: &str = "\x1b[38;5;117m";
    pub const GOAT_GREY: &str = "\x1b[38;5;250m";
}

/// The headline goat. A proud G.O.A.T. holding an ice cream cone.
pub const GOAT: &str = r#"
                       ___
                  ,= ,-_-. =,
                 ((_/)o o(\_))
                  `-'(. .)`-'        MGE_GOAT
                      \_/            the Greatest Of All Tools
"#;

/// A scoop-on-a-cone, printed beside or below the goat.
pub const ICE_CREAM: &str = r#"
        (  )
       (    )
      ( **** )       a coding agent that runs on
       \****/         your GPU and free APIs
        \**/
         \/
         /\
        /  \
       /____\
"#;

/// Compact one-line mark for prompts / status lines.
pub const MARK: &str = "🐐🍦 mge";

/// Render the full startup banner with color, given a version string.
pub fn banner(version: &str) -> String {
    use ansi::*;
    let mut out = String::new();
    out.push_str(PISTACHIO);
    out.push_str(GOAT);
    out.push_str(RESET);
    out.push_str(STRAWBERRY);
    out.push_str(ICE_CREAM);
    out.push_str(RESET);
    out.push_str(&format!(
        "{DIM}        MGE_GOAT v{version} — Maximum Greatness Engine{RESET}\n",
    ));
    out
}

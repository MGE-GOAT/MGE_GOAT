//! MGE_GOAT visual identity: the goat 🐐 + melting ice cream 🍦 scene, an ANSI
//! palette for the splash, and the animation frames shared by the splash and the
//! ratatui TUI side-panel.

/// ANSI color helpers for the (non-ratatui) splash/REPL output.
pub mod ansi {
    pub const RESET: &str = "\x1b[0m";
    pub const BOLD: &str = "\x1b[1m";
    pub const DIM: &str = "\x1b[2m";

    // App palette: blue / gray / pink / white (+ pistachio for "settled").
    pub const STRAWBERRY: &str = "\x1b[38;5;211m"; // pink
    pub const PISTACHIO: &str = "\x1b[38;5;151m";
    pub const SKY: &str = "\x1b[38;5;117m"; // blue
    pub const GOAT_GREY: &str = "\x1b[38;5;250m";
    pub const DIFF_ADD: &str = "\x1b[38;5;114m"; // green (+ lines)
    pub const DIFF_DEL: &str = "\x1b[38;5;210m"; // red (− lines)
}

/// Compact one-line mark for prompts / status lines.
pub const MARK: &str = "🐐🍦 mge";

/// The goat scene: backward-curving horns, a face with eyes, a chin beard, and a
/// melting ice-cream cone on the right. ASCII-only so byte offset == column,
/// which keeps the `set_char` overlay logic simple and UTF-8-safe.
const SCENE_RAW: &str = r#"  /\_/\         .-"-.
 / o   \,      / o o \
( ==  Y  )~~~ (   T   )
 \   ^_/        \ ___ /
  )     (       (_____)
 (   |   )       )   (
  \  |  /       ( ' , )
   '-W-'         \   /
                  \ /
                   V"#;

/// Column where the goat (left) gives way to the ice cream (right), for colour.
pub const SCENE_SPLIT: usize = 14;

fn set_char(lines: &mut [String], row: usize, col: usize, ch: char) {
    if row >= lines.len() {
        return;
    }
    let line = &mut lines[row];
    while line.len() <= col {
        line.push(' ');
    }
    // SAFETY: SCENE is ASCII and we only ever write ASCII glyphs, so byte index
    // == column and we never split a multibyte char.
    unsafe { line.as_bytes_mut()[col] = ch as u8 };
}

/// Number of rows in the goat scene.
pub fn scene_height() -> usize {
    SCENE_RAW.split('\n').count()
}

/// The clean, un-animated goat scene (no sweat/blink overlays). Used when the
/// agent is idle so the mascot looks crisp rather than noisy.
pub fn scene_base() -> Vec<String> {
    SCENE_RAW.split('\n').map(|s| s.to_string()).collect()
}

/// Build animation frame `i` as plain (uncoloured) lines. Shared by the splash
/// and the TUI side-panel; callers apply their own colour. Pass `i` advancing
/// over time to animate; a fixed `i` renders a still frame.
pub fn scene_frame(i: usize) -> Vec<String> {
    let mut lines: Vec<String> = SCENE_RAW.split('\n').map(|s| s.to_string()).collect();

    // The goat blinks its open eye now and then (try-hard squint).
    if i.is_multiple_of(8) {
        set_char(&mut lines, 4, 6, '-');
    }

    // Sweat droplets flicker by the horns and to the right of the scoop.
    let g = ['.', '\'', ',', '*'];
    for (k, &(r, c)) in [(2usize, 2usize), (0, 4), (4, 25), (6, 24), (8, 22)]
        .iter()
        .enumerate()
    {
        if (i + k) % 4 < 2 {
            set_char(&mut lines, r, c, g[(i + k) % g.len()]);
        }
    }

    // A melt-drop oozes off the cone and falls (rows 8..=11, in the ice-cream
    // colour zone so it renders pink).
    let drop_row = 8 + (i % 4);
    set_char(
        &mut lines,
        drop_row,
        18,
        if drop_row >= 11 { ':' } else { 'o' },
    );

    lines
}

/// Colour a set of scene lines: goat side grey, ice-cream side pink.
fn scene_colored(lines: &[String]) -> String {
    use ansi::*;
    let mut out = String::new();
    for line in lines {
        let split = SCENE_SPLIT.min(line.len());
        let (left, right) = line.split_at(split);
        out.push_str(&format!("{GOAT_GREY}{left}{STRAWBERRY}{right}{RESET}\n"));
    }
    out
}

/// Render animation frame `i` coloured for the ANSI splash.
fn frame(i: usize) -> String {
    scene_colored(&scene_frame(i))
}

/// Coloured one-shot banner (used by `doctor` and the line REPL header).
pub fn banner(version: &str) -> String {
    use ansi::*;
    let mut out = scene_colored(&scene_base());
    out.push_str(&format!(
        "{DIM}        MGE_GOAT v{version} — the Greatest Of All Tools 🐐🍦{RESET}\n",
    ));
    out
}

/// Play the splash animation in place, then leave the version tagline. Falls back
/// to a single static frame when stdout is not a TTY (e.g. piped/CI).
pub fn play_splash(version: &str) {
    use ansi::*;
    use std::io::{IsTerminal, Write};

    let mut out = std::io::stdout();
    let height = scene_height();

    if !out.is_terminal() {
        print!("{}", frame(0));
        println!("{DIM}        MGE_GOAT v{version} 🐐🍦 — the Greatest Of All Tools{RESET}");
        return;
    }

    const FRAMES: usize = 28;
    for i in 0..FRAMES {
        print!("{}", frame(i));
        out.flush().ok();
        std::thread::sleep(std::time::Duration::from_millis(110));
        if i + 1 < FRAMES {
            print!("\x1b[{height}A"); // move cursor up to redraw in place
        }
    }
    println!(
        "{PISTACHIO}{BOLD}        M G E · G O A T{RESET}  {DIM}v{version} — the Greatest Of All Tools 🐐🍦{RESET}"
    );
}

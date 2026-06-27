//! Pixel-art mascot: an evil goat licking a cute crying ice cream, drawn as
//! Unicode upper-half-block (`▀`) cells — fg = top pixel, bg = bottom pixel —
//! giving two truecolor pixels per character cell.
//!
//! `scene_frame(thinking, tick)` produces a continuous animation: the ice cream
//! always melts, both blink and bob; idle = goat chuckles + ice cream worried &
//! glancing at the goat; thinking = goat licks (tongue in/out) + ice cream cries
//! & sweats. Ported from the Python prototype in scratchpad/art.
//!
//! Grids are addressed by [row][col] coordinates, so column-range loops that
//! mutate cells are intentional (not iterator-convertible).
#![allow(clippy::needless_range_loop)]

use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};

// ── base sprites ─────────────────────────────────────────────────────────────
const GOAT: &[&str] = &[
    "     d        d     ",
    "     gd      dg     ",
    "      dd    dd      ",
    "      gd    dg      ",
    "      dd    dd      ",
    "    dgddggggddgd    ",
    " dg dgGGGGGGGGgd gd ",
    "dg dgGGGGGGGGGGgd gd",
    "O  dgGddGGGGddGgd   ",
    "   dgGYYYGGYYYGgd   ", // 9  eyes top
    "   dgGkkkGGkkkGgd   ", // 10 slit
    "   dgGYYYGGYYYGgd   ", // 11 eyes bottom
    "    dgGGGGGGGGgd    ",
    "    dgGGGwwGGGgd    ",
    "     dgkWWWWkgd     ", // 14 mouth top
    "     dgrWrrWrgd     ", // 15 mouth mid
    "      dgkrrkgd      ", // 16 mouth bottom
    "       dgGGgd       ",
    "        dBBd        ",
];

const ICE: &[&str] = &[
    "   SSSS   ",
    "  SSSSSS  ",
    "  SWSSSS  ",
    "  ssSSss  ",
    "  UUUUUU  ",
    " UWWUWWU  ", // 5 eye sclera
    " UWkUWkU  ", // 6 pupils
    " UUUUUUU  ",
    " UUkkUUU  ", // 8 mouth
    "  uuuuu   ",
    "  cCCCc   ",
    "   CcC    ",
    "    C     ",
];

/// Palette: sprite char -> RGB. Unknown / space => transparent (None).
fn palette(ch: char) -> Option<Color> {
    let (r, g, b) = match ch {
        'o' => (58, 63, 77),
        'G' => (210, 215, 224),
        'g' => (144, 151, 166),
        'd' => (107, 113, 128),
        'Y' => (255, 225, 70),
        'k' => (30, 30, 36),
        'B' => (238, 241, 247),
        'T' => (255, 143, 176),
        'S' => (245, 106, 138),
        's' => (212, 74, 108),
        'p' => (235, 95, 125),
        'W' => (255, 255, 255),
        'U' => (125, 114, 224),
        'u' => (91, 80, 192),
        'C' => (224, 168, 102),
        'c' => (176, 122, 58),
        'L' => (169, 208, 255),
        'r' => (190, 40, 58),
        'O' => (240, 195, 40),
        _ => return None,
    };
    Some(Color::Rgb(r, g, b))
}

type Grid = Vec<Vec<char>>;

fn lol(rows: &[&str]) -> Grid {
    let w = rows.iter().map(|r| r.chars().count()).max().unwrap_or(0);
    rows.iter()
        .map(|r| {
            let mut v: Vec<char> = r.chars().collect();
            v.resize(w, ' ');
            v
        })
        .collect()
}

fn put(g: &mut Grid, r: usize, c: usize, ch: char) {
    if r < g.len() && c < g[r].len() {
        g[r][c] = ch;
    }
}

// ── per-frame face states ────────────────────────────────────────────────────
fn goat_frame(thinking: bool, t: usize) -> Grid {
    let mut g = lol(GOAT);
    let blink = matches!(t % 14, 12 | 13);
    if blink {
        for c in 6..14 {
            for r in [9usize, 10, 11] {
                if "YkG".contains(g[r][c]) {
                    g[r][c] = if r == 10 { 'd' } else { 'g' };
                }
            }
        }
    }
    if !thinking {
        // chuckling: mouth opens/closes
        if t % 4 < 2 {
            put(&mut g, 14, 8, 'k');
            put(&mut g, 14, 11, 'k');
            put(&mut g, 16, 8, 'r');
            put(&mut g, 16, 11, 'r');
        } else {
            put(&mut g, 15, 7, 'g');
            put(&mut g, 15, 12, 'g');
        }
    } else {
        // happy/content while licking
        put(&mut g, 15, 8, 'W');
        put(&mut g, 15, 11, 'W');
    }
    g
}

fn ice_frame(thinking: bool, t: usize) -> Grid {
    let mut g = lol(ICE);
    let blink = matches!(t % 16, 14 | 15);
    if blink {
        for (r, c) in [
            (5, 2),
            (5, 3),
            (5, 5),
            (5, 6),
            (6, 2),
            (6, 3),
            (6, 5),
            (6, 6),
        ] {
            put(&mut g, r, c, 'u');
        }
        for c in [2usize, 3, 5, 6] {
            put(&mut g, 6, c, 'k');
        }
    } else if !thinking {
        // worried, glancing LEFT toward the goat
        put(&mut g, 6, 2, 'k');
        put(&mut g, 6, 3, 'W');
        put(&mut g, 6, 5, 'k');
        put(&mut g, 6, 6, 'W');
    }
    // mouth
    if !thinking {
        put(&mut g, 8, 3, 'U');
        put(&mut g, 8, 4, 'k');
    } else if t % 4 < 2 {
        put(&mut g, 8, 3, 'k');
        put(&mut g, 8, 4, 'k');
    } else {
        put(&mut g, 8, 3, 'U');
        put(&mut g, 8, 4, 'k');
    }
    g
}

/// Shear a grid so its top leans right (ice cream held at an angle).
fn tilt(grid: Grid, shear: f32) -> Grid {
    let h = grid.len();
    let mut out: Grid = Vec::with_capacity(h);
    for (y, row) in grid.into_iter().enumerate() {
        let shift = (((h - 1 - y) as f32) * shear).round() as usize;
        let mut newrow = vec![' '; shift];
        newrow.extend(row);
        out.push(newrow);
    }
    let w = out.iter().map(|r| r.len()).max().unwrap_or(0);
    for r in &mut out {
        r.resize(w, ' ');
    }
    out
}

const CW: usize = 36;
const CH: usize = 32;

fn stamp(canvas: &mut Grid, grid: &Grid, ox: usize, oy: usize) {
    for (y, row) in grid.iter().enumerate() {
        for (x, &ch) in row.iter().enumerate() {
            if ch != ' ' && oy + y < CH && ox + x < CW {
                canvas[oy + y][ox + x] = ch;
            }
        }
    }
}

fn set(canvas: &mut Grid, r: usize, c: usize, ch: char) {
    if r < CH && c < CW {
        canvas[r][c] = ch;
    }
}

/// Build the full mascot scene for the given mode and animation tick.
fn scene_frame(thinking: bool, t: usize) -> Grid {
    let goat = goat_frame(thinking, t);
    let ice = tilt(ice_frame(thinking, t), 0.28);
    let bob = if (t / 3).is_multiple_of(2) { 0 } else { 1 };

    let mut canvas: Grid = vec![vec![' '; CW]; CH];
    stamp(&mut canvas, &goat, 0, 2 + bob);
    let (ix, iy) = (16usize, 16 + bob);
    stamp(&mut canvas, &ice, ix, iy);

    // tongue licks in/out (thinking): emerges from the mouth, reaches the cone
    if thinking && (t % 4) < 2 {
        for (x, y) in [
            (9, 18),
            (10, 18),
            (11, 19),
            (12, 19),
            (13, 20),
            (14, 20),
            (15, 21),
            (16, 21),
            (17, 22),
            (18, 22),
            (19, 23),
        ] {
            set(&mut canvas, y + bob, x, 'T');
        }
    }

    // melt drip — ALWAYS
    let dy = t % 6;
    set(
        &mut canvas,
        iy + 9 + dy,
        ix + 4,
        if dy < 4 { 'p' } else { 's' },
    );

    // tears (thinking)
    if thinking {
        set(&mut canvas, iy + 8 + (t % 5), ix + 3, 'L');
        set(&mut canvas, iy + 7 + ((t + 2) % 5), ix + 7, 'L');
    }

    // sweat: slight idle, more when thinking
    if !thinking {
        if t % 6 < 3 {
            set(&mut canvas, iy + 2, ix + 8, 'W');
        }
    } else {
        set(&mut canvas, iy + 1, ix + 8, 'W');
        if t.is_multiple_of(3) {
            set(&mut canvas, iy + 3, ix + 9, 'W');
        }
    }
    canvas
}

/// Number of character rows the rendered mascot occupies (half of pixel height).
pub fn rows() -> u16 {
    (CH / 2) as u16
}

/// Render the mascot for `(thinking, tick)` as half-block lines. Transparent
/// pixels use `Color::Reset` so the sprite blends with the panel background.
pub fn render(thinking: bool, tick: usize) -> Vec<Line<'static>> {
    let canvas = scene_frame(thinking, tick);
    // Left margin so the mascot sits a little right of the panel border.
    const MARGIN: usize = 2;
    let mut lines = Vec::with_capacity(CH / 2);
    let mut r = 0;
    while r + 1 < CH {
        let (top, bot) = (&canvas[r], &canvas[r + 1]);
        let mut spans = Vec::with_capacity(CW + MARGIN);
        if MARGIN > 0 {
            spans.push(Span::raw(" ".repeat(MARGIN)));
        }
        for c in 0..CW {
            let tc = palette(top[c]);
            let bc = palette(bot[c]);
            match (tc, bc) {
                (None, None) => spans.push(Span::raw(" ")),
                (t_, b_) => spans.push(Span::styled(
                    "▀",
                    Style::default()
                        .fg(t_.unwrap_or(Color::Reset))
                        .bg(b_.unwrap_or(Color::Reset)),
                )),
            }
        }
        lines.push(Line::from(spans));
        r += 2;
    }
    lines
}

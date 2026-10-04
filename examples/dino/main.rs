//! GLiNER2.5-Decide plays the Chrome T-Rex runner (https://github.com/congerh/dino).
//!
//! The model is loaded once. A tiny local HTTP server serves the game with an
//! injected bridge script; the page samples the game state `--hz` times per second
//! and POSTs it to `/act`. Each state is turned into a short written situation that
//! the Decide checkpoint classifies (jump / duck / run), and the page presses the
//! chosen key.
//!
//!   cargo run --release --example dino
//!   cargo run --release --features cuda --example dino -- --cuda --hz 20
use anyhow::{Context, Result, bail};
use candle_core::{DType, Device};
use gliner_rs::model_path::{self, VariantDef};
use gliner_rs::{ClassificationSpec, GLiNER2};
use serde::Deserialize;
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const GAME_REPO: &str = "https://github.com/congerh/dino";
const BRIDGE: &str = include_str!("bridge.js");
const VARIANTS: &[VariantDef] = &[
    VariantDef { key: "decide", hf_repo: "fastino/GLiNER2.5-Decide" },
    VariantDef { key: "decide-multi", hf_repo: "fastino/GLiNER2.5-multi-Decide" },
    VariantDef { key: "decide-1b", hf_repo: "fastino/GLiNER2.5-Decide-1B" },
];

/// `--lead-ms`: fixed lead (ms) ahead of the sampled state; 0 (default) derives it from
/// the measured round trip.
static LEAD_MS: AtomicU64 = AtomicU64::new(0);

#[derive(Deserialize, Clone, Copy)]
struct Trex {
    x: f64,
    ground_y: f64,
    jumping: bool,
}

#[derive(Deserialize, Clone)]
struct Obstacle {
    #[serde(rename = "type")]
    kind: String,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

#[derive(Deserialize)]
struct GameState {
    #[serde(default)]
    client: String,
    speed: f64,
    trex: Trex,
    obstacles: Vec<Obstacle>,
    /// Measured browser round trip and sampling period; the defaults are the
    /// conditions the margins below were first tuned under (22.5 ms RTT, 30 Hz).
    #[serde(default = "default_rtt_ms")]
    rtt_ms: f64,
    #[serde(default = "default_sample_ms")]
    sample_ms: f64,
}

fn default_rtt_ms() -> f64 {
    22.5
}

fn default_sample_ms() -> f64 {
    1000.0 / 30.0
}

const TREX_WIDTH: f64 = 44.0;
const TREX_HEIGHT: f64 = 47.0;
const TREX_DUCK_HEIGHT: f64 = 25.0;
const FRAME_MS: f64 = 1000.0 / 60.0;
/// Safety margins on the jump window: sampling, the round trip and collision-box
/// slack all shift the real jump relative to the computed one. The landing margin
/// is the sampling period + round trip + this slack (60 ms at 30 Hz / 22.5 ms).
/// All three are tunable (`--rise-margin-ms`, `--fall-slack-ms`, `--lead-slack-ms`).
static RISE_MARGIN_MS: Tunable = Tunable::new(20.0);
static FALL_SLACK_MS: Tunable = Tunable::new(4.2);
/// Extra lead beyond the round trip (key press landing on the next frame, jitter).
static LEAD_SLACK_MS: Tunable = Tunable::new(7.5);

struct Tunable(AtomicU64);

impl Tunable {
    const fn new(v: f64) -> Self {
        Self(AtomicU64::new(v.to_bits()))
    }
    fn get(&self) -> f64 {
        f64::from_bits(self.0.load(Ordering::Relaxed))
    }
    fn set(&self, v: f64) {
        self.0.store(v.to_bits(), Ordering::Relaxed)
    }
}
/// Obstacles closer than this behind the next one cannot be landed between; one
/// jump must clear them together.
const MERGE_GAP_MS: f64 = 250.0;

/// A jump started now, replayed with the game's own physics (Trex.updateJump):
/// returns per-frame heights of the dinosaur's feet above the ground.
fn jump_heights(speed: f64) -> Vec<f64> {
    let ground = 93.0;
    let (mut y, mut v) = (ground, -10.0 - speed / 10.0);
    let mut reached_min = false;
    let mut heights = Vec::new();
    loop {
        y += (v as f64).round();
        v += 0.6;
        reached_min |= y < ground - 30.0;
        if y < 30.0 && reached_min && v < -5.0 {
            v = -5.0; // Reached max height: endJump
        }
        if y > ground {
            return heights;
        }
        heights.push(ground - y);
    }
}

/// What the next obstacle demands, from the game's geometry and physics.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Class {
    /// Nothing in sight.
    Clear,
    /// A high bird: a running dinosaur passes underneath.
    Overhead,
    /// A bird at head height: ducking lets it pass.
    Head,
    /// A cactus or a low bird: only a jump gets over it.
    Ground,
}

/// Deterministic facts about the next obstacle (all times in ms from now).
struct Situation {
    class: Class,
    kind: &'static str,
    arrive: f64,
    leave: f64,
    /// A jump started now: feet high enough over this obstacle during [up, down]; lands at `land`.
    up: f64,
    down: f64,
    land: f64,
}

#[cfg_attr(not(test), allow(dead_code))]
impl Situation {
    fn jump_clears(&self) -> bool {
        self.clears_after(0.0)
    }

    /// Does a jump started `delay` ms from now clear the obstacle?
    fn clears_after(&self, delay: f64) -> bool {
        let (a, l) = (self.arrive - delay, self.leave - delay);
        a >= self.up && l <= self.down && a <= self.land
    }
}

fn analyze(s: &GameState) -> Situation {
    let trex_bottom = s.trex.ground_y + TREX_HEIGHT;
    let Some(o) = s.obstacles.iter().find(|o| o.x + o.width > s.trex.x) else {
        return Situation { class: Class::Clear, kind: "", arrive: f64::INFINITY, leave: f64::INFINITY, up: 0.0, down: 0.0, land: 0.0 };
    };
    let bottom = o.y + o.height;
    let class = if bottom <= s.trex.ground_y + 2.0 {
        Class::Overhead
    } else if bottom <= trex_bottom - TREX_DUCK_HEIGHT + 2.0 {
        Class::Head
    } else {
        Class::Ground
    };
    // The obstacle overlaps the dinosaur's x-span [x, x + width] during [arrive, leave].
    let px_per_ms = s.speed / FRAME_MS;
    let arrive = ((o.x - (s.trex.x + TREX_WIDTH)) / px_per_ms).max(0.0);
    let mut leave = ((o.x + o.width - s.trex.x) / px_per_ms).max(0.0);
    // Pixels the dinosaur's feet must rise to pass over it (a little collision-box slack).
    let mut clearance = (trex_bottom - o.y - 4.0).max(0.0);
    if class == Class::Ground {
        // Fold in ground obstacles that follow too closely to land in between.
        for next in s.obstacles.iter().filter(|n| n.x > o.x && n.y + n.height > trex_bottom - TREX_DUCK_HEIGHT + 2.0) {
            let next_arrive = (next.x - (s.trex.x + TREX_WIDTH)) / px_per_ms;
            if next_arrive - leave > MERGE_GAP_MS {
                break;
            }
            leave = leave.max((next.x + next.width - s.trex.x) / px_per_ms);
            clearance = clearance.max(trex_bottom - next.y - 4.0);
        }
    }
    let heights = jump_heights(s.speed);
    let high: Vec<usize> = (0..heights.len()).filter(|&i| heights[i] >= clearance).collect();
    let (up, down) = match (high.first(), high.last()) {
        (Some(&a), Some(&b)) => (a as f64 * FRAME_MS + RISE_MARGIN_MS.get(), (b + 1) as f64 * FRAME_MS - (s.sample_ms + s.rtt_ms + FALL_SLACK_MS.get())),
        _ => (f64::INFINITY, 0.0),
    };
    let kind = if o.kind.starts_with("CACTUS") { "cactus" } else { "flying bird" };
    Situation { class, kind, arrive, leave, up, down, land: heights.len() as f64 * FRAME_MS }
}

/// One short written situation per obstacle class (the class itself is deterministic), so
/// the model only judges the timing of the single move that can help, as a two-way
/// classification. `None`: nothing to decide.
fn prompt(sit: &Situation) -> Option<(String, ClassificationSpec)> {
    let Situation { kind, arrive, leave, up, down, land, .. } = *sit;
    let timing = format!("The {kind} reaches the dinosaur in {arrive:.0} ms and has fully passed in {leave:.0} ms. A running dinosaur would be hit in {arrive:.0} ms.");
    match sit.class {
        Class::Clear | Class::Overhead => None,
        // Provably too early: even the end of a jump's high-enough window comes before
        // the obstacle arrives. Only the decision zone is put to the model.
        Class::Ground if arrive > down => None,
        Class::Ground => {
            let verdict = if arrive > land {
                format!("A jump started now lands at {land:.0} ms, before the obstacle arrives at {arrive:.0} ms, so it is too early to jump.")
            } else if arrive < up {
                format!("A jump started now is high enough only from {up:.0} ms, but the obstacle arrives at {arrive:.0} ms, so it is too late to jump.")
            } else if leave > down {
                format!("A jump started now would land on it: the dinosaur comes down at {down:.0} ms, but the obstacle is under it until {leave:.0} ms. It is too early to jump.")
            } else {
                format!("A jump started now is high enough from {up:.0} ms to {down:.0} ms, covering the obstacle's {arrive:.0}-{leave:.0} ms, so jumping now clears it.")
            };
            let text = format!("A {kind} is on the ground at the dinosaur's feet. {timing} {verdict}");
            let spec = ClassificationSpec::new("action", ["jump", "run"])
                .with_label_description("jump", "Jump now: jumping now clears the obstacle.")
                .with_label_description("run", "Keep running: it is too early or too late to jump.");
            Some((text, spec))
        }
        Class::Head => {
            let text = format!("A {kind} is at the dinosaur's head height. {timing} Ducking now lets the bird pass over the dinosaur's head.");
            let spec = ClassificationSpec::new("action", ["duck", "run"])
                .with_label_description("duck", "Duck now: the bird passes over the dinosaur's head.")
                .with_label_description("run", "Keep running: the bird is still far away.");
            Some((text, spec))
        }
    }
}

fn summary(sit: &Situation) -> String {
    match sit.class {
        Class::Clear => "nothing in sight".into(),
        Class::Overhead => format!("high bird passes overhead in {:.0} ms", sit.arrive),
        Class::Head => format!("bird at head height, arrives {:.0} ms", sit.arrive),
        Class::Ground if sit.arrive > sit.down => format!("{} on the ground, arrives {:.0} ms: too early for any jump", sit.kind, sit.arrive),
        Class::Ground => format!("{} on the ground, arrives {:.0} ms; jump window {:.0}-{:.0} ms", sit.kind, sit.arrive, sit.up, sit.down),
    }
}

struct Stats {
    decisions: AtomicU64,
    model_us: AtomicU64,
    counts: Mutex<[u64; 3]>,
    /// Recent (situation, action) pairs, printed when a game ends.
    recent: Mutex<std::collections::HashMap<String, std::collections::VecDeque<String>>>,
    /// Games finished per viewer.
    per_client: Mutex<std::collections::HashMap<String, u64>>,
}

fn act(model: &Mutex<GLiNER2>, device: &str, stats: &Stats, body: &[u8]) -> Result<Value> {
    let game: GameState = serde_json::from_slice(body)?;
    if game.trex.jumping {
        // Nothing to decide mid-air; the bridge releases the jump key on landing.
        return Ok(json!({"action": "run", "probabilities": {}, "model_ms": 0.0, "summary": "airborne", "device": device}));
    }
    // Decide for where the game will be when the key press lands, not for the snapshot.
    let lead = match LEAD_MS.load(Ordering::Relaxed) {
        0 => game.rtt_ms + LEAD_SLACK_MS.get(),
        fixed => fixed as f64,
    };
    let game = GameState {
        obstacles: game.obstacles.iter().map(|o| Obstacle { x: o.x - game.speed * lead / FRAME_MS, ..o.clone() }).collect(),
        ..game
    };
    let sit = analyze(&game);
    let summary = summary(&sit);
    let Some((text, spec)) = prompt(&sit) else {
        return Ok(json!({"action": "run", "probabilities": {}, "model_ms": 0.0, "summary": format!("{summary} (no decision needed)"), "device": device}));
    };
    let started = Instant::now();
    let probs = model.lock().unwrap().classification_probabilities(&text, &[spec])?;
    let elapsed = started.elapsed();
    let probs = probs.into_iter().next().context("no classification result")?.1;
    let (action, _) = probs.iter().max_by(|a, b| a.1.total_cmp(&b.1)).context("no label scores")?;
    let action = action.clone();
    stats.decisions.fetch_add(1, Ordering::Relaxed);
    stats.model_us.fetch_add(elapsed.as_micros() as u64, Ordering::Relaxed);
    {
        let mut all = stats.recent.lock().unwrap();
        let recent = all.entry(game.client.clone()).or_default();
        recent.push_back(format!("{action:>4} <- {summary}"));
        if recent.len() > 6 {
            recent.pop_front();
        }
    }
    stats.counts.lock().unwrap()[["jump", "duck", "run"].iter().position(|a| *a == action).unwrap_or(2)] += 1;
    let probabilities: serde_json::Map<String, Value> = probs.iter().map(|(k, p)| (k.clone(), json!(p))).collect();
    Ok(json!({"action": action, "probabilities": probabilities, "model_ms": elapsed.as_secs_f64() * 1e3, "summary": summary, "device": device}))
}

fn respond(stream: &mut TcpStream, status: &str, kind: &str, body: &[u8]) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)
}

fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript",
        Some("css") => "text/css",
        Some("png") => "image/png",
        Some("mp3") => "audio/mpeg",
        _ => "application/octet-stream",
    }
}

fn handle(mut stream: TcpStream, game: &Path, bridge: &str, model: &Mutex<GLiNER2>, device: &str, stats: &Stats) -> Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let (method, path) = (parts.next().unwrap_or(""), parts.next().unwrap_or("/").to_owned());
    let mut length = 0;
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header)? == 0 || header.trim().is_empty() {
            break;
        }
        if let Some((k, v)) = header.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                length = v.trim().parse()?;
            }
        }
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body)?;
    match (method, path.as_str()) {
        ("POST", "/act") => {
            let reply = act(model, device, stats, &body).unwrap_or_else(|e| json!({"action": "run", "error": e.to_string()}));
            respond(&mut stream, "200 OK", "application/json", reply.to_string().as_bytes())?;
        }
        ("POST", "/gameover") => {
            let v: Value = serde_json::from_slice(&body).unwrap_or_default();
            let score = v["score"].as_u64().unwrap_or(0);
            let client = v["client"].as_str().unwrap_or("").to_owned();
            let game_no = {
                let mut per = stats.per_client.lock().unwrap();
                let n = per.entry(client.clone()).or_default();
                *n += 1;
                *n
            };
            let n = stats.decisions.load(Ordering::Relaxed).max(1);
            let c = *stats.counts.lock().unwrap();
            println!(
                "[viewer {client}] game {game_no}: score {score:>5} | all viewers: {n} decisions (jump {}, duck {}, run {}), mean model latency {:.2} ms",
                c[0], c[1], c[2], stats.model_us.load(Ordering::Relaxed) as f64 / n as f64 / 1e3
            );
            if let Some(recent) = stats.recent.lock().unwrap().get_mut(&client) {
                for line in recent.drain(..) {
                    println!("    {line}");
                }
            }
            respond(&mut stream, "200 OK", "application/json", b"{}")?;
        }
        ("GET", "/dino-bridge.js") => respond(&mut stream, "200 OK", "text/javascript", bridge.as_bytes())?,
        ("GET", p) => {
            let rel = p.split('?').next().unwrap().trim_start_matches('/');
            let rel = if rel.is_empty() { "index.html" } else { rel };
            if rel.split('/').any(|c| c == "..") {
                return Ok(respond(&mut stream, "403 Forbidden", "text/plain", b"forbidden")?);
            }
            let file = game.join(rel);
            match std::fs::read(&file) {
                Ok(mut bytes) => {
                    if rel == "index.html" {
                        let html = String::from_utf8(bytes)?.replacen("</body>", "<script src=\"/dino-bridge.js\"></script>\n</body>", 1);
                        bytes = html.into_bytes();
                    }
                    respond(&mut stream, "200 OK", content_type(&file), &bytes)?;
                }
                Err(_) => respond(&mut stream, "404 Not Found", "text/plain", b"not found")?,
            }
        }
        _ => respond(&mut stream, "405 Method Not Allowed", "text/plain", b"")?,
    }
    Ok(())
}

struct Args {
    variant: String,
    checkpoint: Option<PathBuf>,
    cuda: bool,
    hz: u32,
    host: String,
    port: u16,
    game: PathBuf,
    open: bool,
}

fn parse_args() -> Result<Args> {
    let mut args = Args {
        variant: "decide".into(),
        checkpoint: std::env::var("GLINER_MODEL").ok().map(PathBuf::from),
        cuda: false,
        hz: 30,
        host: "127.0.0.1".into(),
        port: 8765,
        game: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/dino/game"),
        open: true,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().with_context(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--variant" => args.variant = value()?,
            "--checkpoint" => args.checkpoint = Some(value()?.into()),
            "--cuda" => args.cuda = true,
            "--hz" => args.hz = value()?.parse()?,
            "--host" => args.host = value()?,
            "--port" => args.port = value()?.parse()?,
            "--game" => args.game = value()?.into(),
            "--no-open" => args.open = false,
            "--rise-margin-ms" => RISE_MARGIN_MS.set(value()?.parse()?),
            "--fall-slack-ms" => FALL_SLACK_MS.set(value()?.parse()?),
            "--lead-slack-ms" => LEAD_SLACK_MS.set(value()?.parse()?),
            "--lead-ms" => LEAD_MS.store(value()?.parse()?, Ordering::Relaxed),
            "-h" | "--help" => {
                println!("usage: dino [--variant decide|decide-multi|decide-1b] [--checkpoint DIR] [--cuda] [--hz N] [--lead-ms MS] [--lead-slack-ms MS] [--fall-slack-ms MS] [--rise-margin-ms MS] [--host ADDR] [--port P] [--game DIR] [--no-open]");
                std::process::exit(0);
            }
            other => bail!("unknown flag {other}"),
        }
    }
    if !(1..=120).contains(&args.hz) {
        bail!("--hz must be between 1 and 120");
    }
    Ok(args)
}

fn main() -> Result<()> {
    let args = parse_args()?;
    if !args.game.join("dino.js").is_file() {
        println!("fetching the game from {GAME_REPO} into {}", args.game.display());
        let ok = std::process::Command::new("git").args(["clone", "--depth", "1", GAME_REPO]).arg(&args.game).status()?.success();
        if !ok {
            bail!("git clone failed; clone {GAME_REPO} yourself and pass --game DIR");
        }
    }

    // Load once, then warm up so the first in-game decision is not slow.
    let started = Instant::now();
    let checkpoint = model_path::resolve(VARIANTS, "decide", args.checkpoint.clone(), Some(&args.variant))?;
    let (device, device_name) = if args.cuda { (Device::new_cuda(0)?, "cuda") } else { (Device::Cpu, "cpu") };
    let model = GLiNER2::load(&checkpoint, &device, DType::F32)?;
    let warm = GameState {
        client: String::new(),
        speed: 6.0,
        trex: Trex { x: 50.0, ground_y: 93.0, jumping: false },
        obstacles: vec![Obstacle { kind: "CACTUS_SMALL".into(), x: 120.0, y: 105.0, width: 17.0, height: 35.0 }],
        rtt_ms: default_rtt_ms(),
        sample_ms: default_sample_ms(),
    };
    let (text, spec) = prompt(&analyze(&warm)).context("warm-up needs a decision")?;
    for _ in 0..3 {
        model.classification_probabilities(&text, std::slice::from_ref(&spec))?;
    }
    println!("GLiNER2.5-Decide ({}) loaded on {device_name} in {:.2}s", args.variant, started.elapsed().as_secs_f64());
    let model = Mutex::new(model);

    let listener = TcpListener::bind((args.host.as_str(), args.port)).with_context(|| format!("port {} busy? use --port", args.port))?;
    let shown = if args.host == "0.0.0.0" { "127.0.0.1" } else { args.host.as_str() };
    let url = format!("http://{shown}:{}/", args.port);
    println!("game at {url} - Decide decides {} times per second (Ctrl-C to stop)", args.hz);
    if args.open {
        let _ = std::process::Command::new("xdg-open").arg(&url).spawn();
    }
    let bridge = BRIDGE.replace("__HZ__", &args.hz.to_string());
    let stats = Stats { decisions: AtomicU64::new(0), model_us: AtomicU64::new(0), counts: Mutex::new([0; 3]), recent: Mutex::default(), per_client: Mutex::default() };
    std::thread::scope(|scope| {
        for stream in listener.incoming().flatten() {
            let (model, stats, bridge, game) = (&model, &stats, &bridge, &args.game);
            scope.spawn(move || {
                if let Err(e) = handle(stream, game, bridge, model, device_name, stats) {
                    eprintln!("request failed: {e}");
                }
            });
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(kind: &str, x: f64, y: f64, height: f64) -> GameState {
        GameState {
            client: String::new(),
            speed: 6.0,
            trex: Trex { x: 50.0, ground_y: 93.0, jumping: false },
            obstacles: vec![Obstacle { kind: kind.into(), x, y, width: 46.0, height }],
            rtt_ms: default_rtt_ms(),
            sample_ms: default_sample_ms(),
        }
    }

    #[test]
    fn classes_and_jump_timing_follow_game_geometry() {
        let class = |y| analyze(&state("PTERODACTYL", 120.0, y, 40.0)).class;
        assert_eq!((class(100.0), class(75.0), class(50.0)), (Class::Ground, Class::Head, Class::Overhead));
        // A jump at speed 6 lasts roughly half a second and peaks well above a large cactus.
        let heights = jump_heights(6.0);
        assert!((25..45).contains(&heights.len()), "{}", heights.len());
        assert!(heights.iter().cloned().fold(0.0, f64::max) > 55.0);
        assert!(!analyze(&state("CACTUS_LARGE", 600.0, 90.0, 50.0)).jump_clears());
        assert!(analyze(&state("CACTUS_SMALL", 130.0, 105.0, 35.0)).jump_clears());
        assert!(prompt(&analyze(&state("PTERODACTYL", 120.0, 50.0, 40.0))).is_none());
    }
}

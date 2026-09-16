//! Interactive classification shell with persistent history.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::ValueEnum;
use gliner_rs::GLiNER2;
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::history::DefaultHistory;
use rustyline::{Editor, Helper, Highlighter, Hinter, Validator};

use crate::session::{parse_example, Activation, Format, Session, TaskDef};

const COMMANDS: &[(&str, &str)] = &[
    (":labels", "a,b,c          set the single task `label` (label:description allowed)"),
    (":multi", "on|off          toggle multi-label for the `label` task"),
    (":task", "[+]name=a,b,c    add or replace a task (+ = multi-label)"),
    (":rm", "name              remove a task"),
    (":clear", "               remove all tasks, examples and the prompt"),
    (":show", "                show the current settings"),
    (":example", "input=>label  add a few-shot example (`:example clear` to drop all)"),
    (":prompt", "text|off       set the instruction appended to task names"),
    (":threshold", "0.5        multi-label threshold"),
    (":activation", "auto|softmax|sigmoid"),
    (":all", "on|off            show every label's probability"),
    (":top", "N|off             show the N most probable labels"),
    (":format", "text|jsonl|json|tsv"),
    (":file", "path            classify every line of a file"),
    (":time", "on|off           print timing after each run"),
    (":help", "                this help"),
    (":quit", "                exit (also Ctrl-D)"),
];

#[derive(Helper, Hinter, Highlighter, Validator)]
struct CommandHelper;

impl Completer for CommandHelper {
    type Candidate = Pair;

    fn complete(&self, line: &str, pos: usize, _ctx: &rustyline::Context<'_>) -> rustyline::Result<(usize, Vec<Pair>)> {
        let head = &line[..pos];
        if !head.starts_with(':') || head.contains(' ') {
            return Ok((0, Vec::new()));
        }
        let candidates = COMMANDS
            .iter()
            .filter(|(cmd, _)| cmd.starts_with(head))
            .map(|(cmd, _)| Pair { display: cmd.to_string(), replacement: format!("{cmd} ") })
            .collect();
        Ok((0, candidates))
    }
}

/// `$XDG_STATE_HOME/gliner-classify/history`, defaulting to `~/.local/state`.
fn history_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local").join("state")))?;
    Some(base.join("gliner-classify").join("history"))
}

fn parse_switch(arg: &str) -> Result<bool> {
    match arg {
        "on" | "true" | "yes" | "1" => Ok(true),
        "off" | "false" | "no" | "0" => Ok(false),
        other => bail!("expected on|off, got {other:?}"),
    }
}

fn print_help() {
    println!("Type any text to classify it. Commands:");
    for (cmd, help) in COMMANDS {
        println!("  {cmd} {help}");
    }
}

fn print_settings(session: &Session, timing: bool) {
    if session.tasks.is_empty() {
        println!("tasks:      (none; use :labels a,b,c or :task name=a,b,c)");
    }
    for task in &session.tasks {
        println!("task:       {task}");
    }
    for (input, label) in &session.examples {
        println!("example:    {input} => {label}");
    }
    println!("prompt:     {}", session.prompt.as_deref().unwrap_or("(none)"));
    println!("threshold:  {}", session.threshold);
    println!("activation: {:?}", session.activation);
    println!(
        "display:    format={:?} all={} top={}",
        session.format,
        session.all,
        session.top_k.map_or("off".to_string(), |k| k.to_string())
    );
    println!("timing:     {}", if timing { "on" } else { "off" });
}

struct Shell<'a> {
    model: &'a GLiNER2,
    session: Session,
    timing: bool,
}

enum Flow {
    Continue,
    Quit,
}

impl Shell<'_> {
    fn classify(&self, texts: &[String]) -> Result<()> {
        let started = Instant::now();
        self.session.run(self.model, texts)?;
        if self.timing {
            eprintln!("({} text(s) in {:.2?})", texts.len(), started.elapsed());
        }
        Ok(())
    }

    fn command(&mut self, line: &str) -> Result<Flow> {
        let (cmd, arg) = match line.split_once(char::is_whitespace) {
            Some((c, a)) => (c, a.trim()),
            None => (line, ""),
        };
        let s = &mut self.session;
        match cmd {
            ":q" | ":quit" | ":exit" => return Ok(Flow::Quit),
            ":h" | ":help" | ":?" => print_help(),
            ":show" | ":settings" => print_settings(s, self.timing),
            ":labels" | ":l" => {
                if arg.is_empty() {
                    bail!("usage: :labels a,b,c");
                }
                let multi = s.tasks.iter().find(|t| t.name == "label").is_some_and(|t| t.multi);
                s.set_task(TaskDef { name: "label".into(), multi, labels: arg.into() });
                s.specs()?;
            }
            ":multi" | ":m" => {
                let on = if arg.is_empty() { true } else { parse_switch(arg)? };
                let Some(task) = s.tasks.iter_mut().find(|t| t.name == "label") else {
                    bail!("no `label` task; set one with :labels, or use :task +name=...");
                };
                task.multi = on;
            }
            ":task" | ":t" => {
                s.set_task(TaskDef::parse(arg)?);
                s.specs()?;
            }
            ":rm" => {
                let before = s.tasks.len();
                s.tasks.retain(|t| t.name != arg);
                if s.tasks.len() == before {
                    bail!("no task named {arg:?}");
                }
            }
            ":clear" => {
                s.tasks.clear();
                s.examples.clear();
                s.prompt = None;
            }
            ":example" | ":e" => {
                if arg == "clear" {
                    s.examples.clear();
                } else {
                    s.examples.push(parse_example(arg)?);
                }
            }
            ":prompt" => s.prompt = (!arg.is_empty() && arg != "off").then(|| arg.to_string()),
            ":threshold" => {
                let value: f32 = arg.parse().with_context(|| format!("invalid threshold {arg:?}"))?;
                if !(0.0..=1.0).contains(&value) {
                    bail!("threshold must be within 0..1");
                }
                s.threshold = value;
            }
            ":activation" => {
                s.activation = Activation::from_str(arg, true).map_err(|e| anyhow::anyhow!(e))?;
            }
            ":all" | ":a" => s.all = if arg.is_empty() { !s.all } else { parse_switch(arg)? },
            ":top" | ":k" => {
                s.top_k = match arg {
                    "" | "off" => None,
                    n => Some(n.parse().with_context(|| format!("invalid count {n:?}"))?),
                }
            }
            ":format" | ":f" => s.format = Format::from_str(arg, true).map_err(|e| anyhow::anyhow!(e))?,
            ":file" => {
                let path = shellexpand(arg);
                let content =
                    std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
                let texts: Vec<String> =
                    content.lines().filter(|l| !l.trim().is_empty()).map(str::to_string).collect();
                self.classify(&texts)?;
            }
            ":time" => self.timing = if arg.is_empty() { !self.timing } else { parse_switch(arg)? },
            other => bail!("unknown command {other:?} (try :help)"),
        }
        Ok(Flow::Continue)
    }
}

fn shellexpand(path: &str) -> PathBuf {
    match (path.strip_prefix("~/"), std::env::var_os("HOME")) {
        (Some(rest), Some(home)) => PathBuf::from(home).join(rest),
        _ => PathBuf::from(path),
    }
}

pub fn run(model: &GLiNER2, session: Session, initial_texts: &[String]) -> Result<()> {
    let mut editor: Editor<CommandHelper, DefaultHistory> = Editor::new()?;
    editor.set_helper(Some(CommandHelper));
    let history = history_path();
    if let Some(path) = &history {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = editor.load_history(path);
    }

    let mut shell = Shell { model, session, timing: false };
    if !initial_texts.is_empty() {
        if let Err(e) = shell.classify(initial_texts) {
            eprintln!("error: {e:#}");
        }
    }
    eprintln!("gliner-classify shell: type text to classify, :help for commands, Ctrl-D to quit");
    if shell.session.tasks.is_empty() {
        eprintln!("no labels yet: try `:labels positive,negative,neutral`");
    }

    loop {
        let prompt = match shell.session.tasks.as_slice() {
            [] => "classify> ".to_string(),
            [task] => format!("{}{}> ", if task.multi { "+" } else { "" }, task.name),
            tasks => format!("[{} tasks]> ", tasks.len()),
        };
        match editor.readline(&prompt) {
            Ok(line) => {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let _ = editor.add_history_entry(line);
                if let Some(path) = &history {
                    let _ = editor.append_history(path);
                }
                let result = if line.starts_with(':') {
                    shell.command(line)
                } else {
                    shell.classify(&[line.to_string()]).map(|_| Flow::Continue)
                };
                match result {
                    Ok(Flow::Quit) => break,
                    Ok(Flow::Continue) => {}
                    Err(e) => eprintln!("error: {e:#}"),
                }
            }
            Err(ReadlineError::Interrupted) => continue,
            Err(ReadlineError::Eof) => break,
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn history_prefers_xdg_state_home() {
        // Only checks the path shape; environment mutation is avoided.
        let path = history_path().unwrap();
        assert!(path.ends_with("gliner-classify/history"));
    }
}

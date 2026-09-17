# gliner-rs

Rust inference for GLiNER2 **boundary-architecture** checkpoints, built on
[candle](https://github.com/huggingface/candle). It is a port of the inference path of
[fastino-ai/GLiNER2](https://github.com/fastino-ai/GLiNER2), and works with any checkpoint
in that family — `fastino/gliner2.5-multi-v1` (default), `gliner2.5-small-v1`,
`gliner2.5-base-v1`, `gliner2-privacy-filter-PII-multi`, `gliguard-LLMGuardrails-300M`,
`GLiNER2-Guardrails-PII-Multi`, and other fine-tunes on the same architecture.

Supported:

- Entity extraction: labels, descriptions, per-label thresholds, `list`/`str` dtypes, and the
  abstention head. Overlap policies are `flat` (default), `nested`, `allow` and `longest`.
- Text classification: single- or multi-label, label descriptions, few-shot examples.
- Relation extraction: typed capped pair proposal, the biaffine relation scorer, and edge
  deduplication.
- Structured extraction (`extract_json`, or `Schema::structure` for the full builder):
  the record head in `natural` mode (a field is the anchor — the first by default, or
  pick one with `StructureSpec::anchor`; each record keeps its own values), plus `latent`
  mode (no declared anchor: the head's own learned selector chooses which mention seeds
  each instance) and `anchorless` mode (document-conditioned learned instance queries,
  for records not seeded by any single span). Per-field `cardinality`
  (`optional_one`/`required_one`/`zero_or_more`/`one_or_more`, `FieldSpec::cardinality`)
  and `exclusive` (`FieldSpec::exclusive`) refine how mentions bind to fields; `str`/`list`
  fields, descriptions, thresholds, and choice fields (`status::[shipped|pending]`,
  including literal-mention binding and prefix scoring) all work in every mode.
  `StructureMode::Legacy` gives the aggregate decoder instead of the record head.
- Regex validators (`RegexValidator`, `EntitySpec::validator` / `FieldSpec::validator`):
  a post-filter on an extracted span's surface text, full or partial match, optionally
  inverted.
- Entity attributes (`AttributeGroup`, `Schema::entity_attributes`): extra typed properties
  force-scored at an already-extracted entity's exact span (e.g. a `color` or `sentiment`
  attribute on a `product` entity), single- or multi-label per group, with an optional
  `applies_to` restriction and `qualify_labels` to disambiguate label text shared across
  groups.
- Any mix of the above in one schema, which runs as a single encoder pass.
- Whitespace and character-level (CJK) word splitters.
- Long documents: `extract_entities_long`, `extract_relations_long`, `extract_json_long`,
  `classify_text_long`, and generic `extract_long` split the text into overlapping word
  chunks, run each chunk through the normal single-pass API, remap every span back to
  character offsets in the original document, and merge duplicate predictions across
  overlaps under the same `flat`/`nested`/`allow`/`longest` overlap policies.
- Batching: `extract_batch`, `extract_entities_batch`, `extract_relations_batch`,
  `extract_json_batch`, `classify_text_batch`, and `classification_probabilities_batch` run
  several texts against the *same* schema in one padded encoder pass instead of one call per
  text; each text's result is identical to what the single-text method would give it alone.
  `gliner-classify` uses this for `-f`/multi-text input.
- CPU, or CUDA with `--features cuda`. The encoder can run in F32 or F16; the heads always
  run in F32.

## Installation

macOS (Apple Silicon) and Linux, via Homebrew:

```sh
brew tap apiplant/tap
brew install apiplant/tap/gliner-rs
```

Arch Linux, via the signed pacman repository at `apiplant.github.io/pacman`
(one-time setup, then `pacman -Sy`/`-Syu` picks up new releases):

```sh
curl -sSfL https://apiplant.github.io/pacman/apiplant.gpg -o /tmp/apiplant.gpg
keyid=$(gpg --show-keys --with-colons /tmp/apiplant.gpg | awk -F: '/^pub:/ { print $5; exit }') && sudo pacman-key --add /tmp/apiplant.gpg && sudo pacman-key --finger "$keyid" && sudo pacman-key --lsign-key "$keyid"
printf '\n[apiplant]\nSigLevel = Required DatabaseOptional\nServer = https://apiplant.github.io/pacman/$arch\n' | sudo tee -a /etc/pacman.conf > /dev/null
sudo pacman -Sy gliner-rs
```

Debian/Ubuntu, via the signed apt repository at `apt.apiplant.com` (one-time
setup, then `apt upgrade` picks up new releases):

```sh
curl -sSfL https://apt.apiplant.com/apiplant-archive-keyring.gpg | sudo tee /usr/share/keyrings/apiplant.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/apiplant.gpg] https://apt.apiplant.com stable main" | sudo tee /etc/apt/sources.list.d/apiplant.list > /dev/null
sudo apt update && sudo apt install gliner-rs
```

Or download the archive, `.deb`, or `.pkg.tar.zst` for your platform from the
[releases page](https://github.com/apiplant/gliner-rs/releases) and install
it directly — the plain archive needs no installation at all, all four
binaries are static enough to run from anywhere. On Linux x86_64 with an
NVIDIA GPU, grab the `gliner-rs-cuda-*-x86_64-unknown-linux-gnu.tar.gz`
archive instead for CUDA-accelerated inference (needs a host driver
compatible with the CUDA 12.6 toolkit it was built against, and pass
`--cuda` to the CLIs to use it).

As a Rust library, or to build the CLIs from source, via crates.io:

```sh
cargo add gliner-rs         # as a library dependency
cargo install gliner-rs     # for the gliner/gliner-classify/gliner-pii/gliner-guardrails binaries
cargo install gliner-rs --features cuda  # with CUDA support
```

| Platform | Ships as |
| --- | --- |
| macOS (Apple Silicon) | archive, Homebrew |
| Linux x86_64 | archive, `.deb` + apt repo, Arch package + pacman repo, Homebrew |
| Linux x86_64, CUDA | archive only |
| Linux aarch64 | archive, `.deb` + apt repo, Homebrew |

No macOS Intel build: only Apple Silicon (`aarch64-apple-darwin`) and Linux
(`x86_64`/`aarch64`) are supported.

See [`packaging/README.md`](packaging/README.md) for how these packages are
built and published.

## Usage

```sh
cargo build --release                  # CPU
cargo build --release --features cuda  # CUDA

gliner \
  --text "Alice works for Acme in Paris." \
  --entities person,company,location \
  --relations works_for,located_in \
  --classify "sentiment=positive,negative,neutral" \
  --spans --confidence
```

If needed it downloads `fastino/gliner2.5-multi-v1` into the cache directory on
first run (see [CLI checkpoint resolution](#cli-checkpoint-resolution)).

- `--json "order=order_id::str,status::[shipped|pending],items::list"` extracts structures
  (repeatable). Use `;` between fields if a description contains commas, and add
  `--legacy-structures` for the aggregate decoder.
- `--entities "dosage:Amounts such as 400mg"` adds a label description.
- `--classify "+aspects=a,b,c"` makes the task multi-label.
- `--char-split` switches to the character-level splitter for Chinese and Japanese.
- `--model-variant {multi,small,base}` picks a different GLiNER2.5 checkpoint (see
  [`gliner-classify`](#gliner-classify-zero-shot-classification-from-the-terminal) below for
  what each one is).

### Library

```rust
use candle_core::{DType, Device};
use gliner_rs::{ClassificationSpec, ExtractOptions, GLiNER2, Schema};

let model = GLiNER2::load("../", &Device::Cpu, DType::F32)?;
let schema = Schema::new()
    .entities(["person", "company"])
    .relations(["works_for"])
    .classification(ClassificationSpec::new("sentiment", ["positive", "negative"]));
let opts = ExtractOptions { include_spans: true, include_confidence: true, ..Default::default() };
let result = model.extract("Alice works for Acme.", &schema, &opts)?;

let orders = model.extract_json(
    "Order #1234 for 3 laptops was shipped via FedEx.",
    &[("order", &["order_id::str", "status::[shipped|pending]", "carrier::str"])],
    &ExtractOptions::default(),
)?;
```

The output JSON follows gliner2's formatted results. `start` and `end` are character
(code point) offsets, which is the same convention as Python string indexing.

Entity attributes attach extra properties to an already-extracted entity's exact span,
instead of proposing spans of their own:

```rust
use gliner_rs::AttributeGroup;

let schema = Schema::new().entities(["product"]).entity_attributes([(
    "color",
    AttributeGroup::new(["red", "blue", "green"]).applies_to(["product"]),
)]);
let result = model.extract("Apple released a red iPhone this year.", &schema, &opts)?;
// entities.product[0] == {"text": "iPhone", "color": {"label": "red", "confidence": 0.94}, ...}
```

A group is single-label by default (softmax + argmax over its values); call
`.multi_label(threshold)` for independent sigmoid decisions instead. `.qualify_labels()`
prefixes a group's values with its name in the model-facing prompt (`"color: red"`) while
keeping the returned label unqualified, for when the same value string is meaningful in more
than one group.

### Long documents

Every extraction method has a `_long` counterpart for text that overruns the encoder's
context window (the boundary architecture's default checkpoints top out at 512 tokens):

```rust
use gliner_rs::ChunkOptions;

let result = model.extract_entities_long(
    &long_document,
    &["person", "company", "location"],
    &ExtractOptions { include_spans: true, ..Default::default() },
    ChunkOptions::default(), // 384-word chunks, 64-word overlap
)?;
```

The document is split into overlapping word windows (`ChunkOptions::chunk_size`/
`chunk_overlap`, in words as counted by the active word splitter), each window is run
through the ordinary single-pass API, and every span in the result is remapped to
character offsets in the *original* document before merging. Predictions that show up in
more than one chunk's overlap region are deduplicated: exact re-detections keep their
highest-confidence copy, and genuine span overlaps are resolved with `opts.overlap_policy`
(default `flat`, i.e. the maximum-total-confidence non-overlapping set — the same four
policies as single-pass entity extraction apply here too). Relation edges are always
merged with `allow`, since a chunk boundary can only ever re-detect the same edge, not
create a genuine overlap to resolve. `extract_relations_long` and `extract_json_long`
never synthesize relations or record fields across a chunk boundary — an edge or field only
survives if both halves it needs were extracted from the same chunk.

`extract_long` is the schema-based generic entry point the four convenience methods above
build on, for schemas that mix entities, relations, structures and classification in one
chunked pass.

### Batching

Every extraction and classification method has a `_batch` counterpart for running several
*independent* texts against the same schema in one encoder pass instead of one call per text:

```rust
let texts = ["Apple released a new iPhone.", "Nvidia unveiled a new GPU."];
let results = model.extract_entities_batch(
    &texts,
    &["company", "product"],
    &ExtractOptions::default(),
)?; // one Value per text, in order
```

The texts are tokenized independently, padded to the batch's longest sequence, and run
through the encoder together; padding is masked out of attention so every text's result is
identical to what the single-text method would give it alone — batching only saves encoder
calls, it never changes an answer. `gliner-classify` uses `classification_probabilities_batch`
for `-f`/multi-text input. As with the rest of the API, one call takes one schema: every text
in a batch is scored against the same entities/labels/structures, only the documents vary.

## CLI checkpoint resolution

`gliner`, `gliner-classify`, `gliner-pii` and `gliner-guardrails` all resolve their model
directory the same way when you don't pass `--model`/`GLINER_MODEL` explicitly: each variant's checkpoint
lives in the gliner-rs cache directory, `$XDG_CACHE_HOME/gliner-rs` (default
`~/.cache/gliner-rs`), under a subdirectory named after its Hugging Face repo (e.g.
`gliner2.5-multi-v1`). If it isn't there yet, it's downloaded automatically on first use.
Set `GLINER_OFFLINE=1` to disable downloading (useful on a constrained connection); resolution
then fails with an error naming the missing file unless the checkpoint is already fully cached.

Each binary supports more than one checkpoint (`--model-variant`, see the tables below).

If you already have checkpoints downloaded elsewhere (e.g. via `git lfs` or the `huggingface-cli`),
symlink the whole collection into the cache directory instead of re-downloading:

```sh
ln -s /path/to/your/gliner/checkpoints ~/.cache/gliner-rs
```

where `/path/to/your/gliner/checkpoints` contains subdirectories named after the Hugging Face
repos (`gliner2.5-multi-v1`, `gliner2-privacy-filter-PII-multi`, ...).

## `gliner-classify`: zero-shot classification from the terminal

`gliner-classify` is a single binary that classifies text straight from command-line flags,
or through an interactive shell that loads the model once and keeps history. Build it
with the rest of the crate:

```sh
cargo build --release                      # add --features cuda for GPU
```

`--model-variant {multi,small,base}` picks between `fastino/gliner2.5-multi-v1` (default,
205M, all languages), `fastino/gliner2.5-small-v1` and `fastino/gliner2.5-base-v1` (smaller,
faster, English-leaning). Each is downloaded automatically into the cache directory on
first use (see [CLI checkpoint resolution](#cli-checkpoint-resolution)).

Every output below is a real run of `fastino/gliner2.5-multi-v1` on CPU.

### One-shot CLI

Texts come from positional arguments, from `-f file` (one per line), or from piped stdin.

**Single-label**
```console
$ gliner-classify -l positive,negative,neutral "I love this phone"
label: positive (1.000)
```

**Multi-label** (`-m`; every label at or above `--threshold`, default 0.5)
```console
$ gliner-classify -m -l camera,performance,battery,display,price \
    "Great camera quality, decent performance, but poor battery life."
label: camera (0.658), performance (0.790), battery (0.526)
```

**Several tasks in one pass**, where `+` makes a task multi-label and `-a` shows every
probability (`*` marks the predictions)
```console
$ gliner-classify -t sentiment=positive,negative -t +topics=technology,sports,politics,finance -a \
    "The Fed raised rates, and tech stocks tumbled."
sentiment: *negative (1.000), positive (0.000) | topics: *finance (0.984), *technology (0.898), *politics (0.542), sports (0.001)
```

**Batch input with a prompt.** Each line of output is the text, a tab, then the result.
```console
$ gliner-classify -l book_flight,cancel_booking,check_status,baggage_info,talk_to_human \
    -p "What does the customer want to do?" \
    "My flight to Rome got moved, can I get my money back?" \
    "where is my suitcase" \
    "just give me a real person please"
My flight to Rome got moved, can I get my money back?	label: cancel_booking (0.721)
where is my suitcase	label: baggage_info (0.993)
just give me a real person please	label: talk_to_human (1.000)
```

**Multilingual.** English labels work on any language.
```console
$ gliner-classify -l positive,negative,neutral "Das Essen war kalt und der Kellner unhöflich." \
    "这家餐厅的服务太棒了！" "C'était correct, sans plus."
Das Essen war kalt und der Kellner unhöflich.	label: negative (0.857)
这家餐厅的服务太棒了！	label: positive (1.000)
C'était correct, sans plus.	label: positive (0.724)
```

**Descriptions and few-shot examples**, with `-k N` for the top N labels
```console
$ gliner-classify -t "queue=billing:Payments invoices refunds,tech:Bugs crashes errors,account:Login password access" \
     -t priority=urgent,normal,low \
     -e "Production is down for all users!!=>urgent" -k 2 \
     "I was charged twice this month and can't log in to fix it"
queue: *billing (0.960), account (0.029) | priority: *urgent (0.895), normal (0.064)
```

**Pipelines** with `--format jsonl|json|tsv`
```console
$ printf "Win a free iPhone now\nLunch tomorrow?\n" | gliner-classify -l spam,ham --format tsv
Win a free iPhone now	label	spam	0.6519
Lunch tomorrow?	label	ham	0.5609

$ gliner-classify -t "+flags=toxic:Insults or harassment,spam:Ads or scams,nsfw:Sexual content,self_harm" \
    --threshold 0.4 --format jsonl "Click here to win a free iPhone, you idiot"
{"text":"Click here to win a free iPhone, you idiot","flags":{"labels":["toxic","nsfw"],"confidences":[0.7689375877380371,0.6499032974243164]}}
```

Zero-shot labels and descriptions are prompts, and wording matters. In the last example the
model flags `nsfw` rather than `spam`, so tune the descriptions and threshold on your own
data. Adding descriptions can even flip a result: for `spam,ham` on
"Win a free iPhone now, click here!", bare labels give `spam` and the descriptions above
give `ham`. The Python library behaves identically.

| Flag | Meaning |
|---|---|
| `-l, --labels a,b:desc,c` | labels of the default task (named by `-n`, default `label`) |
| `-m, --multi` | make the `--labels` task multi-label |
| `-t, --task [+]name=a,b` | add a task (`+` = multi-label), repeatable |
| `-e, --example "text=>label"` | few-shot example, used by tasks that have the label |
| `-p, --prompt TEXT` | instruction appended to the task name |
| `--threshold 0.5` | multi-label cutoff |
| `--activation auto\|softmax\|sigmoid` | override scoring (auto: softmax single, sigmoid multi) |
| `-a, --all` / `-k, --top-k N` | show all or the top N label probabilities |
| `--format text\|jsonl\|json\|tsv` | output format |
| `-f, --file PATH` | read texts line by line |
| `-i, --interactive` | start the shell |
| `--model DIR`, `--cuda`, `--fp16`, `-v` | model location, device, precision, timing |

### Interactive shell

Run `gliner-classify -i`, or run `gliner-classify` with no input in a terminal. You can
preload settings with the usual flags, e.g. `gliner-classify -i -l positive,negative`. The
model loads once; after that, anything you type that isn't a command is classified with
the current settings.

```console
$ gliner-classify -i
loading model from .. ...
model loaded in 344.93ms
gliner-classify shell: type text to classify, :help for commands, Ctrl-D to quit
classify> :labels positive,negative,neutral
label> :task +topics=battery,camera,shipping,price,display
[2 tasks]> :all
[2 tasks]> Arrived late and the camera is blurry
label: *negative (0.991), positive (0.005), neutral (0.004) | topics: *camera (0.764), shipping (0.217), display (0.134), price (0.055), battery (0.007)
```

The prompt shows what's active: `label>` for one task, `+topics>` for a multi-label task,
and `[2 tasks]>` for several.

#### Recipes

**Emotion wheel, top 3**
```console
classify> :labels joy,sadness,anger,fear,surprise,disgust,trust,anticipation
label> :top 3
label> I can't believe they actually picked my design for the launch!
label: *surprise (0.762), trust (0.092), anticipation (0.052)
```

**Support ticket router**
```console
classify> :task queue=billing:Payments invoices refunds,tech:Bugs crashes errors,account:Login password access
queue> :task priority=urgent,normal,low
[2 tasks]> :example Production is down for all users!!=>urgent
[2 tasks]> :top 2
[2 tasks]> I was charged twice this month and can't log in to fix it
queue: *billing (0.960), account (0.029) | priority: *urgent (0.895), normal (0.064)
```

**Chatbot intent detection**
```console
classify> :labels book_flight,cancel_booking,check_status,baggage_info,talk_to_human
label> :prompt What does the customer want to do?
label> where is my suitcase
label: baggage_info (0.993)
label> just give me a real person please
label: talk_to_human (1.000)
```

**Moderation as JSON**
```console
classify> :task +flags=toxic:Insults or harassment,spam:Ads or scams,nsfw:Sexual content,self_harm
+flags> :threshold 0.4
+flags> :format jsonl
+flags> Click here to win a free iPhone, you idiot
{"text":"Click here to win a free iPhone, you idiot","flags":{"labels":["toxic","nsfw"],"confidences":[0.7689375877380371,0.6499032974243164]}}
```

**Batch a file**
```console
classify> :labels spam,ham
label> :format tsv
label> :time
label> :file ~/mail/inbox.txt
Win a free iPhone now	label	spam	0.6519
Lunch tomorrow?	label	ham	0.5609
(2 text(s) in …)
```

#### Commands

| Command | Effect |
|---|---|
| `:labels a,b:desc,c` | set the default `label` task |
| `:multi [on\|off]` | toggle multi-label on the `label` task |
| `:task [+]name=a,b` / `:rm name` | add or replace a task / remove it |
| `:example text=>label` / `:example clear` | add few-shot examples / drop them |
| `:prompt TEXT\|off` | set the instruction |
| `:threshold 0.4`, `:activation softmax` | scoring |
| `:all [on\|off]`, `:top N\|off` | show probabilities |
| `:format text\|jsonl\|json\|tsv` | output format |
| `:file PATH` | classify each line of a file |
| `:time [on\|off]` | print timing after each run |
| `:show`, `:clear`, `:help`, `:quit` | inspect, reset, help, exit |

Tips:
- `:` followed by Tab completes commands.
- Up arrow and Ctrl-R search history, which persists in
  `$XDG_STATE_HOME/gliner-classify/history` (default `~/.local/state/gliner-classify/history`).
- Ctrl-C clears the current line and Ctrl-D exits.
- Enter one command per line: pasting several lines at once can drop some of them.
- For snappy demos, start with `--cuda --fp16` and turn on `:time`. On CPU a short text
  takes about 0.2 s.

## `gliner-pii`: PII detection and redaction

`gliner-pii` runs entity extraction preloaded with the 42-label PII taxonomy from
`fastino/gliner2-privacy-filter-PII-multi`, so a redaction pass is one command with no
label list to write. Texts come from positional arguments, `-f file` (one per line), or
piped stdin.

**Default JSON output** — one object per line, with character spans and confidence for every
label in the taxonomy (empty labels included, so downstream tooling can rely on the shape):
```console
$ gliner-pii "Email john.smith@acme.com or call +1 415 555 0199."
{"text":"Email john.smith@acme.com or call +1 415 555 0199.","entities":{"person":[],"full_name":[],"first_name":[],"middle_name":[],"last_name":[],"date_of_birth":[],"email":[{"text":"john.smith@acme.com","confidence":0.999998927116394,"start":6,"end":25}],"phone_number":[{"text":"+1 415 555 0199","confidence":1.0,"start":34,"end":49}],"address":[], ...}}
```

**Redaction** (`-r`) — prints the text back with matched spans replaced by `[LABEL]` instead
of JSON:
```console
$ gliner-pii -r "Email john.smith@acme.com or call +1 415 555 0199."
Email [EMAIL] or call [PHONE_NUMBER].
```

**A narrower label set** with `-l`, useful when you only care about a few PII types or want
to skip low-signal ones like `sensitive_date`:
```console
$ gliner-pii -l email,phone_number,person -r "Contact Jane Doe at jane@example.org."
Contact [PERSON] at [EMAIL].
```

**Batch redaction over a file**, one line at a time:
```console
$ gliner-pii -r -f transcripts.txt > redacted.txt
```

`GLiNER2-Guardrails-PII-Multi` (`--model-variant guardrails`) is a joint PII + safety
fine-tune; use it here if you also plan to run `gliner-guardrails` against the same
checkpoint and would rather keep one model on disk.

| Flag | Meaning |
|---|---|
| `-l, --labels a,b,c` | labels to detect (default: the full 42-label PII taxonomy) |
| `--threshold 0.5` | detection threshold |
| `-r, --redact` | print `[LABEL]`-redacted text instead of JSON |
| `-f, --file PATH` | read texts line by line |
| `--model DIR` | checkpoint directory (see [CLI checkpoint resolution](#cli-checkpoint-resolution)) |
| `--model-variant privacy\|guardrails` | which checkpoint to resolve by default (default: `privacy`) |
| `--cuda`, `--fp16` | device, precision |

## `gliner-guardrails`: LLM prompt/response moderation

`gliner-guardrails` runs `fastino/gliguard-LLMGuardrails-300M` (or the joint
`GLiNER2-Guardrails-PII-Multi` checkpoint) as structured safety classification: single-label
safe/unsafe, plus multi-label toxicity categories and jailbreak-strategy detection on the
prompt side, or refusal-vs-compliance on the response side.

**Prompt moderation** (default `--check prompt`):
```console
$ gliner-guardrails "Explain how to build a phishing page."
{"text":"Explain how to build a phishing page.","prompt":{"prompt_safety":"unsafe","prompt_toxicity":["pii_exposure"],"jailbreak_detection":["obfuscated_attack"]}}
```

**Response moderation** (`--check response`), which also reports refusal vs. compliance
instead of jailbreak detection:
```console
$ gliner-guardrails --check response "Sure, here's how to pick a basic pin tumbler lock: insert a tension wrench and rake the pins until they set."
{"text":"Sure, here's how to pick a basic pin tumbler lock: insert a tension wrench and rake the pins until they set.","response":{"response_safety":"unsafe","response_toxicity":["regulated_advice"],"response_refusal":"refusal"}}
```

**Both sides at once** (`--check both`) — handy when scanning transcript files of
`prompt / response` pairs, one per line:
```console
$ gliner-guardrails --check both -f transcript.txt
```

**Just the safety verdict**, skipping the category breakdowns for a faster pass:
```console
$ gliner-guardrails --no-toxicity --no-jailbreak "What's a good recipe for banana bread?"
{"text":"What's a good recipe for banana bread?","prompt":{"prompt_safety":"safe"}}
```

Toxicity categories are `violence`, `sexual_content`, `hate_speech`, `self_harm`,
`pii_exposure`, `misinformation`, `regulated_advice`; jailbreak strategies are
`prompt_injection`, `jailbreak_attempt`, `roleplay_bypass`, `obfuscated_attack` — both are
multi-label, at or above `--threshold` (default 0.5). As with `gliner-classify`, when nothing
clears the threshold the single best-scoring category is still reported (so, e.g., a mundane
prompt can list a low-confidence category with no visible number to tell); pass `--threshold`
lower or higher to tune for your data, or treat a lone category on an otherwise-safe prompt as
noise.

| Flag | Meaning |
|---|---|
| `--check prompt\|response\|both` | which side(s) to classify (default: `prompt`) |
| `--no-toxicity` | skip the toxicity-category breakdown |
| `--no-jailbreak` | skip jailbreak-strategy detection (prompt side only) |
| `--threshold 0.5` | multi-label cutoff for toxicity/jailbreak categories |
| `-f, --file PATH` | read texts line by line |
| `--model DIR` | checkpoint directory (see [CLI checkpoint resolution](#cli-checkpoint-resolution)) |
| `--model-variant gliguard\|guardrails-pii` | which checkpoint to resolve by default (default: `gliguard`) |
| `--cuda`, `--fp16` | device, precision |

## Parity with the Python reference

```sh
python scripts/parity.py .. scripts/parity_cases.json > py.jsonl       # needs gliner2 + transformers>=5
cargo run --release --example parity -- .. scripts/parity_cases.json > rs.jsonl
python scripts/compare.py py.jsonl rs.jsonl 1e-4
```

The cases cover English, German and Chinese text, `extract_json` records (multi-record,
choice fields, descriptions, legacy mode, no match), descriptions, mixed multi-task schemas,
emails and URLs, entity attributes (single- and multi-label groups, `applies_to`,
`qualify_labels`), and a text of about 1,000 words that exercises the log-bucketed relative
positions and the windowed boundary attention. On CPU, every span, label and structure
matches, and confidences agree within 1e-4 (usually about 1e-6). The one reported
difference is the order of two spans whose scores differ by 1e-7.

## Implementation notes

- `deberta.rs`: DeBERTa-v2 with disentangled c2p/p2c attention, shared keys and log buckets.
- `heads.rs`: the boundary encoder (windowed attention plus SwiGLU), query marginals with a
  centered inside prefix, the shared document candidate pool (top-k union, per-query quota,
  dedup), the FiLM pool scorer, the classifier MLP and the relation scorer. The discrete
  top-k and dedup steps run on CPU vectors and match torch's stable sort ordering.
- `processor.rs`: prompt layout `( [P] prompt ( [E] label ... ) ) [SEP_STRUCT] ... [SEP_TEXT] words`,
  first-subword routing, and the reference collator's quirk of appending `.` when the text
  doesn't end in `.`, `!` or `?`.
- `records.rs`: natural-mode instance formation, a global exclusive assignment (Hungarian
  solver with the reference tie-breaking), and literal choice-mention binding.
- Choice fields prepend `( struct: field ( a | b ) )` words to the text. All spans are shifted
  by that prefix, and choice values are scored at their prefix positions by the per-query
  pair reranker (rotary endpoints).
- `decode.rs`: weighted-interval overlap resolution with the reference tie-breaking, relation
  pair proposal, and relation edge deduplication.

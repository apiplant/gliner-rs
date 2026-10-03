//! HTTP server for schema extraction (native only), built on ntex.
//!
//! * `POST /v1/extract`: one text plus a schema written in the same syntax as the `gliner` CLI flags:
//!
//!   ```json
//!   {"text": "Alice works for Acme in Paris.",
//!    "entities": ["person", "company", "location:a city or town"],
//!    "relations": ["works_for"],
//!    "json": ["employee=name::str,employer::str"],
//!    "classify": ["sentiment=positive,negative,neutral"],
//!    "threshold": 0.5, "spans": true, "confidence": true, "overlap": "flat"}
//!   ```
//!
//!   At least one of `entities`, `relations`, `json`, `classify` is required. The answer is
//!   `{"model": .., "result": <what the CLI prints>}`.
//! * `GET /health`: `{"status":"ready","model":..}`, no inference.
//!
//! Requests run strictly one at a time (one forward in flight) behind a bounded admission queue: a
//! request that finds `1 + max_queued` requests already admitted gets `429` with `Retry-After: 1`.
//! Bad input is `422` with `{"error": {message, type: "invalid_request_error", code: 422, param}}`; a
//! failed forward is `500 {"detail":"internal error"}` (the cause goes to stderr).
//!
//! [`start_server`] must run inside an ntex runtime. ntex owns SIGINT/SIGTERM: the listener stops,
//! in-flight requests (including a running forward) finish, and [`ServerHandle::wait`] resolves.

use crate::cli_schema::{CliSchemaArgs, build_schema};
use crate::{ExtractOptions, GLiNER2, OverlapPolicy, Schema};
use anyhow::Result;
use ntex::http::{StatusCode, header};
use ntex::server::Server;
use ntex::util::BytesMut;
use ntex::web::types::{Payload, State};
use ntex::web::{self, App, HttpRequest, HttpResponse, HttpServer, ServiceConfig};
use serde::Deserialize;
use serde_json::{Value, json};
use std::net::TcpListener;
use std::sync::Arc;
use tokio::sync::{Mutex, Semaphore};

/// Request-body cap.
pub const MAX_BODY_BYTES: usize = 1_048_576;
/// How much of a rejected (oversized) body is read and discarded before replying.
const DRAIN_LIMIT: usize = 16 * MAX_BODY_BYTES;

/// What the server needs from a model; [`GLiNER2`] implements it, tests use a stand-in.
pub trait Extractor: Send + Sync + 'static {
    /// `Err` means this model cannot answer this schema (reported as `422`, before any forward pass).
    fn validate(&self, schema: &Schema) -> Result<()>;
    /// One extraction (synchronous; runs on the blocking pool).
    fn extract(&self, text: &str, schema: &Schema, opts: &ExtractOptions) -> Result<Value>;
}

impl Extractor for GLiNER2 {
    fn validate(&self, schema: &Schema) -> Result<()> {
        if matches!(self, GLiNER2::Span(_)) && (!schema.structures.is_empty() || !schema.relations.is_empty()) {
            anyhow::bail!("this checkpoint only supports entities and classification (no relations or json structures)");
        }
        Ok(())
    }

    fn extract(&self, text: &str, schema: &Schema, opts: &ExtractOptions) -> Result<Value> {
        GLiNER2::extract(self, text, schema, opts)
    }
}

#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// Waiting slots on top of the one in-flight request.
    pub max_queued: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self { max_queued: 16 }
    }
}

#[derive(Debug)]
pub enum ExecError {
    /// 422: the request is not answerable (`param` names the offending field).
    Invalid { param: &'static str, message: String },
    /// 429: the admission queue is full.
    QueueFull,
    /// 500: the forward pass failed.
    Internal(String),
}

struct Inner {
    extractor: Arc<dyn Extractor>,
    model_name: String,
    model_lock: Arc<Mutex<()>>,
    admission: Arc<Semaphore>,
}

/// Shared state behind the routes (cheap to clone).
#[derive(Clone)]
pub struct ServerState(Arc<Inner>);

impl ServerState {
    pub fn new(extractor: Arc<dyn Extractor>, model_name: impl Into<String>, config: ServerConfig) -> Self {
        let admission = Arc::new(Semaphore::new(config.max_queued.saturating_add(1)));
        Self(Arc::new(Inner { extractor, model_name: model_name.into(), model_lock: Arc::new(Mutex::new(())), admission }))
    }

    pub fn model_name(&self) -> &str {
        &self.0.model_name
    }

    /// Admission slots currently free (0 means the next request is rejected with 429).
    pub fn admission_available_slots(&self) -> usize {
        self.0.admission.available_permits()
    }

    /// Queue and run one extraction.
    pub async fn execute(&self, text: String, schema: Schema, opts: ExtractOptions) -> Result<Value, ExecError> {
        self.0.extractor.validate(&schema).map_err(|e| ExecError::Invalid { param: "schema", message: e.to_string() })?;
        let permit = self.0.admission.clone().try_acquire_owned().map_err(|_| ExecError::QueueFull)?;
        let lock = self.0.model_lock.clone().lock_owned().await;
        let extractor = Arc::clone(&self.0.extractor);
        // The permit and lock move into the blocking closure, so a client disconnect (dropping this
        // future) can never let a second forward overlap one that is still running.
        ntex::rt::spawn_blocking(move || {
            let result = extractor.extract(&text, &schema, &opts);
            drop(lock);
            drop(permit);
            result
        })
        .await
        .map_err(|e| ExecError::Internal(format!("extraction task failed: {e}")))?
        .map_err(|e| ExecError::Internal(e.to_string()))
    }
}

/// The JSON body of `POST /v1/extract`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExtractRequest {
    #[serde(default)]
    model: Option<String>,
    text: String,
    #[serde(default)]
    entities: Vec<String>,
    #[serde(default)]
    relations: Vec<String>,
    #[serde(default)]
    json: Vec<String>,
    #[serde(default)]
    classify: Vec<String>,
    #[serde(default)]
    legacy_structures: bool,
    #[serde(default = "default_threshold")]
    threshold: f32,
    #[serde(default)]
    spans: bool,
    #[serde(default)]
    confidence: bool,
    #[serde(default)]
    overlap: Option<String>,
    #[serde(default)]
    max_words: Option<usize>,
}

fn default_threshold() -> f32 {
    0.5
}

fn json_response(status: StatusCode, retry_after: bool, body: Value) -> HttpResponse {
    let mut builder = HttpResponse::build(status);
    builder.content_type("application/json");
    if retry_after {
        builder.header("retry-after", "1");
    }
    builder.body(body.to_string())
}

fn invalid(param: &str, message: impl Into<String>) -> HttpResponse {
    let message = message.into();
    json_response(
        StatusCode::UNPROCESSABLE_ENTITY,
        false,
        json!({"error": {"message": message, "type": "invalid_request_error", "code": 422, "param": param}}),
    )
}

/// Discard the rest of a body we are rejecting, so the client gets our 422 instead of a connection
/// reset mid-upload. Nothing is buffered, and we give up after [`DRAIN_LIMIT`] bytes.
async fn drain(mut payload: Payload) {
    let mut seen = 0;
    while let Some(Ok(chunk)) = payload.recv().await {
        seen += chunk.len();
        if seen > DRAIN_LIMIT {
            break;
        }
    }
}

async fn read_body(req: &HttpRequest, mut payload: Payload) -> Option<Vec<u8>> {
    let declared = req.headers().get(header::CONTENT_LENGTH).and_then(|v| v.to_str().ok()).and_then(|v| v.parse::<usize>().ok());
    if declared.is_some_and(|n| n > MAX_BODY_BYTES) {
        drain(payload).await;
        return None;
    }
    let mut buf = BytesMut::new();
    while let Some(chunk) = payload.recv().await {
        let chunk = chunk.ok()?;
        if buf.len() + chunk.len() > MAX_BODY_BYTES {
            drain(payload).await;
            return None;
        }
        buf.extend_from_slice(&chunk);
    }
    Some(buf.to_vec())
}

async fn extract(state: State<ServerState>, req: HttpRequest, payload: Payload) -> HttpResponse {
    let Some(bytes) = read_body(&req, payload).await else {
        return invalid("body", "request body exceeds the 1 MiB limit");
    };
    if let Some(ct) = req.headers().get(header::CONTENT_TYPE) {
        let mime = ct.to_str().unwrap_or_default().split(';').next().unwrap_or_default().trim();
        if mime != "application/json" && !mime.ends_with("+json") {
            return invalid("Content-Type", format!("Content-Type must be application/json (got '{mime}')"));
        }
    }
    let body: ExtractRequest = match serde_json::from_slice(&bytes) {
        Ok(b) => b,
        Err(e) => return invalid("body", format!("invalid request: {e}")),
    };
    if body.model.as_deref().is_some_and(|m| m != state.model_name()) {
        return invalid("model", format!("Loaded model is '{}'", state.model_name()));
    }
    if body.text.trim().is_empty() {
        return invalid("text", "text must not be empty");
    }
    if !(0.0..=1.0).contains(&body.threshold) {
        return invalid("threshold", "threshold must be between 0 and 1");
    }
    let overlap_policy = match body.overlap.as_deref().map(str::parse::<OverlapPolicy>).transpose() {
        Ok(p) => p,
        Err(e) => return invalid("overlap", e.to_string()),
    };
    let schema = match build_schema(&CliSchemaArgs {
        entities: body.entities,
        relations: body.relations,
        json: body.json,
        legacy_structures: body.legacy_structures,
        classify: body.classify,
    }) {
        Ok(s) => s,
        Err(e) => return invalid("schema", e.to_string()),
    };
    let opts = ExtractOptions {
        threshold: body.threshold,
        include_confidence: body.confidence,
        include_spans: body.spans,
        overlap_policy,
        max_words: body.max_words,
    };
    match state.execute(body.text, schema, opts).await {
        Ok(result) => json_response(StatusCode::OK, false, json!({"model": state.model_name(), "result": result})),
        Err(ExecError::Invalid { param, message }) => invalid(param, message),
        Err(ExecError::QueueFull) => json_response(StatusCode::TOO_MANY_REQUESTS, true, json!({"detail": "Scoring queue is full"})),
        Err(ExecError::Internal(cause)) => {
            eprintln!("extraction failed: {cause}");
            json_response(StatusCode::INTERNAL_SERVER_ERROR, false, json!({"detail": "internal error"}))
        }
    }
}

async fn health(state: State<ServerState>) -> HttpResponse {
    json_response(StatusCode::OK, false, json!({"status": "ready", "model": state.model_name()}))
}

/// Register the routes (and shared state) on an app: `App::new().configure(routes(state))`.
pub fn routes(state: ServerState) -> impl FnOnce(&mut ServiceConfig) {
    move |cfg| {
        cfg.state(state).route("/v1/extract", web::post().to(extract)).route("/health", web::get().to(health));
    }
}

/// A running server: the bound port, the model identity, and the ntex server.
pub struct ServerHandle {
    pub port: u16,
    pub model_name: String,
    server: Server,
}

impl ServerHandle {
    /// Resolves once the server has stopped (SIGINT/SIGTERM or [`Self::stop`]).
    pub async fn wait(self) {
        let _ = self.server.await;
    }

    /// Graceful stop: no new connections, in-flight requests finish.
    pub async fn stop(self) {
        self.server.stop(true).await;
    }
}

/// Bind `host:port` (port 0 = ephemeral) and serve. The model must already be loaded.
pub async fn start_server(
    host: &str,
    port: u16,
    extractor: Arc<dyn Extractor>,
    model_name: impl Into<String>,
    config: ServerConfig,
) -> Result<ServerHandle> {
    let model_name = model_name.into();
    let state = ServerState::new(extractor, model_name.clone(), config);
    let listener = TcpListener::bind((host, port)).map_err(|e| anyhow::anyhow!("failed to bind {host}:{port}: {e}"))?;
    let port = listener.local_addr()?.port();
    // Inference is serial, so a couple of workers are plenty for connection handling.
    let server = HttpServer::new(async move || App::new().configure(routes(state.clone()))).workers(2).listen(listener)?.run();
    Ok(ServerHandle { port, model_name, server })
}

//! Model-free tests for the HTTP server: routes on ntex's in-process test server over a stand-in `Extractor`.
use anyhow::{Result, bail};
use gliner_rs::server::{Extractor, ServerConfig, ServerState, routes, start_server};
use gliner_rs::{ExtractOptions, Schema};
use ntex::client::ClientResponse;
use ntex::http::StatusCode;
use ntex::web::App;
use ntex::web::test::{self, TestServer};
use serde_json::{Value, json};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// Echoes what it was asked; optionally parks inside the extraction until released. Rejects relations
/// in `validate` like the span architecture does.
#[derive(Default)]
struct Mock {
    fail_once: AtomicBool,
    reject_relations: AtomicBool,
    extractions: AtomicUsize,
    running: AtomicUsize,
    max_running: AtomicUsize,
    gate: Option<Gate>,
}

#[derive(Default)]
struct Gate(Mutex<bool>, Condvar);

impl Gate {
    fn open(&self) {
        *self.0.lock().unwrap() = true;
        self.1.notify_all();
    }
    fn wait(&self) {
        let mut open = self.0.lock().unwrap();
        while !*open {
            open = self.1.wait(open).unwrap();
        }
    }
}

impl Extractor for Mock {
    fn validate(&self, schema: &Schema) -> Result<()> {
        if self.reject_relations.load(Ordering::SeqCst) && !schema.relations.is_empty() {
            bail!("this checkpoint only supports entities and classification");
        }
        Ok(())
    }

    fn extract(&self, text: &str, schema: &Schema, opts: &ExtractOptions) -> Result<Value> {
        let now = self.running.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_running.fetch_max(now, Ordering::SeqCst);
        self.extractions.fetch_add(1, Ordering::SeqCst);
        if let Some(g) = &self.gate {
            g.wait();
        }
        self.running.fetch_sub(1, Ordering::SeqCst);
        if self.fail_once.swap(false, Ordering::SeqCst) {
            bail!("simulated failure");
        }
        Ok(json!({
            "text": text,
            "entities": schema.entities.len(),
            "classifications": schema.classifications.len(),
            "threshold": opts.threshold,
            "spans": opts.include_spans,
            "confidence": opts.include_confidence,
        }))
    }
}

fn body() -> Value {
    json!({
        "text": "Alice works for Acme in Paris.",
        "entities": ["person", "company", "location:a city"],
        "classify": ["sentiment=positive,negative"],
        "threshold": 0.75, "spans": true
    })
}

async fn server(mock: Arc<Mock>, config: ServerConfig) -> (TestServer, ServerState) {
    let state = ServerState::new(mock, "multi", config);
    let s = state.clone();
    (test::server(async move || App::new().configure(routes(s.clone()))).await, state)
}

async fn post(srv: &TestServer, content_type: &str, body: String) -> (StatusCode, Value, ClientResponse) {
    let res = srv.post("/v1/extract").header("content-type", content_type).send_body(body).await.unwrap();
    let bytes = res.body().limit(usize::MAX).await.unwrap();
    (res.status(), serde_json::from_slice(&bytes).unwrap_or(Value::Null), res)
}

#[ntex::test]
async fn extract_builds_the_schema_from_cli_syntax_and_applies_options() {
    let mock = Arc::new(Mock::default());
    let (srv, _) = server(mock.clone(), ServerConfig::default()).await;
    let (status, v, _) = post(&srv, "application/json", body().to_string()).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["model"], "multi");
    assert_eq!(
        v["result"],
        json!({"text": "Alice works for Acme in Paris.", "entities": 3, "classifications": 1, "threshold": 0.75, "spans": true, "confidence": false})
    );
}

#[ntex::test]
async fn health_reports_the_model_without_an_extraction() {
    let mock = Arc::new(Mock::default());
    let (srv, _) = server(mock.clone(), ServerConfig::default()).await;
    let res = srv.get("/health").send().await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let v: Value = serde_json::from_slice(&res.body().await.unwrap()).unwrap();
    assert_eq!(v, json!({"status": "ready", "model": "multi"}));
    assert_eq!(mock.extractions.load(Ordering::SeqCst), 0);
}

#[ntex::test]
async fn invalid_requests_are_422_before_any_extraction() {
    let mock = Arc::new(Mock::default());
    mock.reject_relations.store(true, Ordering::SeqCst);
    let (srv, _) = server(mock.clone(), ServerConfig::default()).await;
    let with = |k: &str, v: Value| {
        let mut b = body();
        b[k] = v;
        b.to_string()
    };
    let big = json!({"text": "x".repeat(2 * 1024 * 1024), "entities": ["a"]}).to_string();
    let cases = [
        ("model", "Loaded model is 'multi'", "application/json", with("model", json!("other"))),
        ("text", "text must not be empty", "application/json", with("text", json!("  "))),
        ("threshold", "between 0 and 1", "application/json", with("threshold", json!(1.5))),
        ("overlap", "unknown overlap_policy", "application/json", with("overlap", json!("sideways"))),
        ("schema", "nothing to do", "application/json", json!({"text": "t"}).to_string()),
        ("schema", "--classify expects", "application/json", json!({"text": "t", "classify": ["oops"]}).to_string()),
        ("schema", "only supports entities", "application/json", json!({"text": "t", "relations": ["r"]}).to_string()),
        ("body", "unknown field", "application/json", with("bogus", json!(1))),
        ("body", "invalid request", "application/json", "{nope".to_string()),
        ("Content-Type", "application/json", "text/plain", body().to_string()),
        ("body", "1 MiB", "application/json", big),
    ];
    for (param, hint, content_type, payload) in cases {
        let (status, v, _) = post(&srv, content_type, payload).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{hint}: {v}");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["code"], 422);
        assert_eq!(v["error"]["param"], param, "{hint}: {v}");
        assert!(v["error"]["message"].as_str().unwrap().contains(hint), "{hint}: {v}");
    }
    assert_eq!(mock.extractions.load(Ordering::SeqCst), 0, "rejected requests never reach the model");
}

#[ntex::test]
async fn failure_is_500_and_the_server_recovers() {
    let mock = Arc::new(Mock { fail_once: AtomicBool::new(true), ..Default::default() });
    let (srv, _) = server(mock.clone(), ServerConfig::default()).await;
    let (status, v, _) = post(&srv, "application/json", body().to_string()).await;
    assert_eq!((status, v), (StatusCode::INTERNAL_SERVER_ERROR, json!({"detail": "internal error"})));
    let (status, ..) = post(&srv, "application/json", body().to_string()).await;
    assert_eq!(status, StatusCode::OK);
}

#[ntex::test]
async fn full_queue_is_429_with_retry_after_and_extractions_stay_serial() {
    let mock = Arc::new(Mock { gate: Some(Gate::default()), ..Default::default() });
    let (srv, state) = server(mock.clone(), ServerConfig { max_queued: 1 }).await;
    let srv = Arc::new(srv);
    let spawn_request = || {
        let srv = srv.clone();
        ntex::rt::spawn(async move { post(&srv, "application/json", body().to_string()).await.0 })
    };

    let a = spawn_request(); // in flight, parked inside the extraction
    while mock.running.load(Ordering::SeqCst) == 0 {
        ntex::time::sleep(ntex::time::Millis(5)).await;
    }
    let b = spawn_request(); // takes the single waiting slot
    while state.admission_available_slots() > 0 {
        ntex::time::sleep(ntex::time::Millis(5)).await;
    }
    let (status, v, res) = post(&srv, "application/json", body().to_string()).await;
    assert_eq!((status, v), (StatusCode::TOO_MANY_REQUESTS, json!({"detail": "Scoring queue is full"})));
    assert_eq!(res.header("retry-after").unwrap(), "1");

    mock.gate.as_ref().unwrap().open();
    assert_eq!(a.await.unwrap(), StatusCode::OK);
    assert_eq!(b.await.unwrap(), StatusCode::OK);
    assert_eq!(mock.extractions.load(Ordering::SeqCst), 2, "the 429 request never reached the model");
    assert_eq!(mock.max_running.load(Ordering::SeqCst), 1, "one extraction at a time");
}

#[ntex::test]
async fn stop_lets_an_in_flight_request_finish_then_refuses_connections() {
    let mock = Arc::new(Mock { gate: Some(Gate::default()), ..Default::default() });
    let handle = start_server("127.0.0.1", 0, mock.clone(), "multi", ServerConfig::default()).await.unwrap();
    assert_ne!(handle.port, 0);
    let addr = format!("127.0.0.1:{}", handle.port);
    let request = {
        let addr = addr.clone();
        ntex::rt::spawn_blocking(move || raw_post(&addr, &body().to_string()))
    };
    while mock.running.load(Ordering::SeqCst) == 0 {
        ntex::time::sleep(ntex::time::Millis(5)).await;
    }
    let gate = mock.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(150));
        gate.gate.as_ref().unwrap().open();
    });
    handle.stop().await;
    assert_eq!(request.await.unwrap(), Some(200), "the in-flight request completes during graceful stop");
    assert!(std::net::TcpStream::connect(&addr).is_err(), "the listener is closed after stop");
}

/// Bare HTTP/1.1 POST (no client dependency): the status code, or `None` if the connection failed.
fn raw_post(addr: &str, body: &str) -> Option<u16> {
    use std::io::{Read, Write};
    let mut stream = std::net::TcpStream::connect(addr).ok()?;
    let req = format!(
        "POST /v1/extract HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(req.as_bytes()).ok()?;
    let mut raw = String::new();
    stream.read_to_string(&mut raw).ok()?;
    raw.lines().next()?.split_whitespace().nth(1)?.parse().ok()
}

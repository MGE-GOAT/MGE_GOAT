//! OpenAI-compatible provider: `POST {base_url}/chat/completions` with `stream: true`.
//!
//! This single client serves OpenRouter, NVIDIA NIM, and a local llama.cpp
//! `llama-server` — they all implement the same wire format.

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures_util::stream::{BoxStream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{ChatRequest, LlmProvider, Message, Role, StreamEvent, ToolDef};

/// A configured OpenAI-compatible endpoint.
pub struct OpenAiCompat {
    name: String,
    base_url: String,
    api_key: Option<String>,
    http: reqwest::Client,
    /// Extra headers some gateways want (e.g. OpenRouter's referer/title). Optional.
    extra_headers: Vec<(String, String)>,
}

impl OpenAiCompat {
    pub fn new(
        name: impl Into<String>,
        base_url: impl Into<String>,
        api_key: Option<String>,
    ) -> Self {
        Self {
            name: name.into(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key,
            http: reqwest::Client::new(),
            extra_headers: Vec::new(),
        }
    }

    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_headers.push((key.into(), value.into()));
        self
    }

    fn endpoint(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }
}

// ── Wire serialization ───────────────────────────────────────────────────────

#[derive(Serialize)]
struct WireMessage<'a> {
    role: &'a str,
    content: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<&'a str>,
}

fn role_str(r: Role) -> &'static str {
    match r {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

fn message_to_wire(m: &Message) -> WireMessage<'_> {
    let tool_calls = if m.tool_calls.is_empty() {
        None
    } else {
        Some(
            m.tool_calls
                .iter()
                .map(|tc| {
                    json!({
                        "id": tc.id,
                        "type": "function",
                        "function": { "name": tc.name, "arguments": tc.arguments },
                    })
                })
                .collect(),
        )
    };
    WireMessage {
        role: role_str(m.role),
        content: &m.content,
        tool_calls,
        tool_call_id: m.tool_call_id.as_deref(),
    }
}

fn tool_to_wire(t: &ToolDef) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        },
    })
}

fn build_body(req: &ChatRequest) -> Value {
    let messages: Vec<WireMessage> = req.messages.iter().map(message_to_wire).collect();
    let mut body = json!({
        "model": req.model,
        "messages": messages,
        "stream": true,
    });
    if !req.tools.is_empty() {
        body["tools"] = Value::Array(req.tools.iter().map(tool_to_wire).collect());
    }
    if let Some(t) = req.temperature {
        body["temperature"] = json!(t);
    }
    if let Some(m) = req.max_tokens {
        body["max_tokens"] = json!(m);
    }
    body
}

// ── Streaming chunk parsing ──────────────────────────────────────────────────

#[derive(Deserialize)]
struct ChunkChoice {
    #[serde(default)]
    delta: Delta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize, Default)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<DeltaToolCall>>,
}

#[derive(Deserialize)]
struct DeltaToolCall {
    #[serde(default)]
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<DeltaFunction>,
}

#[derive(Deserialize)]
struct DeltaFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Deserialize)]
struct ChunkResponse {
    #[serde(default)]
    choices: Vec<ChunkChoice>,
}

/// Parse one SSE `data:` payload into zero or more [`StreamEvent`]s.
fn parse_chunk(data: &str) -> Result<Vec<StreamEvent>> {
    let chunk: ChunkResponse =
        serde_json::from_str(data).with_context(|| format!("decoding stream chunk: {data}"))?;
    let mut events = Vec::new();
    for choice in chunk.choices {
        if let Some(text) = choice.delta.content {
            if !text.is_empty() {
                events.push(StreamEvent::TextDelta(text));
            }
        }
        if let Some(calls) = choice.delta.tool_calls {
            for c in calls {
                let (name, args) = match c.function {
                    Some(f) => (f.name, f.arguments.unwrap_or_default()),
                    None => (None, String::new()),
                };
                events.push(StreamEvent::ToolCallDelta {
                    index: c.index,
                    id: c.id,
                    name,
                    arguments_delta: args,
                });
            }
        }
        if let Some(reason) = choice.finish_reason {
            events.push(StreamEvent::Done { finish_reason: Some(reason) });
        }
    }
    Ok(events)
}

#[async_trait]
impl LlmProvider for OpenAiCompat {
    fn name(&self) -> &str {
        &self.name
    }

    async fn stream_chat(
        &self,
        req: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamEvent>>> {
        let body = build_body(&req);
        let mut builder = self
            .http
            .post(self.endpoint())
            .header("content-type", "application/json")
            .header("accept", "text/event-stream");

        if let Some(key) = &self.api_key {
            builder = builder.bearer_auth(key);
        }
        for (k, v) in &self.extra_headers {
            builder = builder.header(k.as_str(), v.as_str());
        }

        let resp = builder
            .json(&body)
            .send()
            .await
            .with_context(|| format!("connecting to provider '{}'", self.name))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!(
                "provider '{}' returned HTTP {}: {}",
                self.name,
                status,
                text.trim()
            ));
        }

        // Convert the byte stream into SSE events, then into StreamEvents.
        let event_stream = resp.bytes_stream().eventsource();
        let mapped = event_stream.flat_map(|item| {
            let events: Vec<Result<StreamEvent>> = match item {
                Ok(ev) => {
                    let data = ev.data;
                    if data.trim() == "[DONE]" {
                        vec![Ok(StreamEvent::Done { finish_reason: None })]
                    } else {
                        match parse_chunk(&data) {
                            Ok(evs) => evs.into_iter().map(Ok).collect(),
                            Err(e) => vec![Err(e)],
                        }
                    }
                }
                Err(e) => vec![Err(anyhow!("SSE stream error: {e}"))],
            };
            futures_util::stream::iter(events)
        });

        Ok(mapped.boxed())
    }
}

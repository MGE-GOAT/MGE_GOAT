//! LLM provider abstraction.
//!
//! Everything the agent loop needs to talk to a model lives here: the message
//! model, tool-call types, streaming events, and the [`LlmProvider`] trait.
//! Concrete backends (currently OpenAI-compatible) live in submodules.

pub mod openai_compat;

use anyhow::Result;
use async_trait::async_trait;
use futures_util::stream::BoxStream;
use serde::{Deserialize, Serialize};

/// Conversation role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// A function/tool the model is allowed to call. `parameters` is a JSON Schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// A tool invocation requested by the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Raw JSON arguments string as produced by the model.
    pub arguments: String,
}

/// One message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    #[serde(default)]
    pub content: String,
    /// Present on assistant messages that requested tool calls.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// Set on `tool` role messages: which call this result answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: Role::System, content: content.into(), tool_calls: vec![], tool_call_id: None }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: Role::User, content: content.into(), tool_calls: vec![], tool_call_id: None }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: content.into(), tool_calls: vec![], tool_call_id: None }
    }
    /// A tool-result message answering a specific tool call.
    pub fn tool_result(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            tool_calls: vec![],
            tool_call_id: Some(call_id.into()),
        }
    }
}

/// A request to generate a (streamed) completion.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDef>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
}

impl ChatRequest {
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self { model: model.into(), messages, tools: vec![], temperature: None, max_tokens: None }
    }
    pub fn with_tools(mut self, tools: Vec<ToolDef>) -> Self {
        self.tools = tools;
        self
    }
}

/// Incremental events emitted while a completion streams in.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// A chunk of assistant text.
    TextDelta(String),
    /// A streamed fragment of a tool call. `index` ties fragments together;
    /// `id`/`name` arrive on the first fragment, `arguments_delta` accumulates.
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments_delta: String,
    },
    /// Generation finished. `finish_reason` is the provider's reason, if given.
    Done { finish_reason: Option<String> },
}

/// A backend capable of producing streamed chat completions.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Human-readable provider name (for logs/UI).
    fn name(&self) -> &str;

    /// Stream a completion. Errors surface both at call time (connection/HTTP)
    /// and per-event (mid-stream decode failures).
    async fn stream_chat(
        &self,
        req: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamEvent>>>;
}

/// Accumulates [`StreamEvent`]s into a finished [`Message`]. Used by callers that
/// want the whole assistant turn (text + any tool calls) after the stream ends.
#[derive(Debug, Default)]
pub struct TurnAccumulator {
    text: String,
    /// Indexed tool-call builders: (id, name, args).
    calls: Vec<(String, String, String)>,
    pub finish_reason: Option<String>,
}

impl TurnAccumulator {
    pub fn push(&mut self, ev: &StreamEvent) {
        match ev {
            StreamEvent::TextDelta(t) => self.text.push_str(t),
            StreamEvent::ToolCallDelta { index, id, name, arguments_delta } => {
                if *index >= self.calls.len() {
                    self.calls.resize(*index + 1, (String::new(), String::new(), String::new()));
                }
                let slot = &mut self.calls[*index];
                if let Some(id) = id {
                    slot.0 = id.clone();
                }
                if let Some(name) = name {
                    slot.1 = name.clone();
                }
                slot.2.push_str(arguments_delta);
            }
            StreamEvent::Done { finish_reason } => {
                self.finish_reason = finish_reason.clone();
            }
        }
    }

    /// Finalize into an assistant message.
    pub fn into_message(self) -> Message {
        let tool_calls = self
            .calls
            .into_iter()
            .filter(|(_, name, _)| !name.is_empty())
            .map(|(id, name, arguments)| ToolCall { id, name, arguments })
            .collect();
        Message { role: Role::Assistant, content: self.text, tool_calls, tool_call_id: None }
    }
}

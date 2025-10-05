use once_cell::sync::Lazy;

const SYSTEM_PROMPT: Lazy<String> = Lazy::new(|| {
    std::env::var("SYSTEM_PROMPT").unwrap_or_else(|_| "You are a helpful assistant.".to_string())
});

const DEV_PROMPT: Lazy<String> =
    Lazy::new(|| std::env::var("DEV_PROMPT").unwrap_or_else(|_| "".to_string()));

const CONTEXT_K: usize = 200;

pub mod llama;
pub mod openrouter;

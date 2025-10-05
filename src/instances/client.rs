use std::sync::Arc;

use once_cell::sync::Lazy;
use openrouter_rs::OpenRouterClient;

#[allow(dead_code)]
static CLIENT_INSTANCE: Lazy<Arc<OpenRouterClient>> = Lazy::new(|| {
    let api_key = std::env::var("OPENROUTER_TOKEN")
        .expect("OPENROUTER_TOKEN must be set in .env file or environment variables");

    let client = OpenRouterClient::builder()
        .api_key(&api_key)
        .build()
        .expect("OpenRouter client intialization");

    Arc::new(client)
});

// Helper function to access the DB
#[allow(dead_code)]
pub fn get_or_client() -> Arc<OpenRouterClient> {
    CLIENT_INSTANCE.clone()
}

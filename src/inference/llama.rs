use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_batch::LlamaBatch,
    model::{AddBos, Special},
    sampling::LlamaSampler,
};
use once_cell::sync::Lazy;
use std::io::Write;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::{
    conversation::{Conversation, DBMessage},
    inference::{CONTEXT_K, DEV_PROMPT, SYSTEM_PROMPT},
    instances::{llama::get_llama, rocksdb::get_rocks_db},
};

#[allow(dead_code)]
pub const MODEL_LOCATION: Lazy<String> = Lazy::new(|| {
    std::env::var("LLAMA_MODEL_PATH")
        .expect("LLAMA_MODEL_PATH must be set in .env file or environment variables")
});

#[allow(dead_code)]
pub async fn infer_and_print(
    convo_id: u32,
    prompt: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let db = get_rocks_db();
    let (backend, model) = get_llama();

    // Colored output: cyan for user, green for AI
    println!("\x1b[36mUSER:\x1b[0m {prompt}");
    print!("\x1b[32mAI:\x1b[0m ");
    std::io::stdout().flush()?;

    // Start loading animation
    let loading = Arc::new(AtomicBool::new(true));
    let loading_clone = loading.clone();

    let loading_task = tokio::spawn(async move {
        let dots = ["   ", ".  ", ".. ", "..."];
        let mut idx = 0;
        while loading_clone.load(Ordering::Relaxed) {
            print!("\r\x1b[32mAI:\x1b[0m {}", dots[idx]);
            std::io::stdout().flush().ok();
            idx = (idx + 1) % dots.len();
            tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;
        }
        print!("\r\x1b[32mAI:\x1b[0m ");
        std::io::stdout().flush().ok();
    });

    let convo = Conversation::new(convo_id);

    // Build the full prompt with conversation history
    let full_prompt = convo.build_llama_prompt(
        &db,
        SYSTEM_PROMPT.as_str(),
        DEV_PROMPT.as_str(),
        prompt,
        Some(CONTEXT_K),
    )?;

    // Initialize context with GPU support
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(4096).unwrap()))
        .with_n_threads(8)
        .with_n_threads_batch(8);

    let mut ctx = model.new_context(&backend, ctx_params)?;

    // Tokenize the prompt
    let tokens_list = model.str_to_token(&full_prompt, AddBos::Always)?;

    let n_ctx = ctx.n_ctx() as i32;
    let n_len = 2048; // max tokens to generate
    let n_kv_req = tokens_list.len() as i32 + n_len;

    if n_kv_req > n_ctx {
        return Err("Context size too small for prompt + generation".into());
    }

    // Create batch and add tokens
    let mut batch = LlamaBatch::new(512, 1);
    let last_index = (tokens_list.len() - 1) as i32;

    for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
        let is_last = i == last_index;
        batch.add(token, i, &[0], is_last)?;
    }

    ctx.decode(&mut batch)?;

    // Generate response
    let mut n_cur = batch.n_tokens();
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut sampler =
        LlamaSampler::chain_simple([LlamaSampler::dist(1234), LlamaSampler::greedy()]);

    let mut assistant_result = String::new();

    while n_cur <= n_len {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        // Check for end of generation
        if model.is_eog_token(token) {
            println!("\x1b[0m"); // Reset color
            break;
        }

        // Stop loading animation on first token
        if n_cur == batch.n_tokens() {
            loading.store(false, Ordering::Relaxed);
        }

        // Decode token to string
        let output_bytes = model.token_to_bytes(token, Special::Tokenize)?;
        let mut output_string = String::with_capacity(32);
        let (_, _, _) = decoder.decode_to_string(&output_bytes, &mut output_string, false);

        print!("\x1b[32m{}\x1b[0m", output_string); // Green color for AI output
        std::io::stdout().flush()?;

        assistant_result.push_str(&output_string);

        // Prepare next iteration
        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        n_cur += 1;

        ctx.decode(&mut batch)?;
    }

    loading.store(false, Ordering::Relaxed);
    loading_task.abort();

    // Save the conversation
    convo.save_message(
        &db,
        &DBMessage {
            author_type: 0,
            content: prompt.to_string(),
        },
    )?;

    convo.save_message(
        &db,
        &DBMessage {
            author_type: 1,
            content: assistant_result,
        },
    )?;

    Ok(())
}

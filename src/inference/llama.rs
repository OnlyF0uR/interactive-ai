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

    println!("\n┌─ \x1b[36;1mYou\x1b[0m");
    println!("│  {}", prompt);
    println!("└─");
    println!("┌─ \x1b[32;1mAI\x1b[0m");
    print!("│ ");
    std::io::stdout().flush()?;

    // Start loading animation
    let loading = Arc::new(AtomicBool::new(true));
    let loading_clone = loading.clone();

    let loading_task = tokio::spawn(async move {
        while loading_clone.load(Ordering::Relaxed) {
            print!("."); // Print dots sequentially
            std::io::stdout().flush().ok();
            tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;
        }
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
        .with_n_ctx(Some(NonZeroU32::new(8192).unwrap()))
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
    let mut batch = LlamaBatch::new(2048, 1);
    let last_index = (tokens_list.len() - 1) as i32;

    for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
        let is_last = i == last_index;
        batch.add(token, i, &[0], is_last)?;
    }

    ctx.decode(&mut batch)?;

    // Generate response
    let mut n_cur = batch.n_tokens();
    let mut decoder = encoding_rs::UTF_8.new_decoder();

    // Add temperature and other sampling parameters
    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::temp(0.7),
        LlamaSampler::top_k(40),
        LlamaSampler::top_p(0.95, 1),
        LlamaSampler::dist(1234),
    ]);

    let mut assistant_result = String::new();
    let mut printed_chars = 0; // Track how many characters we've printed
    let mut first_token = true;
    let mut last_was_space = false;
    let mut loading_task_handle = Some(loading_task);

    // Define stop sequences
    let stop_sequences = vec![
        "<|im_end|>",
        "<|im_start|>",
        "User:",
        "\nUser:",
        "\n\nUser:",
    ];

    let max_stop_len = stop_sequences
        .iter()
        .map(|s| s.chars().count())
        .max()
        .unwrap_or(0);

    while n_cur <= n_len {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        // Check for end of generation
        if model.is_eog_token(token) {
            break;
        }

        // Stop loading animation on first token
        if first_token {
            loading.store(false, Ordering::Relaxed);
            // Wait for loading task to finish
            if let Some(task) = loading_task_handle.take() {
                let _ = task.await;
            }
            first_token = false;
        }

        // Decode token to string
        let output_bytes = model.token_to_bytes(token, Special::Tokenize)?;
        let mut output_string = String::with_capacity(32);
        let (_, _, _) = decoder.decode_to_string(&output_bytes, &mut output_string, false);

        // Add to accumulated result
        assistant_result.push_str(&output_string);

        // Check if any stop sequence is present
        let mut found_stop = false;
        for seq in &stop_sequences {
            if let Some(pos) = assistant_result.find(seq) {
                // Truncate result at stop sequence (pos is already byte-aligned from find())
                assistant_result.truncate(pos);
                found_stop = true;
                break;
            }
        }

        if found_stop {
            break;
        }

        // Calculate how many characters are safe to print (leave buffer for potential stop sequences)
        let total_chars = assistant_result.chars().count();
        let safe_chars = if total_chars > max_stop_len {
            total_chars - max_stop_len
        } else {
            0
        };

        // Print only the safe portion we haven't printed yet
        if safe_chars > printed_chars {
            // Collect the characters to print (from printed_chars to safe_chars)
            let chars_to_print: String = assistant_result
                .chars()
                .skip(printed_chars)
                .take(safe_chars - printed_chars)
                .collect();

            for ch in chars_to_print.chars() {
                if ch == '\n' {
                    print!("\n│ ");
                    last_was_space = false;
                } else if ch == ' ' {
                    if !last_was_space {
                        print!("{}", ch);
                    }
                    last_was_space = true;
                } else {
                    print!("{}", ch);
                    last_was_space = false;
                }
            }
            std::io::stdout().flush()?;
            printed_chars = safe_chars;
        }

        // Prepare next iteration
        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        n_cur += 1;

        ctx.decode(&mut batch)?;
    }

    // Print any remaining text that wasn't printed (the buffered portion)
    let total_chars = assistant_result.chars().count();
    if printed_chars < total_chars {
        let remaining_chars: String = assistant_result.chars().skip(printed_chars).collect();

        for ch in remaining_chars.chars() {
            if ch == '\n' {
                print!("\n│ ");
                last_was_space = false;
            } else if ch == ' ' {
                if !last_was_space {
                    print!("{}", ch);
                }
                last_was_space = true;
            } else {
                print!("{}", ch);
                last_was_space = false;
            }
        }
        std::io::stdout().flush()?;
    }

    loading.store(false, Ordering::Relaxed);

    println!("\n└─");

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

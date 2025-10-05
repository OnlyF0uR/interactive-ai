use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use futures_util::StreamExt;
use once_cell::sync::Lazy;
use openrouter_rs::{Message, api::chat::ChatCompletionRequest, types::Role};

use crate::{
    conversation::{Conversation, DBMessage},
    inference::{CONTEXT_K, DEV_PROMPT, SYSTEM_PROMPT},
    instances::{openrouter::get_or_client, rocksdb::get_rocks_db},
};

#[allow(dead_code)]
const OPENROUTER_MODEL: Lazy<String> = Lazy::new(|| {
    std::env::var("OPENROUTER_MODEL")
        .expect("OPENROUTER_MODEL must be set in .env file or environment variables")
});

#[allow(dead_code)]
pub async fn infer_and_print(
    convo_id: u32,
    prompt: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let db = get_rocks_db();
    let client = get_or_client();

    println!("\n┌─ \x1b[36;1mYou\x1b[0m");
    println!("│  {}", prompt);
    println!("└─");
    println!("┌─ \x1b[32;1mAI\x1b[0m");
    print!("│");
    std::io::stdout().flush()?;

    // Start loading animation
    let loading = Arc::new(AtomicBool::new(true));
    let loading_clone = loading.clone();

    let loading_task = tokio::spawn(async move {
        let dots = ["   ", ".  ", ".. ", "..."];
        let mut idx = 0;
        while loading_clone.load(Ordering::Relaxed) {
            print!("\r│ {}", dots[idx]);
            std::io::stdout().flush().ok();
            idx = (idx + 1) % dots.len();
            tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;
        }
        // Clear the dots completely - overwrite with spaces then reset cursor
        print!("\r│     \r│ ");
        std::io::stdout().flush().ok();
    });

    let convo = Conversation::new(convo_id);

    // Get the old messages
    let mut messages = vec![
        Message::new(Role::System, SYSTEM_PROMPT.as_str()),
        Message::new(Role::Developer, DEV_PROMPT.as_str()),
    ];
    // Add the old messages to context
    convo.add_openrouter_messages(&db, &mut messages, Some(CONTEXT_K))?;

    // Add the new user prompt
    let user_prompt = Message::new(Role::User, prompt);
    messages.push(user_prompt);

    let request = ChatCompletionRequest::builder()
        .model(OPENROUTER_MODEL.as_str())
        .messages(messages)
        .build()?;

    // The final result
    let stream = client.stream_chat_completion(&request).await?;

    let loading_clone2 = loading.clone();
    let assistent_result = stream
        .filter_map(|event| async { event.ok() })
        .fold(
            (String::new(), true, false),
            |(mut acc, mut first_chunk, mut last_was_space), chunk| {
                let loading_ref = loading_clone2.clone();
                async move {
                    if first_chunk {
                        loading_ref.store(false, Ordering::Relaxed);
                        // Wait a tiny bit for loading task to clear
                        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                        first_chunk = false;
                    }
                    if let Some(content) = chunk.choices[0].content() {
                        // Handle newlines and spaces properly
                        for ch in content.chars() {
                            if ch == '\n' {
                                print!("\n│ ");
                                std::io::stdout().flush().ok();
                                last_was_space = false;
                            } else if ch == ' ' {
                                if !last_was_space {
                                    print!("{}", ch);
                                    std::io::stdout().flush().ok();
                                    last_was_space = true;
                                }
                            } else {
                                print!("{}", ch);
                                std::io::stdout().flush().ok();
                                last_was_space = false;
                            }
                        }
                        acc.push_str(content);
                    }
                    (acc, first_chunk, last_was_space)
                }
            },
        )
        .await
        .0;

    loading.store(false, Ordering::Relaxed);
    let _ = loading_task.await;

    println!("\n└─");

    // Debug printing
    // println!("{}", assistent_result);

    // If all is okay we save the original message
    convo.save_message(
        &db,
        &DBMessage {
            author_type: 0,
            content: prompt.to_string(),
        },
    )?;
    // And the messages from the llm
    convo.save_message(
        &db,
        &DBMessage {
            author_type: 1,
            content: assistent_result,
        },
    )?;

    Ok(())
}

mod conversation;
mod inference;
mod instances;

use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv()?;

    println!("Interactive chat mode. Type 'exit' or 'quit' to end the conversation.\n");

    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut user_prompt = String::new();
        io::stdin().read_line(&mut user_prompt)?;

        let user_prompt = user_prompt.trim();

        if user_prompt.is_empty() {
            continue;
        }

        if user_prompt.eq_ignore_ascii_case("exit") || user_prompt.eq_ignore_ascii_case("quit") {
            println!("Goodbye!");
            break;
        }

        inference::openrouter::infer_and_print(0, &user_prompt).await?;
        // inference::llama::infer_and_print(0, &user_prompt).await?;

        println!(); // Add spacing between exchanges
    }

    Ok(())
}

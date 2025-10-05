use bincode::{Decode, Encode};
use openrouter_rs::Message;

#[derive(Debug)]
pub struct Conversation {
    id: u32,
}

#[derive(Debug, Encode, Decode)]
pub struct DBMessage {
    pub author_type: u8, // 0 for user, 1 for assistent
    pub content: String,
}

impl Conversation {
    pub fn new(convo_id: u32) -> Self {
        Self { id: convo_id }
    }

    pub fn save_message(
        &self,
        db: &rocksdb::DB,
        msg: &DBMessage,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros();
        let key = format!("message:{}:{}", self.id, now);

        let config = bincode::config::standard();
        let value = bincode::encode_to_vec(msg, config)?;
        db.put(key.as_bytes(), &value)?;
        Ok(())
    }

    // Function to load all sorting by timestamp which is part of the key
    // where the newest message is last
    #[allow(dead_code)]
    pub fn add_openrouter_messages(
        &self,
        db: &rocksdb::DB,
        buffer: &mut Vec<Message>,
        last_k: Option<usize>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let prefix = format!("message:{}:", self.id);
        let iter = db.prefix_iterator(prefix.as_bytes());
        let mut messages = Vec::new();

        let config = bincode::config::standard();

        for item in iter {
            let (key, value) = item?;
            let key_str = String::from_utf8_lossy(&key);
            // Extract timestamp from key format "message:{id}:{timestamp}"
            let timestamp: u128 = key_str
                .split(':')
                .nth(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);

            let (msg, _): (DBMessage, _) = bincode::decode_from_slice(&value, config)?;
            messages.push((timestamp, msg));
        }

        // Sort by timestamp (oldest first, newest last)
        messages.sort_by_key(|(ts, _)| *ts);

        // Apply last_k limit if specified
        let messages_to_process = if let Some(k) = last_k {
            if messages.len() > k {
                &messages[messages.len() - k..]
            } else {
                &messages[..]
            }
        } else {
            &messages[..]
        };

        // Parse the messages for open router and append to buffer
        for (_, m) in messages_to_process {
            let role = if m.author_type == 0 {
                openrouter_rs::types::Role::User
            } else {
                openrouter_rs::types::Role::Assistant
            };
            buffer.push(Message::new(role, &m.content));
        }

        Ok(())
    }

    // Format messages for llama-cpp context
    #[allow(dead_code)]
    pub fn build_llama_prompt(
        &self,
        db: &rocksdb::DB,
        system_prompt: &str,
        dev_prompt: &str,
        user_prompt: &str,
        last_k: Option<usize>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let mut prompt = String::new();

        // Add system and developer prompts
        prompt.push_str(&format!("System: {}\n\n", system_prompt));
        prompt.push_str(&format!("Developer: {}\n\n", dev_prompt));

        // Load conversation history
        let prefix = format!("message:{}:", self.id);
        let iter = db.prefix_iterator(prefix.as_bytes());
        let mut messages = Vec::new();

        let config = bincode::config::standard();

        for item in iter {
            let (key, value) = item?;
            let key_str = String::from_utf8_lossy(&key);
            let timestamp: u128 = key_str
                .split(':')
                .nth(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);

            let (msg, _): (DBMessage, _) = bincode::decode_from_slice(&value, config)?;
            messages.push((timestamp, msg));
        }

        // Sort by timestamp (oldest first, newest last)
        messages.sort_by_key(|(ts, _)| *ts);

        // Apply last_k limit if specified
        let messages_to_process = if let Some(k) = last_k {
            if messages.len() > k {
                &messages[messages.len() - k..]
            } else {
                &messages[..]
            }
        } else {
            &messages[..]
        };

        // Format conversation history
        for (_, m) in messages_to_process {
            let role = if m.author_type == 0 {
                "User"
            } else {
                "Assistant"
            };
            prompt.push_str(&format!("{}: {}\n\n", role, m.content));
        }

        // Add current user prompt
        prompt.push_str(&format!("User: {}\n\nAssistant:", user_prompt));

        Ok(prompt)
    }
}

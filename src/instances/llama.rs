use llama_cpp_2::{
    llama_backend::LlamaBackend,
    model::{LlamaModel, params::LlamaModelParams},
};
use once_cell::sync::Lazy;
use std::sync::Arc;

use crate::inference;

#[allow(dead_code)]
pub static LLAMA_INSTANCE: Lazy<(Arc<LlamaBackend>, Arc<LlamaModel>)> = Lazy::new(|| {
    // Initialize backend
    let backend = Arc::new(LlamaBackend::init().expect("Failed to initialize llama backend"));

    // Configure parameters with GPU offloading - ensure all layers go to GPU
    let params = LlamaModelParams::default()
        .with_n_gpu_layers(1000)
        .with_use_mlock(false);

    // Load model using the same backend
    let model = Arc::new(
        LlamaModel::load_from_file(&backend, inference::llama::MODEL_LOCATION.as_str(), &params)
            .expect("Failed to load model"),
    );

    (backend, model)
});

#[allow(dead_code)]
pub fn get_llama() -> (Arc<LlamaBackend>, Arc<LlamaModel>) {
    LLAMA_INSTANCE.clone()
}

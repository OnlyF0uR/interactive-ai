use once_cell::sync::Lazy;
use rocksdb::{DB, Options};
use std::sync::Arc;

// Configure RocksDB options
fn rocksdb_options() -> Options {
    let mut opts = Options::default();
    opts.create_if_missing(true);
    opts
}

// Global RocksDB instance wrapped in Arc
static DB_INSTANCE: Lazy<Arc<DB>> = Lazy::new(|| {
    let db = DB::open(&rocksdb_options(), "users_db").expect("Failed to open RocksDB");
    Arc::new(db)
});

// Helper function to access the DB
pub fn get_rocks_db() -> Arc<DB> {
    DB_INSTANCE.clone()
}

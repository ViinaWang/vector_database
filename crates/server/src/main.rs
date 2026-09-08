//! vectordb-server: 数据库目录 + HTTP。

// 入口处失败直接退出比传播错误更合适。
#![allow(clippy::expect_used)]

use vectordb_core::Database;
use vectordb_server::api;

fn main() {
    let path = std::env::var("VDB_PATH").unwrap_or_else(|_| "./data".into());
    let addr = std::env::var("VDB_ADDR").unwrap_or_else(|_| "127.0.0.1:7280".into());

    let db = match Database::open(&path) {
        Ok(db) => db,
        Err(e) => {
            eprintln!("failed to open database at {path}: {e}");
            std::process::exit(1);
        }
    };
    if !db.recovery_clean() {
        eprintln!("warning: wal had a corrupt tail at open; it was truncated");
    }

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(&addr).await.expect("bind");
        eprintln!("vectordb-server listening on http://{addr} (data: {path})");
        axum::serve(listener, api::router(db)).await.expect("serve");
    });
}

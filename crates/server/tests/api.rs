#![allow(clippy::unwrap_used)]

//! REST API 集成测试: `router` + `ServiceExt::oneshot`，不起真实端口。

use axum::Router;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tempfile::{TempDir, tempdir};
use tower::ServiceExt;
use vectordb_core::Database;
use vectordb_server::api::router;

fn app(dir: &TempDir) -> Router {
    router(Database::open(dir.path()).unwrap())
}

/// 发一个请求，返回状态码与解析后的响应体（非 JSON 时退化为字符串）。
async fn send(app: &Router, method: Method, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    let mut builder = Request::builder().method(method).uri(uri);
    let body = match body {
        Some(v) => {
            builder = builder.header("content-type", "application/json");
            Body::from(v.to_string())
        }
        None => Body::empty(),
    };
    let resp = app
        .clone()
        .oneshot(builder.body(body).unwrap())
        .await
        .unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes)
            .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()))
    };
    (status, v)
}

async fn get(app: &Router, uri: &str) -> (StatusCode, Value) {
    send(app, Method::GET, uri, None).await
}

async fn delete(app: &Router, uri: &str) -> (StatusCode, Value) {
    send(app, Method::DELETE, uri, None).await
}

async fn post(app: &Router, uri: &str, body: Value) -> (StatusCode, Value) {
    send(app, Method::POST, uri, Some(body)).await
}

async fn put(app: &Router, uri: &str, body: Value) -> (StatusCode, Value) {
    send(app, Method::PUT, uri, Some(body)).await
}

// ---------------------------------------------------------------- 测试

#[tokio::test]
async fn healthz_reports_ok() {
    let dir = tempdir().unwrap();
    let app = app(&dir);

    let (status, body) = get(&app, "/healthz").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["ok"], json!(true));
    assert_eq!(body["recovery_clean"], json!(true));
    assert_eq!(body["collections"], json!([]));
}

#[tokio::test]
async fn create_collection_index_config_and_defaults() {
    let dir = tempdir().unwrap();
    let app = app(&dir);

    let (status, body) = post(
        &app,
        "/collections",
        json!({"name": "flat3", "dim": 4, "metric": "l2", "index": {"kind": "flat"}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["name"], json!("flat3"));
    assert_eq!(body["dim"], json!(4));
    assert_eq!(body["metric"], json!("l2"));
    assert_eq!(body["index"], json!("flat"));
    assert_eq!(body["points"], json!(0));

    // 缺省: hnsw 索引 + cosine 度量
    let (status, body) = post(&app, "/collections", json!({"name": "def3", "dim": 3})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["index"], json!("hnsw"));
    assert_eq!(body["metric"], json!("cosine"));

    let (status, body) = get(&app, "/collections/flat3").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["name"], json!("flat3"));
    assert_eq!(body["index"], json!("flat"));
    assert_eq!(body["points"], json!(0));

    let (status, body) = get(&app, "/collections").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["collections"], json!(["def3", "flat3"]));
}

#[tokio::test]
async fn duplicate_create_conflicts_then_drop_allows_recreate() {
    let dir = tempdir().unwrap();
    let app = app(&dir);

    let (status, _) = post(&app, "/collections", json!({"name": "dup", "dim": 2})).await;
    assert_eq!(status, StatusCode::OK);

    let (status, body) = post(&app, "/collections", json!({"name": "dup", "dim": 2})).await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert!(body.to_string().contains("already exists"));

    let (status, body) = delete(&app, "/collections/dup").await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    assert_eq!(body, json!(null));

    let (status, _) = get(&app, "/collections/dup").await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let (status, _) = post(&app, "/collections", json!({"name": "dup", "dim": 2})).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn missing_collection_is_404() {
    let dir = tempdir().unwrap();
    let app = app(&dir);

    let (status, _) = get(&app, "/collections/nope").await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let (status, _) = delete(&app, "/collections/nope").await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let (status, _) = put(
        &app,
        "/collections/nope/points",
        json!([{"id": 1, "vector": [1.0]}]),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let (status, _) = post(&app, "/collections/nope/points:get", json!({"ids": [1]})).await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let (status, _) = post(&app, "/collections/nope/query", json!({"vector": [1.0]})).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn points_full_lifecycle() {
    let dir = tempdir().unwrap();
    let app = app(&dir);

    let (status, _) = post(
        &app,
        "/collections",
        json!({"name": "docs", "dim": 3, "index": {"kind": "flat"}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let (status, body) = put(
        &app,
        "/collections/docs/points",
        json!([
            {"id": 1, "vector": [1.0, 0.0, 0.0], "payload": {"lang": "en", "year": 2023}},
            {"id": 2, "vector": [0.0, 1.0, 0.0], "payload": {"lang": "zh", "year": 2022}},
            {"id": "s3", "vector": [0.9, 0.1, 0.0], "payload": {"lang": "en", "year": 2024}},
        ]),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["upserted"], json!(3));

    let (status, body) = get(&app, "/collections/docs").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["points"], json!(3));

    // get: 顺序与入参一致，缺失为 null
    let (status, body) = post(
        &app,
        "/collections/docs/points:get",
        json!({"ids": [1, "s3", 99]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["points"][0]["id"], json!(1));
    assert_eq!(
        body["points"][0]["payload"],
        json!({"lang": "en", "year": 2023})
    );
    assert_eq!(body["points"][1]["id"], json!("s3"));
    assert_eq!(body["points"][2], json!(null));

    // query by vector
    let (status, body) = post(
        &app,
        "/collections/docs/query",
        json!({"vector": [1.0, 0.0, 0.0], "top_k": 2}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let hits = &body["points"];
    assert_eq!(hits.as_array().unwrap().len(), 2);
    assert_eq!(hits[0]["id"], json!(1));
    assert!(hits[0]["score"].as_f64().unwrap() > hits[1]["score"].as_f64().unwrap());
    assert_eq!(hits[0]["payload"], json!({"lang": "en", "year": 2023}));
    assert_eq!(hits[0]["vector"], json!(null)); // with_vector 缺省 false

    // query 带 filter
    let (status, body) = post(
        &app,
        "/collections/docs/query",
        json!({"vector": [1.0, 0.0, 0.0], "filter": {"field": "lang", "op": {"kind": "match", "value": "zh"}}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["points"].as_array().unwrap().len(), 1);
    assert_eq!(body["points"][0]["id"], json!(2));

    // query by id（自身排除）
    let (status, body) = post(
        &app,
        "/collections/docs/query",
        json!({"id": 1, "top_k": 1}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["points"][0]["id"], json!("s3"));

    // scroll 分页
    let (status, body) = post(&app, "/collections/docs/scroll", json!({"limit": 2})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["points"].as_array().unwrap().len(), 2);
    assert_eq!(body["next_offset"], json!(2));
    let (status, body) = post(
        &app,
        "/collections/docs/scroll",
        json!({"offset": 2, "limit": 2}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["points"].as_array().unwrap().len(), 1);
    assert_eq!(body["points"][0]["id"], json!("s3"));
    assert_eq!(body["next_offset"], json!(null));

    // payload:set 整体覆盖
    let (status, body) = post(
        &app,
        "/collections/docs/payload:set",
        json!({"ids": [1], "payload": {"lang": "en", "tag": "x"}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["updated"], json!(1));
    let (_, body) = post(&app, "/collections/docs/points:get", json!({"ids": [1]})).await;
    assert_eq!(
        body["points"][0]["payload"],
        json!({"lang": "en", "tag": "x"})
    );

    // payload:clear
    let (status, body) = post(
        &app,
        "/collections/docs/payload:clear",
        json!({"ids": ["s3"]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["updated"], json!(1));
    let (_, body) = post(&app, "/collections/docs/points:get", json!({"ids": ["s3"]})).await;
    assert_eq!(body["points"][0]["payload"], json!(null));

    // delete 返回删除计数
    let (status, body) = post(
        &app,
        "/collections/docs/points:delete",
        json!({"ids": [2, 99]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["deleted"], json!(1));
    let (_, body) = get(&app, "/collections/docs").await;
    assert_eq!(body["points"], json!(2));
    let (_, body) = post(&app, "/collections/docs/points:delete", json!({"ids": [2]})).await;
    assert_eq!(body["deleted"], json!(0));

    // compact 后数据与覆盖层保持
    let (status, body) = post(&app, "/collections/docs/compact", json!({})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["compacted"], json!(true));
    let (_, body) = post(
        &app,
        "/collections/docs/query",
        json!({"vector": [1.0, 0.0, 0.0], "top_k": 10}),
    )
    .await;
    assert_eq!(body["points"].as_array().unwrap().len(), 2);
    assert_eq!(body["points"][0]["id"], json!(1));
    let (_, body) = post(&app, "/collections/docs/points:get", json!({"ids": [1]})).await;
    assert_eq!(
        body["points"][0]["payload"],
        json!({"lang": "en", "tag": "x"})
    );
}

#[tokio::test]
async fn dimension_mismatch_is_400() {
    let dir = tempdir().unwrap();
    let app = app(&dir);

    let (status, _) = post(&app, "/collections", json!({"name": "d2", "dim": 2})).await;
    assert_eq!(status, StatusCode::OK);

    let (status, body) = put(
        &app,
        "/collections/d2/points",
        json!([{"id": 1, "vector": [1.0, 0.0, 0.0]}]),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body.to_string().contains("dimension mismatch"));

    let (status, _) = post(
        &app,
        "/collections/d2/query",
        json!({"vector": [1.0], "top_k": 1}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn flush_is_204_and_data_survives_reopen() {
    let dir = tempdir().unwrap();
    let app = app(&dir);

    let (status, _) = post(&app, "/collections", json!({"name": "docs", "dim": 2})).await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = put(
        &app,
        "/collections/docs/points",
        json!([
            {"id": 1, "vector": [1.0, 0.0]},
            {"id": 2, "vector": [0.0, 1.0]},
        ]),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let (status, body) = post(&app, "/flush", json!({})).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    assert_eq!(body, json!(null));

    // 释放目录锁后重开，段落盘的数据仍在
    drop(app);
    let db = Database::open(dir.path()).unwrap();
    assert_eq!(db.collections(), vec!["docs"]);
    assert_eq!(db.collection("docs").unwrap().count(), 2);
}

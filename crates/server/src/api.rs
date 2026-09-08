//! REST API。端点与库 API 一一对应，无隐藏状态。

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

use vdb::{CollectionConfig, Condition, Database, ExternalId, Metric, Point, Query};
use vectordb_core as vdb;

/// 路由共享的应用状态。
pub struct AppState {
    /// 数据库句柄。
    pub db: Database,
}

/// 构建全部 REST 端点的路由（无监听、无生命周期任务，可自由组合）。
pub fn router(db: Database) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route(
            "/collections",
            get(list_collections).post(create_collection),
        )
        .route(
            "/collections/{name}",
            get(collection_info).delete(drop_collection),
        )
        .route(
            "/collections/{name}/points",
            axum::routing::put(upsert_points),
        )
        .route("/collections/{name}/points:get", post(get_points))
        .route("/collections/{name}/points:delete", post(delete_points))
        .route("/collections/{name}/scroll", post(scroll))
        .route("/collections/{name}/query", post(query))
        .route("/collections/{name}/payload:set", post(set_payload))
        .route("/collections/{name}/payload:clear", post(clear_payload))
        .route("/collections/{name}/compact", post(compact_collection))
        .route("/flush", post(flush))
        .with_state(Arc::new(AppState { db }))
}

// ---------------------------------------------------------------- 错误映射

struct ApiError(vdb::Error);

impl From<vdb::Error> for ApiError {
    fn from(e: vdb::Error) -> Self {
        ApiError(e)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let code = match &self.0 {
            vdb::Error::CollectionNotFound(_) => StatusCode::NOT_FOUND,
            vdb::Error::CollectionExists(_) => StatusCode::CONFLICT,
            vdb::Error::DimensionMismatch { .. }
            | vdb::Error::Invalid(_)
            | vdb::Error::WalCorrupt { .. } => StatusCode::BAD_REQUEST,
            vdb::Error::AlreadyOpen(_) => StatusCode::LOCKED,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (code, self.0.to_string()).into_response()
    }
}

type ApiResult<T> = Result<T, ApiError>;

// ---------------------------------------------------------------- DTO

#[derive(Deserialize)]
struct CreateCollReq {
    name: String,
    dim: usize,
    metric: Option<String>,
    index: Option<vdb::IndexKind>,
}

#[derive(Serialize)]
struct CollInfo {
    name: String,
    dim: usize,
    metric: String,
    index: String,
    points: u64,
}

#[derive(Deserialize)]
struct PointReq {
    id: Value,
    vector: Vec<f32>,
    payload: Option<Value>,
}

#[derive(Deserialize)]
struct IdsReq {
    ids: Vec<Value>,
}

#[derive(Deserialize)]
struct ScrollReq {
    offset: Option<u64>,
    limit: Option<usize>,
    filter: Option<Condition>,
}

#[derive(Deserialize)]
struct QueryReq {
    vector: Option<Vec<f32>>,
    id: Option<Value>,
    top_k: Option<usize>,
    filter: Option<Condition>,
    with_payload: Option<bool>,
    with_vector: Option<bool>,
    score_threshold: Option<f32>,
}

#[derive(Deserialize)]
struct SetPayloadReq {
    ids: Vec<Value>,
    payload: Value,
}

fn parse_metric(s: &str) -> ApiResult<Metric> {
    match s.to_ascii_lowercase().as_str() {
        "l2" => Ok(Metric::L2),
        "cosine" => Ok(Metric::Cosine),
        "dot" => Ok(Metric::Dot),
        other => Err(vdb::Error::Invalid(format!(
            "unknown metric {other:?}, expected l2|cosine|dot"
        ))
        .into()),
    }
}

// IndexKind 无 Display，match 成稳定字符串供展示
fn index_kind_name(kind: &vdb::IndexKind) -> &'static str {
    match kind {
        vdb::IndexKind::Flat => "flat",
        vdb::IndexKind::Hnsw { .. } => "hnsw",
    }
}

fn parse_id(v: &Value) -> ApiResult<ExternalId> {
    match v {
        Value::Number(n) => n
            .as_i64()
            .map(ExternalId::Num)
            .ok_or_else(|| vdb::Error::Invalid(format!("id {v} out of i64 range")).into()),
        Value::String(s) => Ok(ExternalId::Str(s.clone())),
        other => {
            Err(vdb::Error::Invalid(format!("id must be number or string, got {other}")).into())
        }
    }
}

fn coll_of(db: &Database, name: &str) -> ApiResult<vdb::Collection> {
    db.collection(name)
        .ok_or_else(|| vdb::Error::CollectionNotFound(name.into()).into())
}

// ---------------------------------------------------------------- handlers

async fn healthz(State(s): State<Arc<AppState>>) -> Json<Value> {
    Json(serde_json::json!({
        "ok": true,
        "recovery_clean": s.db.recovery_clean(),
        "collections": s.db.collections(),
    }))
}

async fn list_collections(State(s): State<Arc<AppState>>) -> Json<Value> {
    Json(serde_json::json!({ "collections": s.db.collections() }))
}

async fn create_collection(
    State(s): State<Arc<AppState>>,
    Json(req): Json<CreateCollReq>,
) -> ApiResult<Json<CollInfo>> {
    let metric = parse_metric(req.metric.as_deref().unwrap_or("cosine"))?;
    let config = CollectionConfig::new(req.dim, metric)?.with_index(req.index.unwrap_or_default());
    let c = s.db.create_collection(&req.name, config)?;
    Ok(Json(CollInfo {
        name: c.name().into(),
        dim: c.config().dim,
        metric: format!("{:?}", c.config().metric).to_lowercase(),
        index: index_kind_name(&c.config().index).into(),
        points: 0,
    }))
}

async fn collection_info(
    State(s): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> ApiResult<Json<CollInfo>> {
    let c = coll_of(&s.db, &name)?;
    Ok(Json(CollInfo {
        name: name.clone(),
        dim: c.config().dim,
        metric: format!("{:?}", c.config().metric).to_lowercase(),
        index: index_kind_name(&c.config().index).into(),
        points: c.count(),
    }))
}

async fn drop_collection(
    State(s): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> ApiResult<StatusCode> {
    s.db.drop_collection(&name)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn upsert_points(
    State(s): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(points): Json<Vec<PointReq>>,
) -> ApiResult<Json<Value>> {
    let c = coll_of(&s.db, &name)?;
    let pts: Vec<Point> = points
        .iter()
        .map(|p| {
            Ok(Point::new(
                parse_id(&p.id)?,
                p.vector.clone(),
                p.payload.clone(),
            ))
        })
        .collect::<Result<Vec<_>, ApiError>>()?;
    c.upsert(&pts)?;
    Ok(Json(serde_json::json!({ "upserted": pts.len() })))
}

async fn get_points(
    State(s): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<IdsReq>,
) -> ApiResult<Json<Value>> {
    let c = coll_of(&s.db, &name)?;
    let ids: Vec<ExternalId> = req.ids.iter().map(parse_id).collect::<ApiResult<_>>()?;
    let points = c.get(&ids)?;
    Ok(Json(serde_json::json!({ "points": points })))
}

async fn delete_points(
    State(s): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<IdsReq>,
) -> ApiResult<Json<Value>> {
    let c = coll_of(&s.db, &name)?;
    let ids: Vec<ExternalId> = req.ids.iter().map(parse_id).collect::<ApiResult<_>>()?;
    let deleted = c.delete(&ids)?;
    Ok(Json(serde_json::json!({ "deleted": deleted })))
}

async fn scroll(
    State(s): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<ScrollReq>,
) -> ApiResult<Json<Value>> {
    let c = coll_of(&s.db, &name)?;
    let page = c.scroll(
        req.offset.unwrap_or(0),
        req.limit.unwrap_or(10),
        req.filter.as_ref(),
    )?;
    Ok(Json(serde_json::json!({
        "points": page.points,
        "next_offset": page.next_offset,
    })))
}

async fn query(
    State(s): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<QueryReq>,
) -> ApiResult<Json<Value>> {
    let c = coll_of(&s.db, &name)?;
    let q = Query {
        vector: req.vector,
        id: req.id.as_ref().map(parse_id).transpose()?,
        top_k: req.top_k.unwrap_or(10),
        filter: req.filter,
        with_payload: req.with_payload.unwrap_or(true),
        with_vector: req.with_vector.unwrap_or(false),
        score_threshold: req.score_threshold,
    };
    let hits = c.query(&q)?;
    Ok(Json(serde_json::json!({ "points": hits })))
}

async fn set_payload(
    State(s): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<SetPayloadReq>,
) -> ApiResult<Json<Value>> {
    let c = coll_of(&s.db, &name)?;
    let ids: Vec<ExternalId> = req.ids.iter().map(parse_id).collect::<ApiResult<_>>()?;
    let updated = c.set_payload(&ids, &req.payload)?;
    Ok(Json(serde_json::json!({ "updated": updated })))
}

async fn clear_payload(
    State(s): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<IdsReq>,
) -> ApiResult<Json<Value>> {
    let c = coll_of(&s.db, &name)?;
    let ids: Vec<ExternalId> = req.ids.iter().map(parse_id).collect::<ApiResult<_>>()?;
    let updated = c.clear_payload(&ids)?;
    Ok(Json(serde_json::json!({ "updated": updated })))
}

async fn flush(State(s): State<Arc<AppState>>) -> ApiResult<StatusCode> {
    s.db.flush()?;
    Ok(StatusCode::NO_CONTENT)
}

async fn compact_collection(
    State(s): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> ApiResult<Json<Value>> {
    let c = coll_of(&s.db, &name)?;
    c.compact()?;
    Ok(Json(serde_json::json!({ "compacted": true })))
}

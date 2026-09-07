//! vectordb 的 WebAssembly 绑定（内存后端）。
//!
//! 当前形态: 纯内存库，数据生命周期 = 实例生命周期; 需要持久化时用
//! `scroll` 导出、重启用 `upsert` 导回。浏览器 OPFS + Worker 的持久化
//! 形态是路线图上的 M4（见仓库 README）。
//!
//! 接口约定: 复杂参数/返回值走 JSON 字符串，避免跨边界的类型胶水。
//! Point JSON: `{"id": 1 | "a", "vector": [...], "payload": {...}}`。

use wasm_bindgen::prelude::*;

use vdb::{CollectionConfig, Condition, Database, ExternalId, Metric, Point, Query};
use vectordb_core as vdb;

fn js(e: vdb::Error) -> JsError {
    JsError::new(&e.to_string())
}

fn parse_ids(json: &str) -> Result<Vec<ExternalId>, JsError> {
    let ids: Vec<ExternalId> =
        serde_json::from_str(json).map_err(|e| js(vdb::Error::Serde(e.to_string())))?;
    Ok(ids)
}

/// 内存数据库实例。
#[wasm_bindgen]
pub struct WasmDatabase {
    db: Database,
}

#[wasm_bindgen]
impl WasmDatabase {
    /// 新建空内存库。
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WasmDatabase, JsError> {
        Database::open_memory()
            .map(|db| WasmDatabase { db })
            .map_err(js)
    }

    /// 建集合。metric: "l2" | "cosine" | "dot"。
    pub fn create_collection(&self, name: &str, dim: usize, metric: &str) -> Result<(), JsError> {
        let m = match metric.to_ascii_lowercase().as_str() {
            "l2" => Metric::L2,
            "cosine" => Metric::Cosine,
            "dot" => Metric::Dot,
            other => {
                return Err(JsError::new(&format!(
                    "unknown metric {other:?}, expected l2|cosine|dot"
                )));
            }
        };
        self.db
            .create_collection(name, CollectionConfig::new(dim, m)?)
            .map(|_| ())
            .map_err(js)
    }

    /// 集合名列表。
    #[wasm_bindgen(js_name = "collections")]
    pub fn collections(&self) -> Result<String, JsError> {
        serde_json::to_string(&self.db.collections())
            .map_err(|e| js(vdb::Error::Serde(e.to_string())))
    }

    /// 批量写入。points_json: Point JSON 数组。返回写入数。
    pub fn upsert(&self, coll: &str, points_json: &str) -> Result<usize, JsError> {
        let c = self
            .db
            .collection(coll)
            .ok_or_else(|| js(vdb::Error::CollectionNotFound(coll.into())))?;
        let pts: Vec<Point> =
            serde_json::from_str(points_json).map_err(|e| js(vdb::Error::Serde(e.to_string())))?;
        c.upsert(&pts)?;
        Ok(pts.len())
    }

    /// 按 ID 取点，返回 `{"points": [Point|null, ...]}`。
    pub fn get(&self, coll: &str, ids_json: &str) -> Result<String, JsError> {
        let c = self
            .db
            .collection(coll)
            .ok_or_else(|| js(vdb::Error::CollectionNotFound(coll.into())))?;
        let pts = c.get(&parse_ids(ids_json)?)?;
        serde_json::to_string(&serde_json::json!({ "points": pts }))
            .map_err(|e| js(vdb::Error::Serde(e.to_string())))
    }

    /// 按 ID 删除，返回删除数。
    pub fn delete(&self, coll: &str, ids_json: &str) -> Result<usize, JsError> {
        let c = self
            .db
            .collection(coll)
            .ok_or_else(|| js(vdb::Error::CollectionNotFound(coll.into())))?;
        c.delete(&parse_ids(ids_json)?).map_err(js)
    }

    /// 分页遍历，返回 `{"points": [...], "next_offset": n|null}`。
    pub fn scroll(&self, coll: &str, offset: u64, limit: usize) -> Result<String, JsError> {
        let c = self
            .db
            .collection(coll)
            .ok_or_else(|| js(vdb::Error::CollectionNotFound(coll.into())))?;
        let page = c.scroll(offset, limit, None)?;
        serde_json::to_string(&serde_json::json!({
            "points": page.points,
            "next_offset": page.next_offset,
        }))
        .map_err(|e| js(vdb::Error::Serde(e.to_string())))
    }

    /// 邻近查询。query_json: `{"vector": [...], "top_k": 10, "filter": {...}}`
    /// （vector 也可换为 `"id": ...`）。返回 `{"points": [ScoredPoint...]}`。
    pub fn query(&self, coll: &str, query_json: &str) -> Result<String, JsError> {
        let c = self
            .db
            .collection(coll)
            .ok_or_else(|| js(vdb::Error::CollectionNotFound(coll.into())))?;
        let q: Query =
            serde_json::from_str(query_json).map_err(|e| js(vdb::Error::Serde(e.to_string())))?;
        let hits = c.query(&q)?;
        serde_json::to_string(&serde_json::json!({ "points": hits }))
            .map_err(|e| js(vdb::Error::Serde(e.to_string())))
    }

    /// 过滤遍历。filter_json: Condition JSON（见 core::filter 文档）。
    #[wasm_bindgen(js_name = "scrollFiltered")]
    pub fn scroll_filtered(
        &self,
        coll: &str,
        offset: u64,
        limit: usize,
        filter_json: &str,
    ) -> Result<String, JsError> {
        let c = self
            .db
            .collection(coll)
            .ok_or_else(|| js(vdb::Error::CollectionNotFound(coll.into())))?;
        let f: Condition =
            serde_json::from_str(filter_json).map_err(|e| js(vdb::Error::Serde(e.to_string())))?;
        let page = c.scroll(offset, limit, Some(&f))?;
        serde_json::to_string(&serde_json::json!({
            "points": page.points,
            "next_offset": page.next_offset,
        }))
        .map_err(|e| js(vdb::Error::Serde(e.to_string())))
    }

    /// 覆盖 payload。
    pub fn set_payload(
        &self,
        coll: &str,
        ids_json: &str,
        payload_json: &str,
    ) -> Result<usize, JsError> {
        let c = self
            .db
            .collection(coll)
            .ok_or_else(|| js(vdb::Error::CollectionNotFound(coll.into())))?;
        let payload: serde_json::Value =
            serde_json::from_str(payload_json).map_err(|e| js(vdb::Error::Serde(e.to_string())))?;
        c.set_payload(&parse_ids(ids_json)?, &payload).map_err(js)
    }

    /// 集合存活点数。
    pub fn count(&self, coll: &str) -> Result<u64, JsError> {
        let c = self
            .db
            .collection(coll)
            .ok_or_else(|| js(vdb::Error::CollectionNotFound(coll.into())))?;
        Ok(c.count())
    }
}

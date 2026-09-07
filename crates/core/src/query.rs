//! 查询请求与结果结构。

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::filter::Condition;
use crate::id::ExternalId;

/// 邻近查询。用 builder 方法链构造。
///
/// ```
/// use vectordb_core::{Metric, Query};
///
/// let q = Query::vector(vec![0.1, 0.2])
///     .top_k(5)
///     .filter(vectordb_core::Condition::matches("lang", "en"))
///     .with_vector(true);
/// assert_eq!(q.top_k, 5);
/// # _ = Metric::L2;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    /// 查询向量; 与 `id` 二选一。
    pub vector: Option<Vec<f32>>,
    /// 用已有点的向量作查询（自身会被排除）。
    pub id: Option<ExternalId>,
    /// 返回条数，默认 10。
    pub top_k: usize,
    /// 过滤条件。
    pub filter: Option<Condition>,
    /// 结果带 payload，默认 true。
    pub with_payload: bool,
    /// 结果带原始向量，默认 false。
    pub with_vector: bool,
    /// 分数阈值: L2 为上界（距离 <= t），Cosine/Dot 为下界（分值 >= t）。
    pub score_threshold: Option<f32>,
}

impl Query {
    /// 以向量发起查询。
    pub fn vector(v: Vec<f32>) -> Self {
        Query {
            vector: Some(v),
            id: None,
            top_k: 10,
            filter: None,
            with_payload: true,
            with_vector: false,
            score_threshold: None,
        }
    }

    /// 以已有点的向量发起查询（该点自身被排除）。
    pub fn nearest_to(id: impl Into<ExternalId>) -> Self {
        Query {
            vector: None,
            id: Some(id.into()),
            ..Query::vector(Vec::new())
        }
    }

    /// 设置返回条数。
    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// 设置过滤条件。
    pub fn filter(mut self, c: Condition) -> Self {
        self.filter = Some(c);
        self
    }

    /// 是否带 payload。
    pub fn with_payload(mut self, yes: bool) -> Self {
        self.with_payload = yes;
        self
    }

    /// 是否带原始向量。
    pub fn with_vector(mut self, yes: bool) -> Self {
        self.with_vector = yes;
        self
    }

    /// 设置分数阈值。
    pub fn threshold(mut self, t: f32) -> Self {
        self.score_threshold = Some(t);
        self
    }
}

/// 命中结果。分值语义见 [`crate::kernel`]（L2 越小越好，其余越大越好）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredPoint {
    /// 点 ID。
    pub id: ExternalId,
    /// 分数。
    pub score: f32,
    /// payload（查询未要求时为 None）。
    pub payload: Option<Value>,
    /// 原始向量（查询未要求时为 None）。
    pub vector: Option<Vec<f32>>,
}

/// [`crate::engine::Collection::scroll`] 的一页。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrollPage {
    /// 本页数据点。
    pub points: Vec<crate::segment::Point>,
    /// 下一页偏移; None 表示遍历完毕。
    pub next_offset: Option<u64>,
}

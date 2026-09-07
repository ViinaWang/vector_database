//! 段模型（ADR 0002）: 内存可变段（[`mutable`]）与磁盘不可变段（[`immutable`]）。
//!
//! 一个 collection 的数据 = 一个可变段 + 任意个不可变段。
//! 不可变段上的删除/payload 更新走小的可重写边车（dels.bin / payloads.overlay.jsonl），
//! 大文件永不重写; [`crate::engine`] 的 compact 负责最终合并。

pub mod immutable;
pub mod mutable;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::id::ExternalId;

/// 一个数据点: 外部 ID + 向量 + 可选 payload。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point {
    /// 用户可见 ID。
    pub id: ExternalId,
    /// 向量，维度须与集合配置一致。
    pub vector: Vec<f32>,
    /// JSON payload，可为空。
    pub payload: Option<Value>,
}

impl Point {
    /// 构造一个点。
    pub fn new(id: impl Into<ExternalId>, vector: Vec<f32>, payload: Option<Value>) -> Self {
        Point {
            id: id.into(),
            vector,
            payload,
        }
    }
}

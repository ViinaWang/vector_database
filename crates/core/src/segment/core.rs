//! 段的不可变核心数据。段本体与 ANN 索引共享同一份 `Arc<SegmentCore>`，
//! 索引内部对具体类型单态访问，避免热路径的动态分发。

use serde_json::Value;

use crate::id::ExternalId;

/// 一个段的数据主体（不含墓碑/覆盖层等可变状态）。
pub struct SegmentCore {
    /// 向量维度。
    pub dim: usize,
    /// 行主序向量，n*dim。
    pub vectors: Vec<f32>,
    /// 每行 L2 范数，与向量行对齐。
    pub norms: Vec<f32>,
    /// 外部 ID。
    pub ids: Vec<ExternalId>,
    /// payload。
    pub payloads: Vec<Option<Value>>,
}

impl SegmentCore {
    /// 总行数（含墓碑行）。
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// 是否为空段。
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// 第 idx 行的向量切片。
    pub fn vector(&self, idx: u32) -> &[f32] {
        let start = idx as usize * self.dim;
        &self.vectors[start..start + self.dim]
    }

    /// 第 idx 行的预计算范数。
    pub fn norm(&self, idx: u32) -> f32 {
        self.norms[idx as usize]
    }
}

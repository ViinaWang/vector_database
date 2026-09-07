//! 索引模块。
//!
//! [`AnnIndex`] 是不可变段的 ANN 检索接口: flat 为精确扫描（默认/对照），
//! hnsw 为近似图索引。索引持有段的 `Arc<SegmentCore>`，内部对具体类型
//! 单态访问数据，热路径无动态分发。

pub mod flat;
pub mod hnsw;

use std::sync::Arc;

use crate::kernel::Metric;
use crate::segment::core::SegmentCore;

/// 不可变段的近邻检索接口。
///
/// `pass` 返回 false 的行在检索中跳过（墓碑 + 过滤条件由此进入遍历）。
/// 返回至多 k 个 (内部序号, 分值)，最优在前; 分值语义见 [`crate::kernel`]。
pub trait AnnIndex: Send + Sync {
    /// 检索。
    fn search(
        &self,
        query: &[f32],
        k: usize,
        pass: Option<&dyn Fn(u32) -> bool>,
    ) -> Vec<(u32, f32)>;
    /// 索引自身结构的近似内存占用（不含向量数据）。
    fn est_bytes(&self) -> u64;
    /// 索引种类名（flat / hnsw）。
    fn kind(&self) -> &'static str;
}

/// 精确扫描索引。正确性基准与小区段路径。
pub struct FlatIndex {
    core: Arc<SegmentCore>,
    metric: Metric,
}

impl FlatIndex {
    /// 构造。
    pub fn new(core: Arc<SegmentCore>, metric: Metric) -> Self {
        FlatIndex { core, metric }
    }
}

impl AnnIndex for FlatIndex {
    fn search(
        &self,
        query: &[f32],
        k: usize,
        pass: Option<&dyn Fn(u32) -> bool>,
    ) -> Vec<(u32, f32)> {
        flat::search(
            self.metric,
            &self.core.vectors,
            &self.core.norms,
            self.core.dim,
            query,
            pass,
            k,
        )
    }

    fn est_bytes(&self) -> u64 {
        0
    }

    fn kind(&self) -> &'static str {
        "flat"
    }
}

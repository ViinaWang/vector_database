//! HNSW 参数。默认值 m=16 / m0=32 / ef_construct=128 / ef_search=512。
//!
//! 实测 recall@10（10k 均匀随机数据，最难场景; 真实嵌入数据因有簇结构会更好）:
//!
//! | dim | ef=128 | ef=256 | ef=512 | ef=1024 |
//! |-----|--------|--------|--------|---------|
//! | 128 (L2)     | 0.90 | 0.96 | 0.99 | 0.99 |
//! | 768 (L2)     | 0.74 | 0.92 | 0.96 | 0.99 |
//! | 768 (Cosine) | 0.64 | 0.81 | 0.93 | 0.99 |
//!
//! ef=1024 时全部 ≥0.99，说明近邻在图上可达，差距只在导航效率——
//! 高维随机数据距离集中，贪心路径需要更宽候选。低延迟场景可调低
//! ef_search; 构建质量优化（更好的邻居选择/增量策略）是 perf/ 分支议题。

use serde::{Deserialize, Serialize};

/// HNSW 构建与检索参数。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HnswParams {
    /// 上层每节点最大出边数。
    pub m: usize,
    /// 第 0 层每节点最大出边数（惯例为 2m）。
    pub m0: usize,
    /// 构建期候选队列宽度。
    pub ef_construct: usize,
    /// 查询期候选队列宽度（结果不足 k 时自动倍增重试）。
    pub ef_search: usize,
}

impl Default for HnswParams {
    fn default() -> Self {
        HnswParams {
            m: 16,
            m0: 32,
            ef_construct: 128,
            ef_search: 512,
        }
    }
}

impl HnswParams {
    /// 指定层级的最大出边数。
    pub fn max_links(&self, level: i32) -> usize {
        if level == 0 { self.m0 } else { self.m }
    }
}

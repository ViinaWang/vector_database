//! 集合配置。集合句柄与生命周期见 [`crate::engine::Collection`]。

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::index::hnsw::params::HnswParams;
use crate::kernel::Metric;

/// 不可变段使用的索引种类。
///
/// JSON: `{"kind":"hnsw", ...参数}` 或 `{"kind":"flat"}`。
/// 默认 hnsw。已写入的旧格式 manifest/WAL 缺该字段时按默认补齐。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum IndexKind {
    /// 精确扫描（对照/测试用，或极小数据集）。
    Flat,
    /// HNSW 近似图索引。
    Hnsw {
        /// 索引参数。
        #[serde(flatten)]
        params: HnswParams,
    },
}

impl Default for IndexKind {
    fn default() -> Self {
        IndexKind::Hnsw {
            params: HnswParams::default(),
        }
    }
}

/// 集合配置。建库后不可变（v0.1）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollectionConfig {
    /// 向量维度，> 0。
    pub dim: usize,
    /// 距离度量。
    pub metric: Metric,
    /// 不可变段索引种类。
    #[serde(default)]
    pub index: IndexKind,
}

impl CollectionConfig {
    /// 构造并校验，默认 HNSW 索引。
    pub fn new(dim: usize, metric: Metric) -> Result<Self> {
        if dim == 0 {
            return Err(Error::Invalid("dimension must be > 0".into()));
        }
        Ok(CollectionConfig {
            dim,
            metric,
            index: IndexKind::default(),
        })
    }

    /// 指定索引种类。
    pub fn with_index(mut self, index: IndexKind) -> Self {
        self.index = index;
        self
    }
}

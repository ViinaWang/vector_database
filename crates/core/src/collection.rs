//! 集合配置。集合句柄与生命周期见 [`crate::engine::Collection`]。

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::kernel::Metric;

/// 集合配置。建库后不可变（v0.1）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollectionConfig {
    /// 向量维度，> 0。
    pub dim: usize,
    /// 距离度量。
    pub metric: Metric,
}

impl CollectionConfig {
    /// 构造并校验。
    pub fn new(dim: usize, metric: Metric) -> Result<Self> {
        if dim == 0 {
            return Err(Error::Invalid("dimension must be > 0".into()));
        }
        Ok(CollectionConfig { dim, metric })
    }
}

//! manifest: 数据库当前状态的权威描述（版本、last_flushed_lsn、各集合的段列表）。
//! 任何落盘切换都通过"写 tmp + rename"原子替换整个文件（ADR 0002）。

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::collection::CollectionConfig;
use crate::error::{Error, Result};
use crate::storage::{Fs, atomic_write};

const FILE: &str = "manifest.json";

/// 数据库 manifest。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// 格式版本，当前恒为 [`crate::VERSION`]。
    pub version: u32,
    /// 已固化进段的最高 lsn; 重放时跳过 <= 该值的 WAL 记录。
    pub last_flushed_lsn: u64,
    /// 集合列表。
    pub collections: Vec<CollectionEntry>,
}

/// 单个集合在 manifest 中的条目。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionEntry {
    /// 集合名。
    pub name: String,
    /// 集合配置。
    pub config: CollectionConfig,
    /// 该集合的不可变段 ID 列表（时间顺序）。
    pub segments: Vec<u64>,
}

impl Default for Manifest {
    fn default() -> Self {
        Manifest {
            version: crate::VERSION,
            last_flushed_lsn: 0,
            collections: Vec::new(),
        }
    }
}

impl Manifest {
    /// 读取; 文件不存在返回 None。
    pub fn load(fs: &dyn Fs, root: &Path) -> Result<Option<Self>> {
        let path = root.join(FILE);
        if !fs.exists(&path) {
            return Ok(None);
        }
        let f = fs.open_rw(&path)?;
        let len = f.len()?;
        let mut buf = vec![0u8; len as usize];
        f.read_at(&mut buf, 0)?;
        let m: Manifest = serde_json::from_slice(&buf)?;
        if m.version > crate::VERSION {
            return Err(Error::Invalid(format!(
                "manifest version {} is newer than supported {}",
                m.version,
                crate::VERSION
            )));
        }
        Ok(Some(m))
    }

    /// 原子保存。
    pub fn save(&self, fs: &dyn Fs, root: &Path) -> Result<()> {
        atomic_write(fs, &root.join(FILE), &serde_json::to_vec_pretty(self)?)
    }

    /// 集合条目可变引用。
    pub fn entry_mut(&mut self, name: &str) -> Option<&mut CollectionEntry> {
        self.collections.iter_mut().find(|c| c.name == name)
    }

    /// 全部段 ID 的最大值（分配新段 ID 用）。
    pub fn max_segment_id(&self) -> u64 {
        self.collections
            .iter()
            .flat_map(|c| c.segments.iter().copied())
            .max()
            .unwrap_or(0)
    }
}

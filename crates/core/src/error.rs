//! 错误类型。库边界上只有这一个枚举，避免调用方匹配内部细节。

use std::io;

use thiserror::Error;

/// 内核统一错误。
#[derive(Debug, Error)]
pub enum Error {
    /// 底层 IO 失败（文件系统或内存后端）。
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    /// 序列化/反序列化失败（manifest、段文件、WAL 记录体）。
    #[error("serialization error: {0}")]
    Serde(String),
    /// 集合已存在。
    #[error("collection {0:?} already exists")]
    CollectionExists(String),
    /// 集合不存在。
    #[error("collection {0:?} not found")]
    CollectionNotFound(String),
    /// 向量维度与集合配置不符。
    #[error("dimension mismatch: collection expects {expected}, got {got}")]
    DimensionMismatch {
        /// 集合配置的维度。
        expected: usize,
        /// 实际收到的维度。
        got: usize,
    },
    /// WAL 尾部损坏（断电/磁盘满导致的部分写入）。截断后数据一致，但应引起注意。
    #[error("wal corrupt tail at offset {offset}: {reason}")]
    WalCorrupt {
        /// 损坏起始偏移。
        offset: u64,
        /// 具体原因。
        reason: String,
    },
    /// 数据库目录已被另一个进程打开（文件锁）。
    #[error("database directory is locked by another process: {0}")]
    AlreadyOpen(String),
    /// 参数不合法（空向量、top_k 为 0、同时给 vector 和 id 等）。
    #[error("invalid argument: {0}")]
    Invalid(String),
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Serde(e.to_string())
    }
}

/// 内核统一 Result。
pub type Result<T> = std::result::Result<T, Error>;

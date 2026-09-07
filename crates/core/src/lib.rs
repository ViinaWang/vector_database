//! vectordb-core: 嵌入式向量数据库内核。
//!
//! 结构（详见各模块文档与 docs/adr/）:
//! - [`storage`] — 文件与目录的 IO 抽象（native / memory 两套实现）
//! - [`wal`] — 先行日志，崩溃恢复的数据来源
//! - [`segment`] — 内存可变段与磁盘不可变段
//! - [`kernel`] — 距离度量（热路径，唯一入口）
//! - [`filter`] — 过滤条件 AST 与求值
//! - [`engine`] — 对外 API（Database / Collection）
//!
//! 并发语义: 一个 Database 可被多线程共享; 读操作并发，写操作按 collection 串行。
//! 所有写先落 WAL 再应用内存，[`engine::Database::flush`] 将内存段固化为不可变段。

pub mod collection;
pub mod engine;
pub mod error;
pub mod filter;
pub mod id;
pub mod index;
pub mod kernel;
pub mod manifest;
pub mod query;
pub mod segment;
pub mod storage;
pub mod wal;

pub use collection::{CollectionConfig, IndexKind};
pub use engine::{Collection, Database, DatabaseOptions};
pub use error::{Error, Result};
pub use filter::{Condition, FieldOp};
pub use id::ExternalId;
pub use index::hnsw::params::HnswParams;
pub use kernel::Metric;
pub use query::{Query, ScoredPoint, ScrollPage};
pub use segment::Point;

/// 库版本报告给日志与 manifest 兼容性检查用。
pub const VERSION: u32 = 1;

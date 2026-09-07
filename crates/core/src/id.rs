//! 点的外部标识。用户可见、可持久化; 内部存储另用每段的 u32 序号。

use serde::{Deserialize, Serialize};
use std::fmt;

/// 用户提供的点 ID，数字或字符串。实现 Hash/Eq，可作 HashMap 键。
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ExternalId {
    /// 数字 ID（JSON number，i64 范围）。
    Num(i64),
    /// 字符串 ID。
    Str(String),
}

impl ExternalId {
    /// 是数字 ID 时返回其值。
    pub fn as_num(&self) -> Option<i64> {
        match self {
            ExternalId::Num(n) => Some(*n),
            ExternalId::Str(_) => None,
        }
    }
    /// 是字符串 ID 时返回其引用。
    pub fn as_str(&self) -> Option<&str> {
        match self {
            ExternalId::Num(_) => None,
            ExternalId::Str(s) => Some(s),
        }
    }
}

impl fmt::Display for ExternalId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExternalId::Num(n) => write!(f, "{n}"),
            ExternalId::Str(s) => write!(f, "{s}"),
        }
    }
}

impl From<i64> for ExternalId {
    fn from(v: i64) -> Self {
        ExternalId::Num(v)
    }
}
impl From<u64> for ExternalId {
    fn from(v: u64) -> Self {
        ExternalId::Num(v as i64)
    }
}
impl From<i32> for ExternalId {
    fn from(v: i32) -> Self {
        ExternalId::Num(i64::from(v))
    }
}
impl From<u32> for ExternalId {
    fn from(v: u32) -> Self {
        ExternalId::Num(i64::from(v))
    }
}
impl From<String> for ExternalId {
    fn from(v: String) -> Self {
        ExternalId::Str(v)
    }
}
impl From<&str> for ExternalId {
    fn from(v: &str) -> Self {
        ExternalId::Str(v.to_owned())
    }
}

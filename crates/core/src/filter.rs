//! 过滤条件 AST 与求值。当前为逐点求值（配合 flat 扫描在算距离前过滤）；
//! 字段索引（倒排/范围）在 v0.2 接入，AST 保持不变。
//!
//! JSON 形态（untagged，靠键区分）:
//! `{"field": "lang", "op": {"kind": "match", "value": "en"}}`
//! `{"field": "year", "op": {"kind": "range", "gte": 2020, "lt": 2025}}`
//! `{"all": {"of": [ ... ]}}`、`{"any": {"of": [ ... ]}}`、`{"not": {"of": { ... }}}`
//! `{"ids": [1, "a"]}`

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::id::ExternalId;

/// 过滤条件。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Condition {
    /// 字段条件。
    Field {
        /// payload 中的字段名（顶层键）。
        field: String,
        /// 字段上的操作。
        op: FieldOp,
    },
    /// ID 集合命中。
    HasId {
        /// 允许的 ID 列表。
        ids: Vec<ExternalId>,
    },
    /// 全部满足（空列表为 true）。
    All {
        /// 子条件。
        of: Vec<Condition>,
    },
    /// 任一满足（空列表为 false）。
    Any {
        /// 子条件。
        of: Vec<Condition>,
    },
    /// 取反。
    Not {
        /// 被取反的子条件。
        of: Box<Condition>,
    },
}

/// 字段上的操作。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FieldOp {
    /// 相等; payload 值为数组时表示包含。
    Match {
        /// 期望值。
        value: Value,
    },
    /// 数值范围，边界均可选，至少给一个。
    Range {
        /// 严格大于。
        gt: Option<f64>,
        /// 大于等于。
        gte: Option<f64>,
        /// 严格小于。
        lt: Option<f64>,
        /// 小于等于。
        lte: Option<f64>,
    },
    /// 字段存在（值为 null 视为不存在）。
    Exists,
}

impl Condition {
    /// 构造相等条件。
    pub fn matches<V: Into<Value>>(field: &str, value: V) -> Self {
        Condition::Field {
            field: field.to_owned(),
            op: FieldOp::Match {
                value: value.into(),
            },
        }
    }
    /// 构造数值范围条件。至少一个边界为 Some。
    pub fn range(
        field: &str,
        gt: Option<f64>,
        gte: Option<f64>,
        lt: Option<f64>,
        lte: Option<f64>,
    ) -> Self {
        Condition::Field {
            field: field.to_owned(),
            op: FieldOp::Range { gt, gte, lt, lte },
        }
    }
    /// 构造存在性条件。
    pub fn exists(field: &str) -> Self {
        Condition::Field {
            field: field.to_owned(),
            op: FieldOp::Exists,
        }
    }
    /// 全部满足。
    pub fn all(of: Vec<Condition>) -> Self {
        Condition::All { of }
    }
    /// 任一满足。
    pub fn any(of: Vec<Condition>) -> Self {
        Condition::Any { of }
    }
    /// 对一个点求值。payload 为 None 或字段缺失按"不匹配"处理。
    ///（取反直接用 `!cond`，见 [`std::ops::Not`] 实现。）
    pub fn eval(&self, id: &ExternalId, payload: Option<&Value>) -> bool {
        match self {
            Condition::Field { field, op } => {
                let v = payload.and_then(|p| p.get(field)).filter(|v| !v.is_null());
                match (op, v) {
                    (FieldOp::Exists, _) => v.is_some(),
                    (_, None) => false,
                    (FieldOp::Match { value }, Some(v)) => {
                        v == value || v.as_array().is_some_and(|arr| arr.contains(value))
                    }
                    (FieldOp::Range { gt, gte, lt, lte }, Some(v)) => {
                        let n = v.as_f64();
                        match n {
                            None => false,
                            Some(n) => {
                                gt.is_none_or(|b| n > b)
                                    && gte.is_none_or(|b| n >= b)
                                    && lt.is_none_or(|b| n < b)
                                    && lte.is_none_or(|b| n <= b)
                            }
                        }
                    }
                }
            }
            Condition::HasId { ids } => ids.contains(id),
            Condition::All { of } => of.iter().all(|c| c.eval(id, payload)),
            Condition::Any { of } => of.iter().any(|c| c.eval(id, payload)),
            Condition::Not { of } => !of.eval(id, payload),
        }
    }
}

impl std::ops::Not for Condition {
    type Output = Condition;
    fn not(self) -> Condition {
        Condition::Not { of: Box::new(self) }
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn eval(c: &Condition, payload: Option<Value>) -> bool {
        c.eval(&ExternalId::Num(1), payload.as_ref())
    }

    #[test]
    fn field_ops() {
        let p = Some(json!({"lang": "en", "year": 2023, "tags": ["a", "b"], "nul": null}));
        assert!(eval(&Condition::matches("lang", "en"), p.clone()));
        assert!(!eval(&Condition::matches("lang", "zh"), p.clone()));
        assert!(eval(&Condition::matches("tags", "a"), p.clone()));
        assert!(eval(
            &Condition::range("year", None, Some(2020.0), Some(2025.0), None),
            p.clone()
        ));
        assert!(!eval(
            &Condition::range("year", None, None, None, Some(2020.0)),
            p.clone()
        ));
        assert!(eval(&Condition::exists("lang"), p.clone()));
        assert!(!eval(&Condition::exists("nul"), p.clone()));
        assert!(!eval(&Condition::exists("missing"), p.clone()));
        assert!(!eval(&Condition::matches("lang", "en"), None));
    }

    #[test]
    fn boolean_ops() {
        let p = Some(json!({"lang": "en"}));
        let en = Condition::matches("lang", "en");
        let zh = Condition::matches("lang", "zh");
        assert!(eval(&Condition::all(vec![en.clone()]), p.clone()));
        assert!(!eval(
            &Condition::all(vec![en.clone(), zh.clone()]),
            p.clone()
        ));
        assert!(eval(
            &Condition::any(vec![zh.clone(), en.clone()]),
            p.clone()
        ));
        assert!(eval(&!zh, p.clone()));
        assert!(eval(&Condition::All { of: vec![] }, p.clone()));
        assert!(!eval(&Condition::Any { of: vec![] }, p.clone()));
    }

    #[test]
    fn has_id() {
        let c = Condition::HasId {
            ids: vec![ExternalId::Num(1), ExternalId::Str("x".into())],
        };
        assert!(c.eval(&ExternalId::Num(1), None));
        assert!(!c.eval(&ExternalId::Num(2), None));
    }

    #[test]
    fn json_roundtrip() {
        let c = Condition::all(vec![
            Condition::matches("lang", "en"),
            Condition::range("year", None, Some(2020.0), None, None),
            !Condition::exists("deleted"),
        ]);
        let s = serde_json::to_string(&c).unwrap();
        let back: Condition = serde_json::from_str(&s).unwrap();
        assert_eq!(c, back);
    }
}

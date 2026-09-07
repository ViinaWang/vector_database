//! 距离度量内核。全库唯一的距离计算入口（ADR 0003）。
//!
//! 分值语义（调用方依赖此约定排序）:
//! - [`Metric::L2`] — 返回**平方**欧氏距离，越小越好（省一次 sqrt，排序不变）
//! - [`Metric::Dot`] — 返回内积，越大越好
//! - [`Metric::Cosine`] — 返回归一化相似度，越大越好，范围 [-1, 1]
//!
//! 当前为标量 + 手工分块实现，依赖自动向量化; SIMD 专化在 perf/ 分支迭代，
//! 以 benches/kernel.rs 与本文件的朴素参照实现守护正确性。

use serde::{Deserialize, Serialize};

/// 距离度量种类。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Metric {
    /// 平方欧氏距离，越小越好。
    L2,
    /// 余弦相似度，越大越好。
    Cosine,
    /// 内积，越大越好（向量已归一化时等价于余弦）。
    Dot,
}

impl Metric {
    /// a 的分值是否严格优于 b。
    pub fn is_better(&self, a: f32, b: f32) -> bool {
        match self {
            Metric::L2 => a < b,
            Metric::Cosine | Metric::Dot => a > b,
        }
    }

    /// 分值是否通过阈值（L2: score <= threshold; 其余: score >= threshold）。
    pub fn passes(&self, score: f32, threshold: f32) -> bool {
        match self {
            Metric::L2 => score <= threshold,
            Metric::Cosine | Metric::Dot => score >= threshold,
        }
    }
}

/// 内积。向量长度不等时取较短者（调用方保证相等，debug 断言兜底）。
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    // 8 组独立累加器，固定 4 元素内层循环，给 LLVM 足够的向量化空间。
    const UNROLL: usize = 8;
    const LANE: usize = 4;
    let step = UNROLL * LANE;
    let n = a.len() / step * step;
    let mut acc = [0.0f32; UNROLL];
    let mut i = 0;
    while i < n {
        for (j, acc_j) in acc.iter_mut().enumerate() {
            let k = i + j * LANE;
            for l in 0..LANE {
                *acc_j += a[k + l] * b[k + l];
            }
        }
        i += step;
    }
    let mut sum: f32 = acc.iter().sum();
    while i < a.len() {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// 平方欧氏距离。
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    const UNROLL: usize = 8;
    const LANE: usize = 4;
    let step = UNROLL * LANE;
    let n = a.len() / step * step;
    let mut acc = [0.0f32; UNROLL];
    let mut i = 0;
    while i < n {
        for (j, acc_j) in acc.iter_mut().enumerate() {
            let k = i + j * LANE;
            for l in 0..LANE {
                let d = a[k + l] - b[k + l];
                *acc_j += d * d;
            }
        }
        i += step;
    }
    let mut sum: f32 = acc.iter().sum();
    while i < a.len() {
        let d = a[i] - b[i];
        sum += d * d;
        i += 1;
    }
    sum
}

/// L2 范数。
pub fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

/// 按度量计算分值。cosine 需要 b 的范数（段内预计算）; 任一范数为 0 时返回 0。
/// 热路径请用 [`score_with`]（预计算查询范数），本函数每行都会重算 a 的范数。
pub fn score(metric: Metric, a: &[f32], b: &[f32], b_norm: f32) -> f32 {
    score_with(metric, a, norm(a), b, b_norm)
}

/// 同 [`score`]，但调用方预计算 a 的范数（扫描/图遍历每查询只算一次）。
pub fn score_with(metric: Metric, a: &[f32], a_norm: f32, b: &[f32], b_norm: f32) -> f32 {
    match metric {
        Metric::L2 => l2_sq(a, b),
        Metric::Dot => dot(a, b),
        Metric::Cosine => {
            if a_norm == 0.0 || b_norm == 0.0 {
                0.0
            } else {
                dot(a, b) / (a_norm * b_norm)
            }
        }
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    fn naive_dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }
    fn naive_l2(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-3 * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn matches_naive_reference() {
        let dims = [1usize, 3, 7, 16, 64, 100, 128, 1536];
        for &dim in &dims {
            let a: Vec<f32> = (0..dim).map(|i| (i % 17) as f32 * 0.5 - 2.0).collect();
            let b: Vec<f32> = (0..dim)
                .map(|i| ((i * 31) % 13) as f32 * 0.25 - 1.0)
                .collect();
            assert!(close(dot(&a, &b), naive_dot(&a, &b)), "dot dim {dim}");
            assert!(close(l2_sq(&a, &b), naive_l2(&a, &b)), "l2 dim {dim}");
            let bn = norm(&b);
            let cos = dot(&a, &b) / (norm(&a) * bn);
            assert!(
                close(score(Metric::Cosine, &a, &b, bn), cos),
                "cos dim {dim}"
            );
        }
    }

    #[test]
    fn zero_vector_cosine_is_zero() {
        let a = vec![1.0f32, 2.0];
        let z = vec![0.0f32, 0.0];
        assert_eq!(score(Metric::Cosine, &a, &z, norm(&z)), 0.0);
    }
}

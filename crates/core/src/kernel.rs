//! 距离度量内核。全库唯一的距离计算入口（ADR 0003）。
//!
//! 分值语义（调用方依赖此约定排序）:
//! - [`Metric::L2`] — 返回**平方**欧氏距离，越小越好（省一次 sqrt，排序不变）
//! - [`Metric::Dot`] — 返回内积，越大越好
//! - [`Metric::Cosine`] — 返回归一化相似度，越大越好，范围 [-1, 1]
//!
//! x86_64 上运行时检测 AVX2+FMA 并走特化实现，其余平台用标量分块
//! （依赖自动向量化）。两种实现的求和顺序不同，结果允许 ulp 级差异;
//! 正确性由本文件的朴素参照测试与 benches/kernel.rs 共同守护。

// ADR 0003: 距离内核是允许 unsafe 的热路径模块，每处 unsafe 附 SAFETY 注释。
#![allow(unsafe_code)]

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

/// 内积。
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if avx2_fma() {
            // SAFETY: avx2_fma() 已运行时确认 CPU 支持 avx2+fma; loadu 不要求对齐。
            return unsafe { dot_avx2(a, b) };
        }
    }
    dot_scalar(a, b)
}

/// 平方欧氏距离。
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if avx2_fma() {
            // SAFETY: 同 dot_avx2。
            return unsafe { l2_sq_avx2(a, b) };
        }
    }
    l2_sq_scalar(a, b)
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

// ---------------------------------------------------------------- 标量实现

fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
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

fn l2_sq_scalar(a: &[f32], b: &[f32]) -> f32 {
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

// ---------------------------------------------------------------- AVX2 实现

#[cfg(target_arch = "x86_64")]
fn avx2_fma() -> bool {
    use std::sync::OnceLock;
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    // SAFETY: avx2_fma() 已运行时确认支持; loadu 不要求对齐，
    // 所有访问的下标由 n = len/8*8 前置约束。
    unsafe {
        let n = a.len() / 8 * 8;
        let mut acc = [_mm256_setzero_ps(); 8];
        let mut i = 0;
        // 8 组 ymm 累加器（64 元素/步），减少 FMA 端口依赖。
        while i + 64 <= n {
            for (j, acc_j) in acc.iter_mut().enumerate() {
                let k = i + j * 8;
                let va = _mm256_loadu_ps(a.as_ptr().add(k));
                let vb = _mm256_loadu_ps(b.as_ptr().add(k));
                *acc_j = _mm256_fmadd_ps(va, vb, *acc_j);
            }
            i += 64;
        }
        while i + 8 <= n {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            acc[0] = _mm256_fmadd_ps(va, vb, acc[0]);
            i += 8;
        }
        let mut s = acc[0];
        for &a in &acc[1..] {
            s = _mm256_add_ps(s, a);
        }
        let mut sum = horizontal_sum(s);
        while i < a.len() {
            sum += a[i] * b[i];
            i += 1;
        }
        sum
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2_sq_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    // SAFETY: 同 dot_avx2。
    unsafe {
        let n = a.len() / 8 * 8;
        let mut acc = [_mm256_setzero_ps(); 8];
        let mut i = 0;
        while i + 64 <= n {
            for (j, acc_j) in acc.iter_mut().enumerate() {
                let k = i + j * 8;
                let va = _mm256_loadu_ps(a.as_ptr().add(k));
                let vb = _mm256_loadu_ps(b.as_ptr().add(k));
                let d = _mm256_sub_ps(va, vb);
                *acc_j = _mm256_fmadd_ps(d, d, *acc_j);
            }
            i += 64;
        }
        while i + 8 <= n {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let d = _mm256_sub_ps(va, vb);
            acc[0] = _mm256_fmadd_ps(d, d, acc[0]);
            i += 8;
        }
        let mut s = acc[0];
        for &a in &acc[1..] {
            s = _mm256_add_ps(s, a);
        }
        let mut sum = horizontal_sum(s);
        while i < a.len() {
            let d = a[i] - b[i];
            sum += d * d;
            i += 1;
        }
        sum
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn horizontal_sum(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let mut x = _mm_add_ps(_mm256_castps256_ps128(v), hi);
    x = _mm_add_ps(x, _mm_movehdup_ps(x));
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    _mm_cvtss_f32(x)
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
        let dims = [1usize, 3, 7, 16, 63, 64, 65, 100, 128, 1536];
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

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn simd_and_scalar_agree_within_tolerance() {
        if !avx2_fma() {
            return;
        }
        let mut seed = 0x1234_5678u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / 2147483648.0) - 1.0
        };
        for dim in [8usize, 128, 768, 1537] {
            let a: Vec<f32> = (0..dim).map(|_| next()).collect();
            let b: Vec<f32> = (0..dim).map(|_| next()).collect();
            let s_dot = dot_scalar(&a, &b);
            let v_dot = unsafe { dot_avx2(&a, &b) };
            assert!(
                (s_dot - v_dot).abs() <= 1e-4 * (1.0 + s_dot.abs()),
                "dot {dim}: {s_dot} vs {v_dot}"
            );
            let s_l2 = l2_sq_scalar(&a, &b);
            let v_l2 = unsafe { l2_sq_avx2(&a, &b) };
            assert!(
                (s_l2 - v_l2).abs() <= 1e-4 * (1.0 + s_l2.abs()),
                "l2 {dim}: {s_l2} vs {v_l2}"
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

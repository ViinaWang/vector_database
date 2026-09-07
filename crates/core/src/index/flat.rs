//! 精确（暴力）最近邻扫描。当前唯一的检索路径，也是后续 ANN 索引的
//! ground truth 与小区段回退路径。
//!
//! 过滤在算距离**之前**应用（`pass` 返回 false 直接跳过），
//! 因此过滤不会损失召回——这是精确扫描相对 ANN 过滤的语义优势。

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use crate::kernel::{self, Metric};

/// f32 不是 Ord，堆里用 total_cmp 语义包装（全序）。
struct HeapKey(f32, u32);

impl PartialEq for HeapKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0) == Ordering::Equal && self.1 == other.1
    }
}
impl Eq for HeapKey {}

impl Ord for HeapKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0).then(self.1.cmp(&other.1))
    }
}

impl PartialOrd for HeapKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// 扫描一段扁平存储，返回分数最好的至多 k 个 (内部序号, 分值)，最优在前。
///
/// - `vectors`: n*dim 行主序; `norms`: n 个预计算范数
/// - `pass`: 存活 + 过滤条件的合成判断（None 视为全部通过）
/// - NaN 分值的行跳过（输入含 NaN 属调用方数据问题）
pub fn search(
    metric: Metric,
    vectors: &[f32],
    norms: &[f32],
    dim: usize,
    query: &[f32],
    pass: Option<&dyn Fn(u32) -> bool>,
    k: usize,
) -> Vec<(u32, f32)> {
    if dim == 0 || k == 0 || vectors.is_empty() {
        return Vec::new();
    }
    debug_assert_eq!(vectors.len() % dim, 0);
    let n = vectors.len() / dim;
    // 归一化分值: 统一为"越大越好"，堆顶是最差的留存者。
    let normed = |raw: f32| match metric {
        Metric::L2 => -raw,
        Metric::Cosine | Metric::Dot => raw,
    };

    let mut heap: BinaryHeap<Reverse<HeapKey>> = BinaryHeap::new();
    for idx in 0..n as u32 {
        if pass.map(|p| !p(idx)).unwrap_or(false) {
            continue;
        }
        let v = &vectors[idx as usize * dim..(idx as usize + 1) * dim];
        let raw = kernel::score(metric, query, v, norms[idx as usize]);
        if raw.is_nan() {
            continue;
        }
        let s = normed(raw);
        if heap.len() < k {
            heap.push(Reverse(HeapKey(s, idx)));
        } else if let Some(&Reverse(HeapKey { 0: worst, .. })) = heap.peek() {
            if s > worst {
                heap.pop();
                heap.push(Reverse(HeapKey(s, idx)));
            }
        }
    }

    let mut out: Vec<(u32, f32)> = heap
        .into_iter()
        .map(|Reverse(HeapKey(s, idx))| (idx, denorm(metric, s)))
        .collect();
    out.sort_by(|a, b| {
        let better_first = match metric {
            Metric::L2 => a.1.total_cmp(&b.1),
            Metric::Cosine | Metric::Dot => b.1.total_cmp(&a.1),
        };
        better_first.then(a.0.cmp(&b.0))
    });
    out
}

fn denorm(metric: Metric, s: f32) -> f32 {
    match metric {
        Metric::L2 => -s,
        Metric::Cosine | Metric::Dot => s,
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_ordering_per_metric() {
        // 三个二维点; query = (1, 0)
        let vectors: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0];
        let norms: Vec<f32> = vectors.chunks(2).map(kernel::norm).collect();
        let q = vec![1.0, 0.0];

        let l2 = search(Metric::L2, &vectors, &norms, 2, &q, None, 2);
        assert_eq!(l2[0].0, 0);
        assert!(Metric::L2.is_better(l2[0].1, l2[1].1));

        let dot = search(Metric::Dot, &vectors, &norms, 2, &q, None, 2);
        assert_eq!(dot[0].0, 0);
        assert_eq!(dot[0].1, 1.0); // 恰好命中自身
    }

    #[test]
    fn filter_skips_before_scoring() {
        let vectors: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let norms: Vec<f32> = vec![1.0, 1.0];
        let hits = search(
            Metric::L2,
            &vectors,
            &norms,
            2,
            &[1.0, 0.0],
            Some(&|i| i != 0),
            5,
        );
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 1);
    }

    #[test]
    fn k_zero_returns_empty() {
        let out = search(Metric::Dot, &[1.0], &[1.0], 1, &[1.0], None, 0);
        assert!(out.is_empty());
    }
}

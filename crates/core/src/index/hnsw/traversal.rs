//! HNSW 遍历原语: 层内 best-first 搜索、贪心下钻、启发式邻居选择。
//! 构建与查询共用，是索引热路径（ADR 0003 的 unsafe 开洞候选区）。
//!
//! 距离统一为"越小越好"的 key: L2 用平方距离，Cosine/Dot 取负分值。

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use roaring::RoaringBitmap;

use crate::kernel::{self, Metric};
use crate::segment::core::SegmentCore;

use super::graph::HnswGraph;

/// 空/哨兵节点号。
pub const EMPTY: u32 = u32::MAX;

/// 遍历上下文: 图 + 数据 + 度量，构建与查询共用的只读三元组。
pub(crate) struct Ctx<'a> {
    /// 图结构。
    pub graph: &'a HnswGraph,
    /// 段数据。
    pub data: &'a SegmentCore,
    /// 距离度量。
    pub metric: Metric,
}

/// f32 不是 Ord，统一 total_cmp 语义（全序）。
pub(crate) struct Kf(pub f32, pub u32);

impl Kf {
    fn key(&self) -> f32 {
        self.0
    }
}
impl PartialEq for Kf {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0) == std::cmp::Ordering::Equal && self.1 == other.1
    }
}
impl Eq for Kf {}

impl Ord for Kf {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0).then(self.1.cmp(&other.1))
    }
}
impl PartialOrd for Kf {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// 分值 → 距离 key。
#[inline]
pub(crate) fn to_key(metric: Metric, score: f32) -> f32 {
    match metric {
        Metric::L2 => score,
        Metric::Cosine | Metric::Dot => -score,
    }
}

/// 距离 key → 分值。
#[inline]
pub(crate) fn to_score(metric: Metric, key: f32) -> f32 {
    match metric {
        Metric::L2 => key,
        Metric::Cosine | Metric::Dot => -key,
    }
}

/// 查询向量到节点 b 的距离 key。q_norm 为查询范数（每次搜索算一次）。
#[inline]
pub(crate) fn q_key(ctx: &Ctx<'_>, query: &[f32], q_norm: f32, b: u32) -> f32 {
    let d = ctx.data;
    to_key(
        ctx.metric,
        kernel::score_with(ctx.metric, query, q_norm, d.vector(b), d.norm(b)),
    )
}

/// 节点 a 到节点 b 的距离 key。
#[inline]
pub(crate) fn node_key(ctx: &Ctx<'_>, a: u32, b: u32) -> f32 {
    let d = ctx.data;
    to_key(
        ctx.metric,
        kernel::score(ctx.metric, d.vector(a), d.vector(b), d.norm(b)),
    )
}

/// 在 level 层从 ep 出发贪心走到局部最近（ef=1 语义，用于层间下钻）。
pub(crate) fn greedy_descend(ctx: &Ctx<'_>, query: &[f32], ep: &mut u32, level: i32) {
    let q_norm = kernel::norm(query);
    let mut best = q_key(ctx, query, q_norm, *ep);
    let mut improved = true;
    while improved {
        improved = false;
        for &nbr in ctx.graph.links(*ep, level) {
            if nbr == EMPTY {
                break;
            }
            let d = q_key(ctx, query, q_norm, nbr);
            if d < best {
                best = d;
                *ep = nbr;
                improved = true;
            }
        }
    }
}

/// 层内 best-first 搜索，返回至多 ef 个最近节点（key 升序）。
/// 过滤版: `pass` 为 false 的节点不进结果，但仍参与图遍历（连通性）。
pub(crate) fn search_layer(
    ctx: &Ctx<'_>,
    query: &[f32],
    eps: &[u32],
    ef: usize,
    level: i32,
    pass: Option<&dyn Fn(u32) -> bool>,
) -> Vec<(u32, f32)> {
    if ef == 0 || eps.is_empty() {
        return Vec::new();
    }
    let q_norm = kernel::norm(query);
    let mut visited = RoaringBitmap::new();
    // 候选小根堆（最近优先）。
    let mut candidates: BinaryHeap<Reverse<Kf>> = BinaryHeap::new();
    // 结果大根堆（堆顶是最差留存者），仅收 pass 通过的节点。
    let mut results: BinaryHeap<Kf> = BinaryHeap::new();

    for &ep in eps {
        if visited.insert(ep) {
            let d = q_key(ctx, query, q_norm, ep);
            if pass.is_none_or(|p| p(ep)) {
                push_result(&mut results, d, ep, ef);
            }
            candidates.push(Reverse(Kf(d, ep)));
        }
    }

    while let Some(Reverse(node)) = candidates.pop() {
        let d = node.key();
        if results.len() == ef && d > results.peek().map(|r| r.key()).unwrap_or(f32::MAX) {
            break;
        }
        for &nbr in ctx.graph.links(node.1, level) {
            if nbr == EMPTY || visited.contains(nbr) {
                continue;
            }
            visited.insert(nbr);
            let nd = q_key(ctx, query, q_norm, nbr);
            let full = results.len() == ef;
            let worst = results.peek().map(|r| r.key()).unwrap_or(f32::MAX);
            if !full || nd < worst {
                candidates.push(Reverse(Kf(nd, nbr)));
                if pass.is_none_or(|p| p(nbr)) {
                    push_result(&mut results, nd, nbr, ef);
                }
            }
        }
    }

    let mut out: Vec<(u32, f32)> = results.into_iter().map(|k| (k.1, k.key())).collect();
    out.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    out
}

fn push_result(heap: &mut BinaryHeap<Kf>, d: f32, node: u32, ef: usize) {
    if heap.len() < ef {
        heap.push(Kf(d, node));
    } else if let Some(worst) = heap.peek() {
        if d < worst.key() {
            heap.pop();
            heap.push(Kf(d, node));
        }
    }
}

/// 启发式邻居选择（论文 Algorithm 4，keepPruned=false，同 hnswlib）:
/// 只保留"到查询近且不与其他已选节点过近"的候选，允许不足 m。
/// 不回填是刻意的: 近重复邻居会稀释方向多样性，高维下尤其伤贪心质量。
pub(crate) fn select_heuristic(ctx: &Ctx<'_>, cands: &[(u32, f32)], m: usize) -> Vec<u32> {
    if cands.len() <= m {
        return cands.iter().map(|(id, _)| *id).collect();
    }
    let mut selected: Vec<u32> = Vec::with_capacity(m);
    let mut discarded: Vec<u32> = Vec::new();
    for &(id, d) in cands {
        if selected.len() == m {
            break;
        }
        let diverse = selected.iter().all(|&s| d < node_key(ctx, id, s));
        if diverse {
            selected.push(id);
        } else {
            discarded.push(id);
        }
    }
    for id in discarded {
        if selected.len() == m {
            break;
        }
        selected.push(id);
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_roundtrip() {
        for m in [Metric::L2, Metric::Cosine, Metric::Dot] {
            let s = 0.37;
            assert_eq!(to_score(m, to_key(m, s)), s);
        }
    }
}

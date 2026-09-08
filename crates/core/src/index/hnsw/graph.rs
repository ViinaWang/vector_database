//! HNSW 图结构: 第 0 层紧凑矩阵（count × m0），上层每节点 ragged。
//! 链接数组以 [`EMPTY`] 哨兵结尾，保持"非空前缀"不变量。

use crate::kernel::Metric;
use crate::segment::core::SegmentCore;

use super::params::HnswParams;
use super::rng::Rng;
use super::traversal::{self, EMPTY};

/// 上层（level ≥ 1）某节点的各层链接。
#[derive(Default)]
pub struct UpperLinks {
    /// layers[l-1] 为 level l 的出边。
    pub layers: Vec<Vec<u32>>,
}

/// HNSW 图。节点号即段内 u32 行号。
pub struct HnswGraph {
    /// 节点总数（含墓碑行）。
    pub count: u32,
    /// 上层最大出边。
    pub m: usize,
    /// 第 0 层最大出边。
    pub m0: usize,
    /// 入口节点（空图为 [`EMPTY`]）。
    pub entry: u32,
    /// 当前最高层（空图为 -1）。
    pub max_level: i32,
    /// 每节点层数。
    pub levels: Vec<u8>,
    /// 第 0 层链接矩阵，count × m0。
    pub layer0: Vec<u32>,
    /// 上层链接，多数节点为空。
    pub upper: Vec<UpperLinks>,
}

/// 层数采样上限。m=16 时 P(level≥32) 约 1e-4×再乘 32 层衰减，足够稀疏。
const MAX_LEVEL: u32 = 32;

impl HnswGraph {
    /// 新建空图。
    pub fn new(count: u32, params: &HnswParams) -> Self {
        let m0 = params.m0;
        HnswGraph {
            count,
            m: params.m,
            m0,
            entry: EMPTY,
            max_level: -1,
            levels: vec![0; count as usize],
            layer0: vec![EMPTY; count as usize * m0],
            upper: (0..count).map(|_| UpperLinks::default()).collect(),
        }
    }

    /// 某节点在 level 层的出边切片（非空前缀，EMPTY 结尾）。
    pub fn links(&self, node: u32, level: i32) -> &[u32] {
        if level == 0 {
            let start = node as usize * self.m0;
            return &self.layer0[start..start + self.m0];
        }
        let n = node as usize;
        if (self.levels[n] as i32) < level {
            return &[];
        }
        self.upper[n]
            .layers
            .get((level - 1) as usize)
            .map_or(&[], |v| v.as_slice())
    }

    fn links_len(&self, node: u32, level: i32) -> usize {
        if level == 0 {
            let row = self.links(node, 0);
            row.iter().position(|&x| x == EMPTY).unwrap_or(row.len())
        } else {
            self.links(node, level).len()
        }
    }

    fn set_links(&mut self, node: u32, level: i32, links: &[u32]) {
        if level == 0 {
            let start = node as usize * self.m0;
            self.layer0[start..start + self.m0].fill(EMPTY);
            self.layer0[start..start + links.len()].copy_from_slice(links);
        } else {
            let li = (level - 1) as usize;
            let n = node as usize;
            if self.upper[n].layers.len() <= li {
                self.upper[n].layers.resize(li + 1, Vec::new());
            }
            self.upper[n].layers[li] = links.to_vec();
        }
    }

    fn push_link(&mut self, node: u32, level: i32, target: u32) {
        if level == 0 {
            let start = node as usize * self.m0;
            let row = &mut self.layer0[start..start + self.m0];
            if let Some(pos) = row.iter().position(|&x| x == EMPTY) {
                row[pos] = target;
            }
        } else {
            let li = (level - 1) as usize;
            let n = node as usize;
            if self.upper[n].layers.len() <= li {
                self.upper[n].layers.resize(li + 1, Vec::new());
            }
            self.upper[n].layers[li].push(target);
        }
    }

    /// 逐点插入构建。调用方保证 idx 递增或任意序均可（在线算法）。
    pub fn insert(
        &mut self,
        idx: u32,
        data: &SegmentCore,
        metric: Metric,
        params: &HnswParams,
        rng: &mut Rng,
    ) {
        let level = sample_level(rng, self.m).min(MAX_LEVEL);
        self.levels[idx as usize] = level as u8;

        if self.entry == EMPTY {
            self.entry = idx;
            self.max_level = level as i32;
            // 首节点无邻居可连，但上层链接槽位仍需分配（links 按层数索引）。
            if level > 0 {
                self.upper[idx as usize]
                    .layers
                    .resize(level as usize, Vec::new());
            }
            return;
        }

        let query = data.vector(idx);
        let mut eps = vec![self.entry];
        let mut scratch = traversal::Scratch::new();

        // 高层贪心下钻到 level+1。
        let mut cur = self.max_level;
        while cur > level as i32 {
            let mut ep = eps[0];
            let ctx = traversal::Ctx {
                graph: self,
                data,
                metric,
            };
            traversal::greedy_descend(&ctx, query, &mut ep, cur);
            eps = vec![ep];
            cur -= 1;
        }

        // 自 min(level, max_level) 到 0 逐层连接。高于 max_level 的层此刻
        // 只有自己，不存在可连节点（也不能给别的节点挂跨层链接）。
        cur = (level as i32).min(self.max_level);
        while cur >= 0 {
            let cands = {
                let ctx = traversal::Ctx {
                    graph: self,
                    data,
                    metric,
                };
                traversal::search_layer(
                    &ctx,
                    query,
                    &eps,
                    params.ef_construct.max(1),
                    cur,
                    None,
                    &mut scratch,
                )
            };
            let m_max = params.max_links(cur);
            let selected = {
                let ctx = traversal::Ctx {
                    graph: self,
                    data,
                    metric,
                };
                traversal::select_heuristic(&ctx, &cands, m_max)
            };
            self.set_links(idx, cur, &selected);

            for &n in &selected {
                let deg = self.links_len(n, cur);
                if deg < m_max {
                    self.push_link(n, cur, idx);
                } else {
                    // 满了: 以 n 为中心重选（含新节点 idx）。
                    let ctx = traversal::Ctx {
                        graph: self,
                        data,
                        metric,
                    };
                    let mut pool: Vec<(u32, f32)> = self
                        .links(n, cur)
                        .iter()
                        .copied()
                        .take_while(|&x| x != EMPTY)
                        .map(|c| (c, traversal::node_key(&ctx, n, c)))
                        .collect();
                    pool.push((idx, traversal::node_key(&ctx, n, idx)));
                    pool.sort_by(|a, b| a.1.total_cmp(&b.1));
                    let newl = traversal::select_heuristic(&ctx, &pool, m_max);
                    self.set_links(n, cur, &newl);
                }
            }

            eps = cands.iter().map(|(id, _)| *id).collect();
            cur -= 1;
        }

        if level as i32 > self.max_level {
            self.entry = idx;
            self.max_level = level as i32;
        }
    }
}

/// 论文层分布: level = floor(-ln(u) / ln(m))。
fn sample_level(rng: &mut Rng, m: usize) -> u32 {
    let ml = 1.0 / (m.max(2) as f32).ln();
    (-rng.next_f32().ln() * ml).floor() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_sampling_sane() {
        let mut r = Rng::new(1);
        let mut histogram = [0usize; 8];
        for _ in 0..100_000 {
            let l = sample_level(&mut r, 16);
            assert!(l < 8, "level {l} unexpectedly high");
            histogram[l as usize] += 1;
        }
        // 层 1 约占 1/16，层 0 占绝对多数。
        assert!(histogram[0] > 90_000);
        assert!(histogram[1] > 3_000 && histogram[1] < 10_000);
        assert!(histogram[2] < histogram[1]);
    }
}

//! HNSW 索引（自研实现，hnswlib 论文语义）。
//!
//! - 构建: 逐点在线插入，层分配 `-ln(u)/ln(m)`，启发式邻居选择（keepPruned）
//! - 删除: 不改图，查询期 `pass` 跳过墓碑（与段的 dels 边车一致）
//! - 检索: 自入口贪心下钻，第 0 层 best-first; 结果不足 k 时 ef 倍增重试
//!   （最多 3 次），缓解高选择性过滤下的欠召回
//! - 持久化: `index.bin`（见 [`HnswIndex::serialize`]）
//!
//! 构建是确定性的（固定种子），同数据同参数 → 同图。

pub mod graph;
pub mod params;
pub mod rng;
pub mod traversal;

use std::sync::Arc;

use crate::error::{Error, Result};
use crate::kernel::Metric;
use crate::segment::core::SegmentCore;

use self::graph::HnswGraph;
pub use self::params::HnswParams;
use self::rng::Rng;
use self::traversal::EMPTY;
use super::AnnIndex;

const MAGIC: u32 = 0x5644_4248; // "VDBH"
const FORMAT_VERSION: u32 = 1;

/// HNSW 索引实例。
pub struct HnswIndex {
    graph: HnswGraph,
    core: Arc<SegmentCore>,
    metric: Metric,
    params: HnswParams,
}

impl HnswIndex {
    /// 从段数据构建。内存中全量逐点插入。
    pub fn build(core: Arc<SegmentCore>, metric: Metric, params: HnswParams) -> Result<Self> {
        let total = core.len();
        if total >= EMPTY as usize {
            return Err(Error::Invalid("segment too large for hnsw".into()));
        }
        let count = total as u32;
        let mut g = HnswGraph::new(count, &params);
        let mut rng = Rng::new(0x5DEE_CE66);
        for idx in 0..count {
            g.insert(idx, &core, metric, &params, &mut rng);
        }
        Ok(HnswIndex {
            graph: g,
            core,
            metric,
            params,
        })
    }

    /// 参数副本。
    pub fn params(&self) -> HnswParams {
        self.params.clone()
    }

    /// 序列化图结构（不含向量，向量在段文件里）。
    pub fn serialize(&self) -> Vec<u8> {
        let g = &self.graph;
        let mut out = Vec::with_capacity(24 + g.levels.len() + g.layer0.len() * 4);
        out.extend_from_slice(&MAGIC.to_le_bytes());
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&g.count.to_le_bytes());
        out.extend_from_slice(&(g.m as u32).to_le_bytes());
        out.extend_from_slice(&(g.m0 as u32).to_le_bytes());
        out.extend_from_slice(&g.entry.to_le_bytes());
        out.extend_from_slice(&g.max_level.to_le_bytes());
        out.extend_from_slice(&(self.params.ef_construct as u32).to_le_bytes());
        out.extend_from_slice(&(self.params.ef_search as u32).to_le_bytes());
        out.extend_from_slice(&g.levels);

        for v in &g.layer0 {
            out.extend_from_slice(&v.to_le_bytes());
        }

        let upper_nodes: Vec<u32> = (0..g.count).filter(|&n| g.levels[n as usize] > 0).collect();
        out.extend_from_slice(&(upper_nodes.len() as u32).to_le_bytes());
        for n in upper_nodes {
            let n = n as usize;
            out.extend_from_slice(&(n as u32).to_le_bytes());
            out.push(g.levels[n]);
            for layer in &g.upper[n].layers {
                out.extend_from_slice(&(layer.len() as u16).to_le_bytes());
                for v in layer {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        out
    }

    /// 从 `index.bin` 字节与段数据装载。
    pub fn load(bytes: Vec<u8>, core: Arc<SegmentCore>, metric: Metric) -> Result<Self> {
        let mut r = Reader { buf: bytes, pos: 0 };
        let magic = r.u32()?;
        if magic != MAGIC {
            return Err(Error::Invalid(format!("bad index magic {magic:#x}")));
        }
        let version = r.u32()?;
        if version != FORMAT_VERSION {
            return Err(Error::Invalid(format!(
                "index format version {version} != {FORMAT_VERSION}"
            )));
        }
        let count = r.u32()?;
        let m = r.u32()? as usize;
        let m0 = r.u32()? as usize;
        let entry = r.u32()?;
        let max_level = r.i32()?;
        let ef_construct = r.u32()? as usize;
        let ef_search = r.u32()? as usize;

        if count as usize != core.len() {
            return Err(Error::Invalid(format!(
                "index count {count} != segment count {}",
                core.len()
            )));
        }

        let levels = r.bytes(count as usize)?.to_vec();
        let layer0_len = count as usize * m0;
        let mut layer0 = Vec::with_capacity(layer0_len);
        for _ in 0..layer0_len {
            layer0.push(r.u32()?);
        }

        let mut upper = (0..count)
            .map(|_| graph::UpperLinks::default())
            .collect::<Vec<_>>();
        let upper_count = r.u32()?;
        for _ in 0..upper_count {
            let n = r.u32()? as usize;
            let layer_count = r.u8()?;
            if n as u32 >= count || layer_count as i32 > max_level {
                return Err(Error::Invalid("index upper links out of range".into()));
            }
            for _ in 0..layer_count {
                let len = r.u16()? as usize;
                let mut v = Vec::with_capacity(len);
                for _ in 0..len {
                    v.push(r.u32()?);
                }
                upper[n].layers.push(v);
            }
        }
        if r.remaining() != 0 {
            return Err(Error::Invalid(format!(
                "index has {} trailing bytes",
                r.remaining()
            )));
        }

        Ok(HnswIndex {
            graph: HnswGraph {
                count,
                m,
                m0,
                entry,
                max_level,
                levels,
                layer0,
                upper,
            },
            core,
            metric,
            params: HnswParams {
                m,
                m0,
                ef_construct,
                ef_search,
            },
        })
    }

    fn query(
        &self,
        query: &[f32],
        k: usize,
        pass: Option<&dyn Fn(u32) -> bool>,
    ) -> Vec<(u32, f32)> {
        if k == 0 || self.graph.entry == EMPTY {
            return Vec::new();
        }
        let ctx = traversal::Ctx {
            graph: &self.graph,
            data: &self.core,
            metric: self.metric,
        };
        let mut ep = self.graph.entry;
        for l in (1..=self.graph.max_level).rev() {
            traversal::greedy_descend(&ctx, query, &mut ep, l);
        }

        let cap = self.params.ef_search.max(k).max(self.graph.count as usize);
        let mut ef = self.params.ef_search.max(k);
        let mut results = traversal::search_layer(&ctx, query, &[ep], ef, 0, pass);
        let mut tries = 0;
        while results.len() < k && ef < cap && tries < 3 {
            ef = (ef * 2).min(cap);
            results = traversal::search_layer(&ctx, query, &[ep], ef, 0, pass);
            tries += 1;
        }

        results
            .into_iter()
            .take(k)
            .map(|(id, key)| (id, traversal::to_score(self.metric, key)))
            .collect()
    }
}

impl AnnIndex for HnswIndex {
    fn search(
        &self,
        query: &[f32],
        k: usize,
        pass: Option<&dyn Fn(u32) -> bool>,
    ) -> Vec<(u32, f32)> {
        self.query(query, k, pass)
    }

    fn est_bytes(&self) -> u64 {
        let g = &self.graph;
        let upper: usize = g
            .upper
            .iter()
            .map(|u| u.layers.iter().map(|l| l.len() * 4).sum::<usize>())
            .sum();
        (g.levels.len() + g.layer0.len() * 4 + upper) as u64
    }

    fn kind(&self) -> &'static str {
        "hnsw"
    }
}

struct Reader {
    buf: Vec<u8>,
    pos: usize,
}

impl Reader {
    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        if self.remaining() < n {
            return Err(Error::Invalid("index file truncated".into()));
        }
        let s = self.pos;
        self.pos += n;
        Ok(&self.buf[s..s + n])
    }
    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(self.u32()? as i32)
    }
    fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn bytes(&mut self, n: usize) -> Result<&[u8]> {
        self.take(n)
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::flat;

    fn gen_core(n: usize, dim: usize, seed: u64) -> SegmentCore {
        let mut rng = Rng::new(seed);
        let mut vectors = Vec::with_capacity(n * dim);
        let mut norms = Vec::new();
        let ids = (0..n as i64).map(crate::id::ExternalId::Num).collect();
        for _ in 0..n {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
            if dim >= 2 {
                // 稍微聚簇，制造有意义的邻居结构。
                v[0] += (v[1] * 3.0).sin();
            }
            norms.push(crate::kernel::norm(&v));
            vectors.extend_from_slice(&v);
        }
        SegmentCore {
            dim,
            vectors,
            norms,
            ids,
            payloads: vec![None; n],
        }
    }

    fn recall_at(hnsw: &[(u32, f32)], truth: &[(u32, f32)], k: usize) -> f32 {
        let got: std::collections::HashSet<u32> = hnsw.iter().map(|(i, _)| *i).collect();
        let want: std::collections::HashSet<u32> = truth.iter().take(k).map(|(i, _)| *i).collect();
        got.intersection(&want).count() as f32 / want.len().max(1) as f32
    }

    #[test]
    fn small_recall_and_roundtrip() {
        let core = Arc::new(gen_core(600, 32, 7));
        let metric = Metric::L2;
        let params = HnswParams::default();
        let idx = HnswIndex::build(core.clone(), metric, params).unwrap();

        let mut rng = Rng::new(99);
        let mut recalls = Vec::new();
        for _ in 0..20 {
            let q: Vec<f32> = (0..32).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
            let truth = flat::search(metric, &core.vectors, &core.norms, core.dim, &q, None, 10);
            let got = idx.query(&q, 10, None);
            recalls.push(recall_at(&got, &truth, 10));
        }
        let avg: f32 = recalls.iter().sum::<f32>() / recalls.len() as f32;
        assert!(avg >= 0.95, "recall@10 too low: {avg:.3}");

        // 序列化往返结果一致。
        let bytes = idx.serialize();
        let re = HnswIndex::load(bytes, core.clone(), metric).unwrap();
        let q: Vec<f32> = (0..32).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        let a = idx.query(&q, 10, None);
        let b = re.query(&q, 10, None);
        assert_eq!(a, b);

        // 墓碑过滤: 排除前一半。
        let got = idx.query(&q, 10, Some(&|i| i >= 300));
        assert!(got.iter().all(|(i, _)| *i >= 300));
    }

    #[test]
    fn empty_core_is_noop() {
        let core = Arc::new(SegmentCore {
            dim: 4,
            vectors: vec![],
            norms: vec![],
            ids: vec![],
            payloads: vec![],
        });
        let idx = HnswIndex::build(core, Metric::Dot, HnswParams::default()).unwrap();
        assert!(idx.query(&[1.0, 0.0, 0.0, 0.0], 5, None).is_empty());
        let re = HnswIndex::load(idx.serialize(), idx.core.clone(), Metric::Dot);
        assert!(re.is_ok());
    }

    #[test]
    fn corrupt_index_rejected() {
        let core = Arc::new(gen_core(10, 4, 3));
        let idx = HnswIndex::build(core.clone(), Metric::L2, HnswParams::default()).unwrap();
        let mut bytes = idx.serialize();
        bytes[0] ^= 0xFF; // 破坏 magic
        assert!(HnswIndex::load(bytes, core, Metric::L2).is_err());
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod graph_invariant {
    use super::*;

    /// 回归: 新节点层数高于 max_level 时，曾给入口节点挂过跨层链接，
    /// 导致 levels 与上层链接向量长度不一致、序列化往返失败。
    #[test]
    fn levels_match_layer_vectors_and_roundtrip() {
        let core = Arc::new(SegmentCore {
            dim: 32,
            vectors: (0..600 * 32).map(|i| (i as f32 * 0.01).sin()).collect(),
            norms: vec![1.0; 600],
            ids: (0..600).map(crate::id::ExternalId::Num).collect(),
            payloads: vec![None; 600],
        });
        let idx = HnswIndex::build(core, Metric::L2, HnswParams::default()).unwrap();
        let g = &idx.graph;
        for n in 0..g.count as usize {
            assert_eq!(
                g.levels[n] as usize,
                g.upper[n].layers.len(),
                "node {n}: levels {} vs layers {}",
                g.levels[n],
                g.upper[n].layers.len()
            );
        }
        let bytes = idx.serialize();
        HnswIndex::load(bytes, idx.core.clone(), Metric::L2).unwrap();
    }
}

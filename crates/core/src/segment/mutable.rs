//! 内存可变段。所有写入先到这里，flush 时固化为不可变段。
//! 覆盖写复用原行（内部序号不变），删除打墓碑。

use roaring::RoaringBitmap;
use serde_json::Value;

use crate::id::ExternalId;
use crate::kernel;
use crate::segment::Point;

/// 内存可变段。
pub struct MutableSegment {
    dim: usize,
    vectors: Vec<f32>,
    norms: Vec<f32>,
    ids: Vec<ExternalId>,
    payloads: Vec<Option<Value>>,
    payload_bytes: usize,
    ext2int: std::collections::HashMap<ExternalId, u32>,
    deleted: RoaringBitmap,
}

impl MutableSegment {
    /// 新建空段。dim 来自集合配置。
    pub fn new(dim: usize) -> Self {
        MutableSegment {
            dim,
            vectors: Vec::new(),
            norms: Vec::new(),
            ids: Vec::new(),
            payloads: Vec::new(),
            payload_bytes: 0,
            ext2int: std::collections::HashMap::new(),
            deleted: RoaringBitmap::new(),
        }
    }

    /// 向量维度。
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// 插入或覆盖一个点（同 ID 覆盖并复活）。
    pub fn upsert(&mut self, p: &Point) {
        if let Some(&idx) = self.ext2int.get(&p.id) {
            let idx = idx as usize;
            let start = idx * self.dim;
            self.vectors[start..start + self.dim].copy_from_slice(&p.vector);
            self.norms[idx] = kernel::norm(&p.vector);
            if let Some(old) = &self.payloads[idx] {
                self.payload_bytes -= old.to_string().len();
            }
            if let Some(new) = &p.payload {
                self.payload_bytes += new.to_string().len();
            }
            self.payloads[idx] = p.payload.clone();
            self.deleted.remove(idx as u32);
        } else {
            let idx = self.ids.len() as u32;
            self.vectors.extend_from_slice(&p.vector);
            self.norms.push(kernel::norm(&p.vector));
            self.ids.push(p.id.clone());
            if let Some(v) = &p.payload {
                self.payload_bytes += v.to_string().len();
            }
            self.payloads.push(p.payload.clone());
            self.ext2int.insert(p.id.clone(), idx);
        }
    }

    /// 删除。返回该 ID 之前是否存活。
    pub fn delete(&mut self, id: &ExternalId) -> bool {
        match self.ext2int.get(id) {
            Some(&idx) if !self.deleted.contains(idx) => {
                self.deleted.insert(idx);
                true
            }
            _ => false,
        }
    }

    /// 整体替换 payload。返回该 ID 是否存活。
    pub fn set_payload(&mut self, id: &ExternalId, payload: Option<Value>) -> bool {
        match self.ext2int.get(id) {
            Some(&idx) if !self.deleted.contains(idx) => {
                let idx = idx as usize;
                if let Some(old) = &self.payloads[idx] {
                    self.payload_bytes -= old.to_string().len();
                }
                if let Some(new) = &payload {
                    self.payload_bytes += new.to_string().len();
                }
                self.payloads[idx] = payload;
                true
            }
            _ => false,
        }
    }

    /// ID 对应的存活内部序号。
    pub fn alive_index(&self, id: &ExternalId) -> Option<u32> {
        let idx = self.ext2int.get(id)?;
        if self.deleted.contains(*idx) {
            None
        } else {
            Some(*idx)
        }
    }

    /// 是否含该 ID 的行（无论存活还是墓碑）。upsert 会复用并复活墓碑行。
    pub fn contains(&self, id: &ExternalId) -> bool {
        self.ext2int.contains_key(id)
    }

    /// 点数据（含墓碑行，调用方需检查 alive）。
    pub fn point(&self, idx: u32) -> Point {
        let i = idx as usize;
        Point {
            id: self.ids[i].clone(),
            vector: self.vectors[i * self.dim..(i + 1) * self.dim].to_vec(),
            payload: self.payloads[i].clone(),
        }
    }

    /// 是否完全为空（无任何行）。
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// 总行数（含墓碑）。
    pub fn total(&self) -> usize {
        self.ids.len()
    }

    /// 存活行数。
    pub fn alive(&self) -> u64 {
        self.ids.len() as u64 - self.deleted.len()
    }

    /// 近似内存占用（向量 + payload），供自动 flush 阈值用。
    pub fn approx_bytes(&self) -> u64 {
        (self.vectors.len() * 4 + self.payload_bytes) as u64
    }

    /// 清空。flush 交换后调用。
    pub fn clear(&mut self) {
        self.vectors.clear();
        self.norms.clear();
        self.ids.clear();
        self.payloads.clear();
        self.payload_bytes = 0;
        self.ext2int.clear();
        self.deleted.clear();
    }

    // 供 flat 扫描与段固化读取。
    /// 扁平向量存储。
    pub fn vectors(&self) -> &[f32] {
        &self.vectors
    }
    /// 预计算范数，与向量行对齐。
    pub fn norms(&self) -> &[f32] {
        &self.norms
    }
    /// ID 列表。
    pub fn ids(&self) -> &[ExternalId] {
        &self.ids
    }
    /// payload 列表。
    pub fn payloads(&self) -> &[Option<Value>] {
        &self.payloads
    }
    /// 墓碑位图。
    pub fn deleted(&self) -> &RoaringBitmap {
        &self.deleted
    }
}

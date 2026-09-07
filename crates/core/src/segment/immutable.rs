//! 不可变段: flush 落盘后的段目录，大文件只读; 删除与 payload 更新写小边车。
//!
//! 目录布局（位于 `<root>/segments/seg-<id>/`）:
//! - `meta.json`            {seg_id, dim, count}
//! - `vectors.bin`          行主序 f32 LE
//! - `norms.bin`            每行一个 f32（L2 范数）
//! - `ids.jsonl`            每行一个外部 ID（JSON number 或 string）
//! - `payloads.jsonl`       每行一个 payload（或 null）
//! - `dels.bin`             墓碑 roaring bitmap（按需创建）
//! - `payloads.overlay.jsonl`  payload 覆盖层（按需创建）
//!
//! 边车用 tmp+rename 原子替换; 打开时全量载入内存。

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::id::ExternalId;
use crate::segment::Point;
use crate::segment::mutable::MutableSegment;
use crate::storage::{Fs, atomic_write};

const META: &str = "meta.json";
const VECTORS: &str = "vectors.bin";
const NORMS: &str = "norms.bin";
const IDS: &str = "ids.jsonl";
const PAYLOADS: &str = "payloads.jsonl";
const DELS: &str = "dels.bin";
const OVERLAY: &str = "payloads.overlay.jsonl";

#[derive(Serialize, Deserialize)]
struct Meta {
    seg_id: u64,
    dim: usize,
    count: usize,
}

/// 不可变段。
pub struct ImmutableSegment {
    /// 段 ID（目录名后缀，全局唯一）。
    pub seg_id: u64,
    dim: usize,
    vectors: Vec<f32>,
    norms: Vec<f32>,
    ids: Vec<ExternalId>,
    payloads: Vec<Option<Value>>,
    ext2int: HashMap<ExternalId, u32>,
    dir: PathBuf,
    fs: std::sync::Arc<dyn Fs>,
    deleted: RwLock<RoaringBitmap>,
    overlay: RwLock<HashMap<ExternalId, Option<Value>>>,
}

/// 写段前的输入快照（从可变段或合并结果借用）。
pub struct SegmentData<'a> {
    /// 扁平向量。
    pub vectors: &'a [f32],
    /// 预计算范数。
    pub norms: &'a [f32],
    /// ID 列表。
    pub ids: &'a [ExternalId],
    /// payload 列表。
    pub payloads: &'a [Option<Value>],
    /// 墓碑（固化时直接带入）。
    pub deleted: &'a RoaringBitmap,
    /// payload 覆盖层（压实时带入; 可变段来源时为 None）。
    pub overlay: Option<&'a HashMap<ExternalId, Option<Value>>>,
}

impl<'a> From<&'a MutableSegment> for SegmentData<'a> {
    fn from(m: &'a MutableSegment) -> Self {
        SegmentData {
            vectors: m.vectors(),
            norms: m.norms(),
            ids: m.ids(),
            payloads: m.payloads(),
            deleted: m.deleted(),
            overlay: None,
        }
    }
}

impl ImmutableSegment {
    /// 将数据写入 `dir`（调用方负责先 create_dir_all，建议指向 `seg-<id>.tmp`，
    /// 成功后由调用方 rename 到正式目录）。写入即含初始墓碑（可变段带来的）。
    pub fn write_new(
        fs: &dyn Fs,
        dir: &Path,
        seg_id: u64,
        dim: usize,
        data: &SegmentData<'_>,
    ) -> Result<()> {
        let count = data.ids.len();
        atomic_write(
            fs,
            &dir.join(META),
            serde_json::to_vec(&Meta { seg_id, dim, count })?.as_slice(),
        )?;

        let mut vbytes = Vec::with_capacity(data.vectors.len() * 4);
        for v in data.vectors {
            vbytes.extend_from_slice(&v.to_le_bytes());
        }
        atomic_write(fs, &dir.join(VECTORS), &vbytes)?;

        let mut nbytes = Vec::with_capacity(data.norms.len() * 4);
        for n in data.norms {
            nbytes.extend_from_slice(&n.to_le_bytes());
        }
        atomic_write(fs, &dir.join(NORMS), &nbytes)?;

        let mut ids = String::new();
        for id in data.ids {
            ids.push_str(&serde_json::to_string(id)?);
            ids.push('\n');
        }
        atomic_write(fs, &dir.join(IDS), ids.as_bytes())?;

        let mut pls = String::new();
        for p in data.payloads {
            pls.push_str(&serde_json::to_string(p)?);
            pls.push('\n');
        }
        atomic_write(fs, &dir.join(PAYLOADS), pls.as_bytes())?;

        if !data.deleted.is_empty() {
            let mut buf = Vec::new();
            data.deleted.serialize_into(&mut buf)?;
            atomic_write(fs, &dir.join(DELS), &buf)?;
        }
        if let Some(ov) = data.overlay {
            if !ov.is_empty() {
                write_overlay_file(fs, &dir.join(OVERLAY), ov)?;
            }
        }
        Ok(())
    }

    /// 从段目录载入（含边车）。
    pub fn open(fs: std::sync::Arc<dyn Fs>, dir: &Path) -> Result<Self> {
        let seg_id = dir
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_prefix("seg-"))
            .and_then(|n| n.parse::<u64>().ok())
            .ok_or_else(|| Error::Invalid(format!("bad segment dir name: {}", dir.display())))?;

        let meta_bytes = read_file(fs.as_ref(), &dir.join(META))?;
        let meta: Meta = serde_json::from_slice(&meta_bytes)?;
        if meta.seg_id != seg_id {
            return Err(Error::Invalid(format!(
                "segment id mismatch in {}: dir says {seg_id}, meta says {}",
                dir.display(),
                meta.seg_id
            )));
        }

        let vbytes = read_file(fs.as_ref(), &dir.join(VECTORS))?;
        if vbytes.len() % 4 != 0 {
            return Err(Error::Invalid(format!("{} size not aligned", VECTORS)));
        }
        let vectors = vbytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<_>>();
        if vectors.len() != meta.count * meta.dim {
            return Err(Error::Invalid(format!(
                "vectors.bin size {} != count*dim {} in {}",
                vectors.len(),
                meta.count * meta.dim,
                dir.display()
            )));
        }

        let nbytes = read_file(fs.as_ref(), &dir.join(NORMS))?;
        let norms = nbytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<_>>();
        if norms.len() != meta.count {
            return Err(Error::Invalid(format!(
                "norms.bin size mismatch in {}",
                dir.display()
            )));
        }

        let ids: Vec<ExternalId> = parse_jsonl(&read_file(fs.as_ref(), &dir.join(IDS))?)?;
        let payloads_raw: Vec<Value> = parse_jsonl(&read_file(fs.as_ref(), &dir.join(PAYLOADS))?)?;
        if ids.len() != meta.count || payloads_raw.len() != meta.count {
            return Err(Error::Invalid(format!(
                "jsonl count mismatch in {}",
                dir.display()
            )));
        }
        let payloads = payloads_raw
            .into_iter()
            .map(|v| if v.is_null() { None } else { Some(v) })
            .collect::<Vec<_>>();

        let deleted = if fs.exists(&dir.join(DELS)) {
            let buf = read_file(fs.as_ref(), &dir.join(DELS))?;
            RoaringBitmap::deserialize_from(buf.as_slice())?
        } else {
            RoaringBitmap::new()
        };

        let overlay = if fs.exists(&dir.join(OVERLAY)) {
            read_overlay_file(&read_file(fs.as_ref(), &dir.join(OVERLAY))?)?
        } else {
            HashMap::new()
        };

        let ext2int = ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), i as u32))
            .collect();

        Ok(ImmutableSegment {
            seg_id,
            dim: meta.dim,
            vectors,
            norms,
            ids,
            payloads,
            ext2int,
            dir: dir.to_path_buf(),
            fs,
            deleted: RwLock::new(deleted),
            overlay: RwLock::new(overlay),
        })
    }

    /// 存活内部序号。
    pub fn alive_index(&self, id: &ExternalId) -> Option<u32> {
        let idx = *self.ext2int.get(id)?;
        if self.is_alive(idx) { Some(idx) } else { None }
    }

    /// 批量打墓碑并持久化边车（一次写盘）。幂等。
    pub fn tombstone_many(&self, idxs: &[u32]) -> Result<()> {
        if idxs.is_empty() {
            return Ok(());
        }
        let bitmap = {
            let mut del = lock_write(&self.deleted)?;
            del.extend(idxs.iter().copied());
            del.clone()
        };
        let mut buf = Vec::new();
        bitmap.serialize_into(&mut buf)?;
        atomic_write(self.fs.as_ref(), &self.dir.join(DELS), &buf)
    }

    /// 批量写 payload 覆盖层（None 表示清除该点 payload）。一次重写边车。
    pub fn set_overlay_batch(&self, entries: &[(ExternalId, Option<Value>)]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let map = {
            let mut ov = lock_write(&self.overlay)?;
            for (id, payload) in entries {
                ov.insert(id.clone(), payload.clone());
            }
            ov.clone()
        };
        write_overlay_file(self.fs.as_ref(), &self.dir.join(OVERLAY), &map)
    }

    /// 点数据（应用覆盖层），向量拷贝。
    pub fn point(&self, idx: u32) -> Point {
        let i = idx as usize;
        let payload = self.payload_of(i);
        Point {
            id: self.ids[i].clone(),
            vector: self.vectors[i * self.dim..(i + 1) * self.dim].to_vec(),
            payload,
        }
    }

    /// 应用覆盖层后的 payload。
    pub fn payload_of(&self, i: usize) -> Option<Value> {
        if let Ok(ov) = self.overlay.read() {
            match ov.get(&self.ids[i]) {
                Some(v) => v.clone(),
                None => self.payloads[i].clone(),
            }
        } else {
            self.payloads[i].clone()
        }
    }

    /// 行是否存活。
    pub fn is_alive(&self, idx: u32) -> bool {
        self.deleted
            .read()
            .map(|d| !d.contains(idx))
            .unwrap_or(false)
    }

    /// 总行数（含墓碑）。
    pub fn total(&self) -> usize {
        self.ids.len()
    }

    /// 存活行数。
    pub fn alive(&self) -> u64 {
        let del = self.deleted.read().map(|d| d.len()).unwrap_or(0);
        self.ids.len() as u64 - del
    }

    // 供扫描/合并读取。
    /// 扁平向量存储。
    pub fn vectors(&self) -> &[f32] {
        &self.vectors
    }
    /// 范数列。
    pub fn norms(&self) -> &[f32] {
        &self.norms
    }
    /// ID 列表。
    pub fn ids(&self) -> &[ExternalId] {
        &self.ids
    }
    /// 维度。
    pub fn dim(&self) -> usize {
        self.dim
    }
    /// payload 覆盖层快照（压实时带入新段）。
    pub fn overlay_snapshot(&self) -> HashMap<ExternalId, Option<Value>> {
        self.overlay.read().map(|ov| ov.clone()).unwrap_or_default()
    }
}

fn read_file(fs: &dyn Fs, path: &Path) -> Result<Vec<u8>> {
    let f = fs.open_rw(path)?;
    let len = f.len()?;
    let mut buf = vec![0u8; len as usize];
    f.read_at(&mut buf, 0)?;
    Ok(buf)
}

fn parse_jsonl<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<Vec<T>> {
    let text =
        std::str::from_utf8(bytes).map_err(|e| Error::Invalid(format!("jsonl not utf-8: {e}")))?;
    let mut out = Vec::new();
    for line in text.lines() {
        if line.is_empty() {
            continue;
        }
        out.push(serde_json::from_str(line)?);
    }
    Ok(out)
}

fn write_overlay_file(
    fs: &dyn Fs,
    path: &Path,
    map: &HashMap<ExternalId, Option<Value>>,
) -> Result<()> {
    let mut buf = String::new();
    for (id, payload) in map {
        buf.push_str(&serde_json::to_string(&OverlayLine {
            id: id.clone(),
            payload: payload.clone(),
        })?);
        buf.push('\n');
    }
    atomic_write(fs, path, buf.as_bytes())
}

fn read_overlay_file(bytes: &[u8]) -> Result<HashMap<ExternalId, Option<Value>>> {
    let lines: Vec<OverlayLine> = parse_jsonl(bytes)?;
    Ok(lines.into_iter().map(|l| (l.id, l.payload)).collect())
}

#[derive(Serialize, Deserialize)]
struct OverlayLine {
    id: ExternalId,
    payload: Option<Value>,
}

// RwLock 的 poison 在这里是不可恢复的 bug 信号，转成错误而非 panic。
fn lock_write<'a, T>(l: &'a RwLock<T>) -> Result<std::sync::RwLockWriteGuard<'a, T>> {
    l.write()
        .map_err(|e| Error::Invalid(format!("lock poisoned: {e}")))
}

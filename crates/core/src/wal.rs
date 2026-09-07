//! 先行日志（ADR 0002）。所有变更先追加 WAL、刷盘，再应用内存。
//!
//! 帧格式: `[len: u32 LE][crc32(body): u32 LE][body]`，body 为 JSON 记录
//! `{"lsn": N, "op": "...", ...}`。尾部不完整帧（断电）在重放时截断。
//!
//! lsn 全库单调递增; manifest.last_flushed_lsn 之前的记录已固化进段，重放时跳过。

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::collection::CollectionConfig;
use crate::error::{Error, Result};
use crate::id::ExternalId;
use crate::segment::Point;
use crate::storage::StorageBackend;

/// 单帧 body 上限，防御损坏的 len 字段导致的巨量分配。
const MAX_FRAME: u32 = 256 << 20;

/// WAL 记录的操作。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Op {
    /// 建集合。
    CreateCollection {
        /// 集合名。
        name: String,
        /// 集合配置。
        config: CollectionConfig,
    },
    /// 删集合。
    DropCollection {
        /// 集合名。
        name: String,
    },
    /// 插入或覆盖点。
    Upsert {
        /// 集合名。
        collection: String,
        /// 点列表。
        points: Vec<Point>,
    },
    /// 删除点。
    Delete {
        /// 集合名。
        collection: String,
        /// 点 ID 列表。
        ids: Vec<ExternalId>,
    },
    /// 设置/合并 payload（整体覆盖该点的 payload）。
    SetPayload {
        /// 集合名。
        collection: String,
        /// 点 ID 列表。
        ids: Vec<ExternalId>,
        /// 新 payload。
        payload: Value,
    },
    /// 清空 payload。
    ClearPayload {
        /// 集合名。
        collection: String,
        /// 点 ID 列表。
        ids: Vec<ExternalId>,
    },
}

/// 一条 WAL 记录。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalRecord {
    /// 全局单调序号。
    pub lsn: u64,
    /// 操作内容。
    #[serde(flatten)]
    pub op: Op,
}

/// 打开时重放的结果。
pub struct ReplayReport {
    /// 按序解析出的记录（含 lsn <= floor 的，调用方自行过滤）。
    pub records: Vec<WalRecord>,
    /// 下一个应分配的 lsn。
    pub next_lsn: u64,
    /// 尾部损坏信息（已截断），None 表示完整。
    pub corrupt_tail: Option<(u64, String)>,
}

/// WAL 句柄。
pub struct Wal {
    backend: Box<dyn StorageBackend>,
    offset: u64,
    next_lsn: u64,
}

impl Wal {
    /// 打开（或新建）WAL 并重放。floor_lsn 取 manifest.last_flushed_lsn，
    /// 保证截断后重新计数也不会与历史 lsn 冲突。
    pub fn open(backend: Box<dyn StorageBackend>, floor_lsn: u64) -> Result<(Self, ReplayReport)> {
        let len = backend.len()?;
        let mut buf = Vec::new();
        if len > 0 {
            buf.resize(len as usize, 0);
            backend.read_at(&mut buf, 0)?;
        }

        let mut records = Vec::new();
        let mut offset: u64 = 0;
        let mut corrupt_tail = None;
        let mut last_lsn = 0u64;

        loop {
            match parse_frame(&buf, offset as usize) {
                FrameParse::Ok(rec, next) => {
                    last_lsn = rec.lsn.max(last_lsn);
                    records.push(rec);
                    offset = next as u64;
                }
                FrameParse::Eof => break,
                FrameParse::Corrupt(reason) => {
                    corrupt_tail = Some((offset, reason));
                    break;
                }
            }
        }
        if let Some((bad, _)) = corrupt_tail {
            backend.truncate(bad)?;
            backend.sync()?;
        }

        let next_lsn = floor_lsn.max(last_lsn + 1);
        let wal = Wal {
            offset,
            next_lsn,
            backend,
        };
        Ok((
            wal,
            ReplayReport {
                records,
                next_lsn,
                corrupt_tail,
            },
        ))
    }

    /// 追加一条记录并刷盘，返回分配的 lsn。
    pub fn append(&mut self, op: &Op) -> Result<u64> {
        let rec = WalRecord {
            lsn: self.next_lsn,
            op: op.clone(),
        };
        let body = serde_json::to_vec(&rec)?;
        if body.len() > MAX_FRAME as usize {
            return Err(Error::Invalid(format!(
                "wal record too large: {} bytes",
                body.len()
            )));
        }
        let crc = crc32fast::hash(&body);
        let mut frame = Vec::with_capacity(body.len() + 8);
        frame.extend_from_slice(&(body.len() as u32).to_le_bytes());
        frame.extend_from_slice(&crc.to_le_bytes());
        frame.extend_from_slice(&body);

        // 先写帧、fsync，成功后才推 next_lsn: 半途失败留下的是可截断的坏尾部。
        self.backend.append(&frame)?;
        self.backend.sync()?;
        self.offset += frame.len() as u64;
        let lsn = rec.lsn;
        self.next_lsn += 1;
        Ok(lsn)
    }

    /// 当前待分配 lsn。
    pub fn next_lsn(&self) -> u64 {
        self.next_lsn
    }

    /// flush 落段完成后调用: 清空日志，offset 归零（next_lsn 继续前进）。
    pub fn reset(&mut self) -> Result<()> {
        self.backend.truncate(0)?;
        self.backend.sync()?;
        self.offset = 0;
        Ok(())
    }
}

enum FrameParse {
    Ok(WalRecord, usize),
    Eof,
    Corrupt(String),
}

fn parse_frame(buf: &[u8], pos: usize) -> FrameParse {
    if pos == buf.len() {
        return FrameParse::Eof;
    }
    if pos + 8 > buf.len() {
        return FrameParse::Corrupt("truncated frame header".into());
    }
    let len = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap_or([0; 4])) as usize;
    if len > MAX_FRAME as usize {
        return FrameParse::Corrupt(format!("frame length {len} out of range"));
    }
    let crc = u32::from_le_bytes(buf[pos + 4..pos + 8].try_into().unwrap_or([0; 4]));
    let end = pos + 8 + len;
    if end > buf.len() {
        return FrameParse::Corrupt("truncated frame body".into());
    }
    let body = &buf[pos + 8..end];
    if crc32fast::hash(body) != crc {
        return FrameParse::Corrupt("crc mismatch".into());
    }
    match serde_json::from_slice(body) {
        Ok(rec) => FrameParse::Ok(rec, end),
        Err(e) => FrameParse::Corrupt(format!("bad record body: {e}")),
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Metric;
    use crate::storage::Fs as _;
    use crate::storage::memory::MemoryFs;
    use std::path::Path;

    fn wal_with_records() -> (MemoryFs, Vec<u64>) {
        let fs = MemoryFs::new();
        let backend = fs.create(Path::new("/wal.log").as_ref()).unwrap();
        let (mut wal, _) = Wal::open(backend, 0).unwrap();
        let ops = [
            Op::CreateCollection {
                name: "c".into(),
                config: CollectionConfig::new(4, Metric::Dot).unwrap(),
            },
            Op::Upsert {
                collection: "c".into(),
                points: vec![Point::new(
                    1,
                    vec![0.1, 0.2, 0.3, 0.4],
                    Some(serde_json::json!({"k": 1})),
                )],
            },
        ];
        let lsns = ops.iter().map(|op| wal.append(op).unwrap()).collect();
        (fs, lsns)
    }

    #[test]
    fn append_then_replay() {
        let (fs, lsns) = wal_with_records();
        let backend = fs.open_rw(Path::new("/wal.log").as_ref()).unwrap();
        let (_, report) = Wal::open(backend, 0).unwrap();
        assert_eq!(report.records.len(), 2);
        assert_eq!(report.records[0].lsn, lsns[0]);
        assert_eq!(report.next_lsn, lsns[1] + 1);
        assert!(report.corrupt_tail.is_none());
    }

    #[test]
    fn corrupt_tail_truncated_and_floor_respected() {
        let (fs, lsns) = wal_with_records();
        let path = Path::new("/wal.log");
        // 掐掉最后一个字节，模拟断电时的部分写入。
        let f = fs.open_rw(path).unwrap();
        let len = f.len().unwrap();
        f.truncate(len - 1).unwrap();
        drop(f);

        let backend = fs.open_rw(path).unwrap();
        let (_, report) = Wal::open(backend, 100).unwrap();
        assert_eq!(report.records.len(), 1); // 第二帧损坏被截断
        assert_eq!(report.next_lsn, 100); // floor 优先于日志内最大 lsn
        assert!(report.corrupt_tail.is_some());
        assert_eq!(lsns.len(), 2);

        // 截断后的文件可继续追加。
        let backend = fs.open_rw(path).unwrap();
        let (mut wal, _) = Wal::open(backend, 100).unwrap();
        wal.append(&Op::DropCollection { name: "c".into() })
            .unwrap();
    }

    #[test]
    fn crc_mismatch_detected() {
        let (fs, _) = wal_with_records();
        let path = Path::new("/wal.log");
        let f = fs.open_rw(path).unwrap();
        let len = f.len().unwrap() as usize;
        let mut buf = vec![0u8; len];
        f.read_at(&mut buf, 0).unwrap();
        // 篡改第一帧 body 的最后一个字节。
        let last = len - 1;
        buf[last] ^= 0xff;
        f.write_at(&buf, 0).unwrap();
        drop(f);

        let backend = fs.open_rw(path).unwrap();
        let (_, report) = Wal::open(backend, 0).unwrap();
        // 只损坏了第二帧，第一帧应完好保留。
        assert_eq!(report.records.len(), 1);
        assert!(report.corrupt_tail.is_some());
    }
}

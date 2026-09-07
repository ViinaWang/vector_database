#![allow(clippy::unwrap_used)]

//! 故障注入: Fs 层注入 IO 错误，验证"写入失败后重开，状态自洽"。
//! 这类测试是段式不可变设计存在的核心理由（ADR 0002）。

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::json;
use tempfile::TempDir;
use vectordb_core::storage::file::NativeFs;
use vectordb_core::storage::{Fs, StorageBackend};
use vectordb_core::{CollectionConfig, Database, DatabaseOptions, Error, Metric, Point};

/// 前缀透传，armed 后所有操作返回 DiskFull。
struct FaultFs {
    inner: NativeFs,
    armed: Arc<AtomicBool>,
}

impl FaultFs {
    fn check(armed: &AtomicBool) -> std::io::Result<()> {
        if armed.load(Ordering::SeqCst) {
            Err(std::io::Error::new(
                std::io::ErrorKind::StorageFull,
                "injected",
            ))
        } else {
            Ok(())
        }
    }
}

impl Fs for FaultFs {
    fn create(&self, path: &Path) -> std::io::Result<Box<dyn StorageBackend>> {
        Self::check(&self.armed)?;
        Ok(Box::new(FaultyBackend {
            inner: self.inner.create(path)?,
            armed: self.armed.clone(),
        }))
    }
    fn open_rw(&self, path: &Path) -> std::io::Result<Box<dyn StorageBackend>> {
        Self::check(&self.armed)?;
        Ok(Box::new(FaultyBackend {
            inner: self.inner.open_rw(path)?,
            armed: self.armed.clone(),
        }))
    }
    fn rename(&self, from: &Path, to: &Path) -> std::io::Result<()> {
        Self::check(&self.armed)?;
        self.inner.rename(from, to)
    }
    fn remove_file(&self, path: &Path) -> std::io::Result<()> {
        Self::check(&self.armed)?;
        self.inner.remove_file(path)
    }
    fn remove_dir_all(&self, path: &Path) -> std::io::Result<()> {
        Self::check(&self.armed)?;
        self.inner.remove_dir_all(path)
    }
    fn create_dir_all(&self, path: &Path) -> std::io::Result<()> {
        Self::check(&self.armed)?;
        self.inner.create_dir_all(path)
    }
    fn exists(&self, path: &Path) -> bool {
        self.inner.exists(path)
    }
    fn list_dir(&self, path: &Path) -> std::io::Result<Vec<PathBuf>> {
        Self::check(&self.armed)?;
        self.inner.list_dir(path)
    }
}

/// 写路径的后端包装: armed 时写失败、读放行。
/// WAL 句柄在 open 时就拿到底层后端，注入必须在这一层才打得到追加写。
struct FaultyBackend {
    inner: Box<dyn StorageBackend>,
    armed: Arc<AtomicBool>,
}

impl StorageBackend for FaultyBackend {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
        self.inner.read_at(buf, offset)
    }
    fn write_at(&self, buf: &[u8], offset: u64) -> std::io::Result<()> {
        FaultFs::check(&self.armed)?;
        self.inner.write_at(buf, offset)
    }
    fn len(&self) -> std::io::Result<u64> {
        self.inner.len()
    }
    fn truncate(&self, new_len: u64) -> std::io::Result<()> {
        FaultFs::check(&self.armed)?;
        self.inner.truncate(new_len)
    }
    fn sync(&self) -> std::io::Result<()> {
        FaultFs::check(&self.armed)?;
        self.inner.sync()
    }
}

fn pt(id: i64) -> Point {
    Point::new(id, vec![id as f32, 1.0], Some(json!({"n": id})))
}

fn fault_fs() -> (Arc<FaultFs>, Arc<AtomicBool>) {
    let armed = Arc::new(AtomicBool::new(false));
    (
        Arc::new(FaultFs {
            inner: NativeFs,
            armed: armed.clone(),
        }),
        armed,
    )
}

#[test]
fn failed_write_leaves_consistent_state() {
    let dir = TempDir::new().unwrap();
    let (fs, armed) = fault_fs();

    let db = Database::open_with(fs, dir.path(), DatabaseOptions::default()).unwrap();
    let c = db
        .create_collection("c", CollectionConfig::new(2, Metric::L2).unwrap())
        .unwrap();
    c.upsert(&[pt(1)]).unwrap();

    // 注入故障: 后续写全部失败。
    armed.store(true, Ordering::SeqCst);
    let err = c.upsert(&[pt(2)]).unwrap_err();
    assert!(matches!(err, Error::Io(ref e) if e.kind() == std::io::ErrorKind::StorageFull));

    // 故障保持下读仍可用。
    assert_eq!(c.count(), 1);
    armed.store(false, Ordering::SeqCst);
    drop(db);

    // 用健康 Fs 重开: point 2 要么完整存在（重放成功）要么不存在，不能半存在。
    let db = Database::open(dir.path()).unwrap();
    let c = db.collection("c").unwrap();
    let pts = c.get(&[1.into(), 2.into()]).unwrap();
    assert!(pts[0].is_some(), "已确认写入的 point 1 必须在");
    match pts[1].as_ref() {
        None => {}
        Some(p) => {
            assert_eq!(p.vector, vec![2.0, 1.0]);
            assert_eq!(p.payload, Some(json!({"n": 2})));
        }
    }
    // 无论哪条路径，计数与 get 一致。
    let n = c.count();
    let alive = pts.iter().filter(|p| p.is_some()).count() as u64;
    assert_eq!(n, alive);
}

#[test]
fn failed_flush_keeps_wal_intact() {
    let dir = TempDir::new().unwrap();
    let (fs, armed) = fault_fs();
    let db = Database::open_with(fs, dir.path(), DatabaseOptions::default()).unwrap();
    let c = db
        .create_collection("c", CollectionConfig::new(2, Metric::L2).unwrap())
        .unwrap();
    c.upsert(&[pt(1), pt(2)]).unwrap();

    // flush 中途失败（段落盘阶段）。
    armed.store(true, Ordering::SeqCst);
    let _ = db.flush().unwrap_err();
    armed.store(false, Ordering::SeqCst);
    drop(db);

    // 重开: WAL 未截断，重放恢复全部数据。
    let db = Database::open(dir.path()).unwrap();
    let c = db.collection("c").unwrap();
    assert_eq!(c.count(), 2);
}

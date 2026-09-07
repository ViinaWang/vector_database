//! 引擎: 对外 API 与写路径编排。
//!
//! 并发模型: 所有写操作（含 flush/compact）持有 `write_mu` 串行;
//! 读操作只拿对应 collection 的 view 读锁，可与写并发。flush 的段落盘
//! 在 view 写锁内完成，期间该集合的读会被阻塞——M1 的已知取舍。
//!
//! 崩溃一致性: 写 = WAL 追加+fsync → 应用内存; flush = 段落盘 →
//! manifest 原子替换 → WAL 截断。恢复语义见 ADR 0002。

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock, Weak};

use serde_json::Value;

use crate::collection::{CollectionConfig, IndexKind};
use crate::error::{Error, Result};
use crate::filter::Condition;
use crate::id::ExternalId;
use crate::index::flat;
use crate::index::hnsw::HnswIndex;
use crate::kernel::Metric;
use crate::manifest::{CollectionEntry, Manifest};
use crate::query::{Query, ScoredPoint, ScrollPage};
use crate::segment::Point;
use crate::segment::immutable::{ImmutableSegment, REBUILD_TOMBSTONE_RATIO};
use crate::segment::mutable::MutableSegment;
use crate::storage::{Fs, memory::MemoryFs};
use crate::wal::{Op, Wal, WalRecord};

const SEGMENTS_DIR: &str = "segments";
const WAL_FILE: &str = "wal.log";
const LOCK_FILE: &str = "vectordb.lock";

/// 数据库级选项。
#[derive(Debug, Clone)]
pub struct DatabaseOptions {
    /// 可变段点数达到该值触发自动 flush。
    pub auto_flush_points: usize,
    /// 可变段近似字节数达到该值触发自动 flush。
    pub auto_flush_bytes: u64,
}

impl Default for DatabaseOptions {
    fn default() -> Self {
        DatabaseOptions {
            auto_flush_points: 50_000,
            auto_flush_bytes: 256 << 20,
        }
    }
}

/// 打开的数据库句柄，可克隆共享。drop 即关闭（进程内文件锁释放）。
#[derive(Clone)]
pub struct Database {
    inner: Arc<DbInner>,
}

/// 集合句柄，可克隆共享。数据库关闭后再调用返回错误。
#[derive(Clone)]
pub struct Collection {
    inner: Arc<CollInner>,
}

struct CollView {
    mutable: MutableSegment,
    immutables: Vec<Arc<ImmutableSegment>>,
}

struct CollInner {
    db: Weak<DbInner>,
    name: String,
    config: CollectionConfig,
    view: RwLock<CollView>,
}

struct DbInner {
    fs: Arc<dyn Fs>,
    root: PathBuf,
    options: DatabaseOptions,
    write_mu: Mutex<()>,
    wal: Mutex<Wal>,
    manifest: Mutex<Manifest>,
    collections: RwLock<HashMap<String, Arc<CollInner>>>,
    next_seg_id: AtomicU64,
    recovery_clean: bool,
    /// native 下持有目录锁防止双开; wasm 恒 None。
    _dir_lock: Option<std::fs::File>,
}

// ---------------------------------------------------------------- Database

impl Database {
    /// 打开（或创建）一个数据库目录，默认选项。
    /// 持有目录锁，同进程/跨进程重复打开同一目录会返回 [`Error::AlreadyOpen`]。
    /// native 专用; wasm 用 [`Database::open_memory`] 或 [`Database::open_with`]。
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open(path: impl AsRef<Path>) -> Result<Database> {
        let root = path.as_ref();
        std::fs::create_dir_all(root)?;
        let dir_lock = lock_directory(root)?;
        Database::open_inner(
            Arc::new(crate::storage::file::NativeFs),
            root,
            DatabaseOptions::default(),
            dir_lock,
        )
    }

    /// 纯内存数据库（不持久化、无目录锁），适合临时数据、测试与 wasm。
    pub fn open_memory() -> Result<Database> {
        Database::open_inner(
            Arc::new(MemoryFs::new()),
            Path::new("/memdb"),
            DatabaseOptions::default(),
            None,
        )
    }

    /// 自定义 Fs 与选项打开（不做目录锁——调用方自理互斥）。
    pub fn open_with(fs: Arc<dyn Fs>, root: &Path, options: DatabaseOptions) -> Result<Database> {
        Database::open_inner(fs, root, options, None)
    }

    fn open_inner(
        fs: Arc<dyn Fs>,
        root: &Path,
        options: DatabaseOptions,
        dir_lock: Option<std::fs::File>,
    ) -> Result<Database> {
        fs.create_dir_all(root)?;
        fs.create_dir_all(&root.join(SEGMENTS_DIR))?;

        let manifest = Manifest::load(fs.as_ref(), root)?.unwrap_or_default();

        let wal_path = root.join(WAL_FILE);
        let wal_backend = if fs.exists(&wal_path) {
            fs.open_rw(&wal_path)?
        } else {
            fs.create(&wal_path)?
        };
        let (wal, report) = Wal::open(wal_backend, manifest.last_flushed_lsn)?;

        let segs_dir = root.join(SEGMENTS_DIR);
        let mut coll_specs: Vec<(String, CollectionConfig, Vec<Arc<ImmutableSegment>>)> =
            Vec::new();
        for entry in &manifest.collections {
            let mut immutables = Vec::new();
            for seg_id in &entry.segments {
                let dir = segs_dir.join(format!("seg-{seg_id}"));
                immutables.push(Arc::new(ImmutableSegment::open(
                    fs.clone(),
                    &dir,
                    entry.config.metric,
                )?));
            }
            coll_specs.push((entry.name.clone(), entry.config.clone(), immutables));
        }

        let inner = Arc::new(DbInner {
            fs,
            root: root.to_path_buf(),
            options,
            write_mu: Mutex::new(()),
            wal: Mutex::new(wal),
            manifest: Mutex::new(manifest),
            collections: RwLock::new(HashMap::new()),
            next_seg_id: AtomicU64::new(0),
            recovery_clean: report.corrupt_tail.is_none(),
            _dir_lock: dir_lock,
        });
        for (name, config, immutables) in coll_specs {
            let coll = Arc::new(CollInner {
                db: Arc::downgrade(&inner),
                name,
                config: config.clone(),
                view: RwLock::new(CollView {
                    mutable: MutableSegment::new(config.dim),
                    immutables,
                }),
            });
            inner
                .collections
                .write()
                .map_err(poison)?
                .insert(coll.name.clone(), coll);
        }

        let max_seg = inner.manifest.lock().map_err(poison)?.max_segment_id();
        inner.next_seg_id.store(max_seg + 1, Ordering::SeqCst);

        gc_orphan_segments(&inner)?;
        replay_wal(&inner, &report.records)?;

        Ok(Database { inner })
    }

    /// 上次打开时 WAL 尾部是否被截断过（崩溃或磁盘问题的信号）。
    pub fn recovery_clean(&self) -> bool {
        self.inner.recovery_clean
    }

    /// 集合名列表（排序后）。
    pub fn collections(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .inner
            .collections
            .read()
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default();
        names.sort();
        names
    }

    /// 取集合句柄。
    pub fn collection(&self, name: &str) -> Option<Collection> {
        self.inner
            .collections
            .read()
            .ok()?
            .get(name)
            .map(|c| Collection { inner: c.clone() })
    }

    /// 建集合并返回句柄。
    pub fn create_collection(&self, name: &str, config: CollectionConfig) -> Result<Collection> {
        if name.is_empty() {
            return Err(Error::Invalid("collection name must be non-empty".into()));
        }
        let _g = self.inner.write_mu.lock().map_err(poison)?;
        if self.collection(name).is_some() {
            return Err(Error::CollectionExists(name.into()));
        }
        self.inner
            .wal
            .lock()
            .map_err(poison)?
            .append(&Op::CreateCollection {
                name: name.into(),
                config: config.clone(),
            })?;
        create_collection_locked(&self.inner, name, config)?;
        self.collection(name)
            .ok_or_else(|| Error::Invalid("collection vanished during create".into()))
    }

    /// 删除集合（含全部段目录）。
    pub fn drop_collection(&self, name: &str) -> Result<()> {
        let _g = self.inner.write_mu.lock().map_err(poison)?;
        if self.collection(name).is_none() {
            return Err(Error::CollectionNotFound(name.into()));
        }
        self.inner
            .wal
            .lock()
            .map_err(poison)?
            .append(&Op::DropCollection { name: name.into() })?;

        let old_dirs = {
            let mut mf = self.inner.manifest.lock().map_err(poison)?;
            let segs: Vec<u64> = mf
                .entry_mut(name)
                .map(|e| std::mem::take(&mut e.segments))
                .unwrap_or_default();
            mf.collections.retain(|c| c.name != name);
            mf.save(self.inner.fs.as_ref(), &self.inner.root)?;
            segs
        };
        self.inner.collections.write().map_err(poison)?.remove(name);
        for seg_id in old_dirs {
            let _ = self
                .inner
                .fs
                .remove_dir_all(&seg_dir(&self.inner.root, seg_id));
        }
        Ok(())
    }

    /// 将所有集合的可变段固化为不可变段，并截断 WAL。持有全局写锁。
    pub fn flush(&self) -> Result<()> {
        flush_locked(&self.inner)
    }
}

// -------------------------------------------------------------- Collection

impl Collection {
    /// 集合名。
    pub fn name(&self) -> &str {
        &self.inner.name
    }

    /// 集合配置。
    pub fn config(&self) -> &CollectionConfig {
        &self.inner.config
    }

    /// 存活点数。
    pub fn count(&self) -> u64 {
        self.inner
            .view
            .read()
            .map(|v| v.mutable.alive() + v.immutables.iter().map(|s| s.alive()).sum::<u64>())
            .unwrap_or(0)
    }

    /// 插入或覆盖点。同 ID 覆盖; 维度不匹配返回错误且整批不写入。
    /// 触达自动 flush 阈值时自动落段。
    pub fn upsert(&self, points: &[Point]) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }
        let db = self.db()?;
        let dim = self.inner.config.dim;
        for p in points {
            if p.vector.len() != dim {
                return Err(Error::DimensionMismatch {
                    expected: dim,
                    got: p.vector.len(),
                });
            }
        }

        let need_flush = {
            let _g = db.write_mu.lock().map_err(poison)?;
            db.wal.lock().map_err(poison)?.append(&Op::Upsert {
                collection: self.inner.name.clone(),
                points: points.to_vec(),
            })?;
            apply_upsert(&db, &self.inner, points)?
        };
        if need_flush {
            flush_locked(&db)?;
        }
        Ok(())
    }

    /// 删除点，返回删除前存活的个数。
    pub fn delete(&self, ids: &[ExternalId]) -> Result<usize> {
        if ids.is_empty() {
            return Ok(0);
        }
        let db = self.db()?;
        let _g = db.write_mu.lock().map_err(poison)?;
        db.wal.lock().map_err(poison)?.append(&Op::Delete {
            collection: self.inner.name.clone(),
            ids: ids.to_vec(),
        })?;
        apply_delete(&db, &self.inner, ids)
    }

    /// 整体覆盖这些点的 payload，返回命中个数。
    pub fn set_payload(&self, ids: &[ExternalId], payload: &Value) -> Result<usize> {
        self.payload_op(ids, Some(payload.clone()))
    }

    /// 清空这些点的 payload，返回命中个数。
    pub fn clear_payload(&self, ids: &[ExternalId]) -> Result<usize> {
        self.payload_op(ids, None)
    }

    fn payload_op(&self, ids: &[ExternalId], payload: Option<Value>) -> Result<usize> {
        if ids.is_empty() {
            return Ok(0);
        }
        let db = self.db()?;
        let _g = db.write_mu.lock().map_err(poison)?;
        let op = match &payload {
            Some(v) => Op::SetPayload {
                collection: self.inner.name.clone(),
                ids: ids.to_vec(),
                payload: v.clone(),
            },
            None => Op::ClearPayload {
                collection: self.inner.name.clone(),
                ids: ids.to_vec(),
            },
        };
        db.wal.lock().map_err(poison)?.append(&op)?;
        apply_payload(&db, &self.inner, ids, payload)
    }

    /// 按 ID 批量取点，顺序与入参一致，缺失为 None。
    pub fn get(&self, ids: &[ExternalId]) -> Result<Vec<Option<Point>>> {
        let view = self.inner.view.read().map_err(poison)?;
        Ok(ids.iter().map(|id| lookup_point(&view, id)).collect())
    }

    /// 分页遍历（可带过滤）。顺序: 可变段在前、不可变段新→旧。
    /// offset 计已产出的点; 只保证"两次调用之间无写入"时游标稳定。
    pub fn scroll(
        &self,
        offset: u64,
        limit: usize,
        filter: Option<&Condition>,
    ) -> Result<ScrollPage> {
        let limit = limit.max(1);
        let view = self.inner.view.read().map_err(poison)?;
        let mut points: Vec<Point> = Vec::with_capacity(limit);
        let mut cursor: u64 = 0;
        let mut next_offset: Option<u64> = None;

        let mut visit = |p: &Point| {
            if points.len() == limit {
                next_offset = Some(cursor);
                return false;
            }
            if cursor >= offset {
                points.push(p.clone());
            }
            cursor += 1;
            true
        };

        'outer: {
            let m = &view.mutable;
            for idx in 0..m.total() as u32 {
                if !m.deleted().contains(idx) {
                    let p = m.point(idx);
                    if filter.is_none_or(|c| c.eval(&p.id, p.payload.as_ref())) && !visit(&p) {
                        break 'outer;
                    }
                }
            }
            for imm in view.immutables.iter().rev() {
                for idx in 0..imm.total() as u32 {
                    if imm.is_alive(idx) {
                        let p = imm.point(idx);
                        if filter.is_none_or(|c| c.eval(&p.id, p.payload.as_ref())) && !visit(&p) {
                            break 'outer;
                        }
                    }
                }
            }
        }
        Ok(ScrollPage {
            points,
            next_offset,
        })
    }

    /// 邻近查询。当前为精确扫描（过滤在算距离前应用，无召回损失）。
    /// 分数阈值在 top_k 截断后应用。
    pub fn query(&self, q: &Query) -> Result<Vec<ScoredPoint>> {
        let metric = self.inner.config.metric;
        let dim = self.inner.config.dim;

        let (query_vec, exclude_id) = match (&q.vector, &q.id) {
            (Some(v), None) => (v.clone(), None),
            (None, Some(id)) => {
                let view = self.inner.view.read().map_err(poison)?;
                match lookup_point(&view, id) {
                    Some(p) => (p.vector, Some(id.clone())),
                    None => return Err(Error::Invalid(format!("query id {id} not found"))),
                }
            }
            _ => {
                return Err(Error::Invalid(
                    "exactly one of query.vector|query.id".into(),
                ));
            }
        };
        if query_vec.len() != dim {
            return Err(Error::DimensionMismatch {
                expected: dim,
                got: query_vec.len(),
            });
        }
        if q.top_k == 0 {
            return Err(Error::Invalid("top_k must be > 0".into()));
        }

        let view = self.inner.view.read().map_err(poison)?;

        #[derive(Clone, Copy)]
        enum Loc {
            Mut,
            Imm(usize),
        }
        let rank = |l: Loc| match l {
            Loc::Mut => 0usize,
            Loc::Imm(i) => i + 1,
        };

        let mut hits: Vec<(f32, u32, Loc)> = Vec::new();
        {
            let m = &view.mutable;
            let pass = |idx: u32| -> bool {
                if m.deleted().contains(idx) {
                    return false;
                }
                if Some(&m.ids()[idx as usize]) == exclude_id.as_ref() {
                    return false;
                }
                filter_pass(
                    q.filter.as_ref(),
                    &m.ids()[idx as usize],
                    m.payloads()[idx as usize].as_ref(),
                )
            };
            for (idx, score) in flat::search(
                metric,
                m.vectors(),
                m.norms(),
                dim,
                &query_vec,
                Some(&pass),
                q.top_k,
            ) {
                hits.push((score, idx, Loc::Mut));
            }
        }
        for (si, imm) in view.immutables.iter().enumerate() {
            let pass = |idx: u32| -> bool {
                if !imm.is_alive(idx) {
                    return false;
                }
                let id = &imm.ids()[idx as usize];
                if Some(id) == exclude_id.as_ref() {
                    return false;
                }
                filter_pass(q.filter.as_ref(), id, imm.payload_of(idx as usize).as_ref())
            };
            for (idx, score) in imm.search(&query_vec, q.top_k, Some(&pass)) {
                hits.push((score, idx, Loc::Imm(si)));
            }
        }

        hits.sort_by(|a, b| {
            let ord = match metric {
                Metric::L2 => a.0.total_cmp(&b.0),
                Metric::Cosine | Metric::Dot => b.0.total_cmp(&a.0),
            };
            ord.then(rank(a.2).cmp(&rank(b.2))).then(a.1.cmp(&b.1))
        });
        hits.truncate(q.top_k);

        let mut out = Vec::with_capacity(hits.len());
        for (score, idx, loc) in hits {
            if let Some(t) = q.score_threshold {
                if !metric.passes(score, t) {
                    continue;
                }
            }
            let point = match loc {
                Loc::Mut => view.mutable.point(idx),
                Loc::Imm(si) => view.immutables[si].point(idx),
            };
            out.push(ScoredPoint {
                id: point.id,
                score,
                payload: if q.with_payload { point.payload } else { None },
                vector: if q.with_vector {
                    Some(point.vector)
                } else {
                    None
                },
            });
        }
        Ok(out)
    }

    /// 把所有段合并为一个新段（清墓碑、固化覆盖层），原子替换 manifest。
    pub fn compact(&self) -> Result<()> {
        let db = self.db()?;
        let _g = db.write_mu.lock().map_err(poison)?;
        compact_locked(&db, &self.inner)
    }

    fn db(&self) -> Result<Arc<DbInner>> {
        self.inner
            .db
            .upgrade()
            .ok_or_else(|| Error::Invalid("database closed".into()))
    }
}

// ------------------------------------------------------------- 内部函数

impl std::fmt::Debug for Database {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Database")
            .field("collections", &self.collections())
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for Collection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Collection")
            .field("name", &self.inner.name)
            .field("config", &self.inner.config)
            .finish_non_exhaustive()
    }
}

fn poison<T>(e: std::sync::PoisonError<T>) -> Error {
    Error::Invalid(format!("lock poisoned: {e}"))
}

fn seg_dir(root: &Path, id: u64) -> PathBuf {
    root.join(SEGMENTS_DIR).join(format!("seg-{id}"))
}

impl DbInner {
    /// 内部取集合（Arc 克隆）。
    fn coll(&self, name: &str) -> Option<Arc<CollInner>> {
        self.collections.read().ok()?.get(name).cloned()
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn lock_directory(root: &Path) -> Result<Option<std::fs::File>> {
    use fs4::fs_std::FileExt;
    use std::fs::OpenOptions;
    let f = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(root.join(LOCK_FILE))?;
    match f.try_lock_exclusive() {
        Ok(true) => Ok(Some(f)),
        Ok(false) => Err(Error::AlreadyOpen(root.display().to_string())),
        Err(e) => Err(Error::AlreadyOpen(format!("{}: {e}", root.display()))),
    }
}

#[cfg(target_arch = "wasm32")]
fn lock_directory(_root: &Path) -> Result<Option<std::fs::File>> {
    Ok(None)
}

fn create_collection_locked(db: &Arc<DbInner>, name: &str, config: CollectionConfig) -> Result<()> {
    {
        let mut mf = db.manifest.lock().map_err(poison)?;
        mf.collections.push(CollectionEntry {
            name: name.into(),
            config: config.clone(),
            segments: Vec::new(),
        });
        mf.save(db.fs.as_ref(), &db.root)?;
    }
    let coll = Arc::new(CollInner {
        db: Arc::downgrade(db),
        name: name.into(),
        config: config.clone(),
        view: RwLock::new(CollView {
            mutable: MutableSegment::new(config.dim),
            immutables: Vec::new(),
        }),
    });
    db.collections
        .write()
        .map_err(poison)?
        .insert(name.into(), coll);
    Ok(())
}

/// 清理 segments/ 下未被 manifest 引用的目录（崩溃 flush 的残留）。
fn gc_orphan_segments(db: &Arc<DbInner>) -> Result<()> {
    let referenced: HashSet<u64> = db
        .manifest
        .lock()
        .map_err(poison)?
        .collections
        .iter()
        .flat_map(|c| c.segments.iter().copied())
        .collect();
    let segs_dir = db.root.join(SEGMENTS_DIR);
    for child in db.fs.list_dir(&segs_dir)? {
        let Some(fname) = child.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let keep = match fname.strip_suffix(".tmp") {
            Some(_) => false, // tmp 一律清理
            None => fname
                .strip_prefix("seg-")
                .and_then(|n| n.parse::<u64>().ok())
                .is_some_and(|id| referenced.contains(&id)),
        };
        if !keep {
            let _ = db.fs.remove_dir_all(&child);
        }
    }
    Ok(())
}

/// 重放 WAL（lsn > last_flushed_lsn 的部分）。apply_* 不再加锁:
/// 调用方（open）是单线程的。
fn replay_wal(db: &Arc<DbInner>, records: &[WalRecord]) -> Result<()> {
    let floor = db.manifest.lock().map_err(poison)?.last_flushed_lsn;
    for rec in records {
        if rec.lsn <= floor {
            continue;
        }
        let coll = |name: &str| -> Result<Arc<CollInner>> {
            db.collections
                .read()
                .map_err(poison)?
                .get(name)
                .cloned()
                .ok_or_else(|| {
                    Error::Invalid(format!(
                        "wal replay: collection {name:?} missing for op at lsn {}",
                        rec.lsn
                    ))
                })
        };
        match &rec.op {
            Op::CreateCollection { name, config } => {
                if db.coll(name).is_none() {
                    create_collection_locked(db, name, config.clone())?;
                }
            }
            Op::DropCollection { name } => {
                let old = {
                    let mut mf = db.manifest.lock().map_err(poison)?;
                    let segs = mf
                        .entry_mut(name)
                        .map(|e| std::mem::take(&mut e.segments))
                        .unwrap_or_default();
                    mf.collections.retain(|c| c.name != *name);
                    mf.save(db.fs.as_ref(), &db.root)?;
                    segs
                };
                db.collections.write().map_err(poison)?.remove(name);
                for id in old {
                    let _ = db.fs.remove_dir_all(&seg_dir(&db.root, id));
                }
            }
            Op::Upsert { collection, points } => {
                let c = coll(collection)?;
                apply_upsert(db, &c, points)?;
            }
            Op::Delete { collection, ids } => {
                let c = coll(collection)?;
                apply_delete(db, &c, ids)?;
            }
            Op::SetPayload {
                collection,
                ids,
                payload,
            } => {
                let c = coll(collection)?;
                apply_payload(db, &c, ids, Some(payload.clone()))?;
            }
            Op::ClearPayload { collection, ids } => {
                let c = coll(collection)?;
                apply_payload(db, &c, ids, None)?;
            }
        }
    }
    Ok(())
}

/// 应用 upsert 到可变段; 对存在于不可变段的同 ID 先打墓碑。
/// 返回是否触达自动 flush 阈值。调用方持有 write_mu（或恢复路径）。
fn apply_upsert(db: &Arc<DbInner>, coll: &Arc<CollInner>, points: &[Point]) -> Result<bool> {
    let mut view = coll.view.write().map_err(poison)?;
    for p in points {
        if !view.mutable.contains(&p.id) {
            // 可变段完全没有这个 ID: 若在不可变段存活则打墓碑（upsert 语义 = 覆盖）。
            let mut tombs: Vec<(usize, Vec<u32>)> = Vec::new();
            for (si, imm) in view.immutables.iter().enumerate() {
                if let Some(idx) = imm.alive_index(&p.id) {
                    tombs.push((si, vec![idx]));
                }
            }
            for (si, idxs) in tombs {
                view.immutables[si].tombstone_many(&idxs)?;
            }
        }
        view.mutable.upsert(p);
    }
    let need = view.mutable.total() >= db.options.auto_flush_points
        || view.mutable.approx_bytes() >= db.options.auto_flush_bytes;
    Ok(need)
}

fn apply_delete(db: &Arc<DbInner>, coll: &Arc<CollInner>, ids: &[ExternalId]) -> Result<usize> {
    let mut rebuild: Vec<usize> = Vec::new();
    let deleted = {
        let mut view = coll.view.write().map_err(poison)?;
        let mut deleted = 0usize;
        let mut tombs: Vec<(usize, Vec<u32>)> = Vec::new();
        for id in ids {
            if view.mutable.delete(id) {
                deleted += 1;
                continue;
            }
            // 可变段没有（或已删）: 找不可变段。至多一个存活（写路径不变量）。
            for (si, imm) in view.immutables.iter().enumerate() {
                if let Some(idx) = imm.alive_index(id) {
                    if let Some(slot) = tombs.iter_mut().find(|(s, _)| *s == si) {
                        slot.1.push(idx);
                    } else {
                        tombs.push((si, vec![idx]));
                    }
                    deleted += 1;
                    break;
                }
            }
        }
        for (si, idxs) in tombs {
            view.immutables[si].tombstone_many(&idxs)?;
        }
        // 墓碑占比超阈值的段: 遍历浪费已成规模，值得就地重建。
        for (si, imm) in view.immutables.iter().enumerate() {
            if imm.tombstone_ratio() > REBUILD_TOMBSTONE_RATIO {
                rebuild.push(si);
            }
        }
        deleted
    };
    for si in rebuild {
        rebuild_segment_locked(db, coll, si)?;
    }
    Ok(deleted)
}

/// 一个不可变段的一批 payload 覆盖层编辑。
type OverlayEdits = Vec<(ExternalId, Option<Value>)>;

fn apply_payload(
    _db: &Arc<DbInner>,
    coll: &Arc<CollInner>,
    ids: &[ExternalId],
    payload: Option<Value>,
) -> Result<usize> {
    let mut view = coll.view.write().map_err(poison)?;
    let mut hit = 0usize;
    let mut overlays: Vec<(usize, OverlayEdits)> = Vec::new();
    for id in ids {
        if view.mutable.set_payload(id, payload.clone()) {
            hit += 1;
            continue;
        }
        for (si, imm) in view.immutables.iter().enumerate() {
            if imm.alive_index(id).is_some() {
                let entry = (id.clone(), payload.clone());
                match overlays.iter_mut().find(|(s, _)| *s == si) {
                    Some(slot) => slot.1.push(entry),
                    None => overlays.push((si, vec![entry])),
                }
                hit += 1;
                break;
            }
        }
    }
    for (si, entries) in overlays {
        view.immutables[si].set_overlay_batch(&entries)?;
    }
    Ok(hit)
}

fn lookup_point(view: &CollView, id: &ExternalId) -> Option<Point> {
    if let Some(idx) = view.mutable.alive_index(id) {
        return Some(view.mutable.point(idx));
    }
    for imm in view.immutables.iter().rev() {
        if let Some(idx) = imm.alive_index(id) {
            return Some(imm.point(idx));
        }
    }
    None
}

fn filter_pass(cond: Option<&Condition>, id: &ExternalId, payload: Option<&Value>) -> bool {
    cond.is_none_or(|c| c.eval(id, payload))
}

/// flush: 段落盘 → manifest 原子替换 → WAL 截断 → 视图交换。
fn flush_locked(db: &Arc<DbInner>) -> Result<()> {
    let _g = db.write_mu.lock().map_err(poison)?;

    let mut names = db
        .collections
        .read()
        .map_err(poison)?
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    names.sort();

    let mut staged: Vec<(String, u64, Arc<ImmutableSegment>)> = Vec::new();
    for name in &names {
        let Some(coll) = db.coll(name) else { continue };
        let view = coll.view.write().map_err(poison)?;
        if view.mutable.is_empty() {
            continue;
        }
        let seg_id = db.next_seg_id.fetch_add(1, Ordering::SeqCst);
        let tmp = seg_dir(&db.root, seg_id).with_extension("tmp");
        let final_dir = seg_dir(&db.root, seg_id);
        db.fs.create_dir_all(&tmp)?;
        ImmutableSegment::write_new(
            db.fs.as_ref(),
            &tmp,
            seg_id,
            view.mutable.core(),
            view.mutable.deleted(),
            None,
        )?;
        db.fs.rename(&tmp, &final_dir)?;
        let seg = Arc::new(ImmutableSegment::open(
            db.fs.clone(),
            &final_dir,
            coll.config.metric,
        )?);
        build_segment_index(&coll.config, &seg)?;
        staged.push((name.clone(), seg_id, seg));
    }
    if staged.is_empty() {
        return Ok(());
    }

    {
        let mut mf = db.manifest.lock().map_err(poison)?;
        for (name, seg_id, _) in &staged {
            let entry = mf.entry_mut(name).ok_or_else(|| {
                Error::Invalid(format!("flush: collection {name:?} missing in manifest"))
            })?;
            entry.segments.push(*seg_id);
        }
        let next_lsn = db.wal.lock().map_err(poison)?.next_lsn();
        mf.last_flushed_lsn = next_lsn - 1;
        mf.save(db.fs.as_ref(), &db.root)?;
    }
    db.wal.lock().map_err(poison)?.reset()?;

    for (name, _, seg) in staged {
        let coll = db
            .coll(&name)
            .ok_or_else(|| Error::Invalid(format!("flush: collection {name:?} vanished")))?;
        let mut view = coll.view.write().map_err(poison)?;
        view.immutables.push(seg);
        view.mutable.clear();
    }
    Ok(())
}

/// compact: 全部段 + 可变段 → 一个新段，manifest 原子替换，旧段目录删除。
fn compact_locked(db: &Arc<DbInner>, coll: &Arc<CollInner>) -> Result<()> {
    let dim = coll.config.dim;
    let old_ids: Vec<u64>;
    let new_seg: Option<Arc<ImmutableSegment>>;

    {
        let mut view = coll.view.write().map_err(poison)?;
        let mut merged = MutableSegment::new(dim);
        // 时间序: 不可变段（旧→新），最后可变段（最新）。
        for imm in &view.immutables {
            for idx in 0..imm.total() as u32 {
                if imm.is_alive(idx) {
                    merged.upsert(&imm.point(idx)); // point() 已应用覆盖层
                }
            }
        }
        for idx in 0..view.mutable.total() as u32 {
            if !view.mutable.deleted().contains(idx) {
                merged.upsert(&view.mutable.point(idx));
            }
        }
        old_ids = view.immutables.iter().map(|s| s.seg_id).collect();

        if merged.is_empty() {
            new_seg = None;
            view.immutables.clear();
            view.mutable.clear();
        } else {
            let seg_id = db.next_seg_id.fetch_add(1, Ordering::SeqCst);
            let tmp = seg_dir(&db.root, seg_id).with_extension("tmp");
            let final_dir = seg_dir(&db.root, seg_id);
            db.fs.create_dir_all(&tmp)?;
            ImmutableSegment::write_new(
                db.fs.as_ref(),
                &tmp,
                seg_id,
                merged.core(),
                merged.deleted(),
                None,
            )?;
            db.fs.rename(&tmp, &final_dir)?;
            let seg = Arc::new(ImmutableSegment::open(
                db.fs.clone(),
                &final_dir,
                coll.config.metric,
            )?);
            build_segment_index(&coll.config, &seg)?;
            new_seg = Some(seg);
            view.immutables = new_seg.clone().into_iter().collect();
            view.mutable.clear();
        }
    }

    {
        let mut mf = db.manifest.lock().map_err(poison)?;
        let entry = mf
            .entry_mut(&coll.name)
            .ok_or_else(|| Error::CollectionNotFound(coll.name.clone()))?;
        entry.segments = new_seg.iter().map(|s| s.seg_id).collect();
        mf.save(db.fs.as_ref(), &db.root)?;
    }
    for id in old_ids {
        let _ = db.fs.remove_dir_all(&seg_dir(&db.root, id));
    }
    Ok(())
}

/// 按集合配置为不可变段构建 ANN 索引并安装。flush/compact/单段重建共用。
fn build_segment_index(config: &CollectionConfig, seg: &Arc<ImmutableSegment>) -> Result<()> {
    if let IndexKind::Hnsw { params } = &config.index {
        let idx = HnswIndex::build(seg.core(), config.metric, params.clone())?;
        let bytes = idx.serialize();
        seg.install_index(Arc::new(idx), Some(bytes))?;
    }
    Ok(())
}

/// 单段就地重建: 只保留存活行（覆盖层固化进新段），原子替换 manifest。
/// 与 compact 的区别是不合并全部段，只处理墓碑读放大已劣化的那一个。
/// 调用方持有 write_mu。
fn rebuild_segment_locked(db: &Arc<DbInner>, coll: &Arc<CollInner>, si: usize) -> Result<()> {
    let old_id: u64;
    let new_seg: Option<Arc<ImmutableSegment>>;
    {
        let mut view = coll.view.write().map_err(poison)?;
        let Some(old) = view.immutables.get(si).cloned() else {
            return Ok(());
        };
        let mut merged = MutableSegment::new(coll.config.dim);
        for idx in 0..old.total() as u32 {
            if old.is_alive(idx) {
                merged.upsert(&old.point(idx)); // point() 已应用覆盖层
            }
        }
        old_id = old.seg_id;

        if merged.is_empty() {
            new_seg = None;
            view.immutables.remove(si);
        } else {
            let seg_id = db.next_seg_id.fetch_add(1, Ordering::SeqCst);
            let tmp = seg_dir(&db.root, seg_id).with_extension("tmp");
            let final_dir = seg_dir(&db.root, seg_id);
            db.fs.create_dir_all(&tmp)?;
            ImmutableSegment::write_new(
                db.fs.as_ref(),
                &tmp,
                seg_id,
                merged.core(),
                merged.deleted(),
                None,
            )?;
            db.fs.rename(&tmp, &final_dir)?;
            let seg = Arc::new(ImmutableSegment::open(
                db.fs.clone(),
                &final_dir,
                coll.config.metric,
            )?);
            build_segment_index(&coll.config, &seg)?;
            new_seg = Some(seg.clone());
            view.immutables[si] = seg;
        }
    }
    {
        let mut mf = db.manifest.lock().map_err(poison)?;
        let entry = mf
            .entry_mut(&coll.name)
            .ok_or_else(|| Error::CollectionNotFound(coll.name.clone()))?;
        match &new_seg {
            Some(s) => {
                if let Some(pos) = entry.segments.iter().position(|&x| x == old_id) {
                    entry.segments[pos] = s.seg_id;
                } else {
                    entry.segments.push(s.seg_id);
                }
            }
            None => entry.segments.retain(|&x| x != old_id),
        }
        mf.save(db.fs.as_ref(), &db.root)?;
    }
    let _ = db.fs.remove_dir_all(&seg_dir(&db.root, old_id));
    Ok(())
}

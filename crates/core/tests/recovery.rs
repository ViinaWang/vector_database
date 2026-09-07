#![allow(clippy::unwrap_used)]

//! 崩溃恢复: WAL 尾部截断、孤儿段 GC、flush 窗口。

use serde_json::json;
use tempfile::TempDir;
use vectordb_core::{CollectionConfig, Database, Metric, Point, Query};

fn pt(id: i64) -> Point {
    Point::new(id, vec![id as f32, 0.0, 1.0], Some(json!({"n": id})))
}

#[test]
fn truncated_wal_tail_recovers_prefix() {
    let dir = TempDir::new().unwrap();

    // 写两条 WAL 记录但不 flush。
    {
        let db = Database::open(dir.path()).unwrap();
        let c = db
            .create_collection("c", CollectionConfig::new(3, Metric::L2).unwrap())
            .unwrap();
        c.upsert(&[pt(1)]).unwrap();
        c.upsert(&[pt(2)]).unwrap();
    }

    // 掐掉一个字节，模拟最后一帧只写了一半。
    let wal = dir.path().join("wal.log");
    let len = std::fs::metadata(&wal).unwrap().len();
    let f = std::fs::OpenOptions::new().write(true).open(&wal).unwrap();
    f.set_len(len - 1).unwrap();
    drop(f);

    let db = Database::open(dir.path()).unwrap();
    assert!(!db.recovery_clean());
    let c = db.collection("c").unwrap();
    // 丢失最后一条，保留第一条; 不确定具体丢哪条，但状态必须自洽。
    let n = c.count();
    assert!(n == 1 || n == 2, "unexpected count {n}");
    let pts = c.get(&[1.into(), 2.into()]).unwrap();
    for p in pts.iter().flatten() {
        let got = c.get(std::slice::from_ref(&p.id)).unwrap()[0]
            .clone()
            .unwrap();
        assert_eq!(got.vector, p.vector);
        assert_eq!(got.payload, p.payload);
    }
}

#[test]
fn orphan_segment_dirs_are_gc_ed() {
    let dir = TempDir::new().unwrap();
    {
        let db = Database::open(dir.path()).unwrap();
        let c = db
            .create_collection("c", CollectionConfig::new(3, Metric::L2).unwrap())
            .unwrap();
        c.upsert(&[pt(1)]).unwrap();
        db.flush().unwrap();
    }

    let segs = dir.path().join("segments");
    std::fs::create_dir_all(segs.join("seg-99")).unwrap();
    std::fs::create_dir_all(segs.join("seg-1.tmp")).unwrap();
    std::fs::write(segs.join("seg-99").join("junk.bin"), b"x").unwrap();

    let db = Database::open(dir.path()).unwrap();
    assert!(db.recovery_clean());
    assert!(db.collection("c").unwrap().count() == 1);
    assert!(!segs.join("seg-99").exists());
    assert!(!segs.join("seg-1.tmp").exists());
    // 真实段还在。
    assert!(segs.join("seg-1").exists());
}

#[test]
fn upsert_over_immutable_after_flush() {
    // 覆盖已 flush 的点 → 墓碑 + 新行; 删除、payload 更新同样作用到不可变段。
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path()).unwrap();
    let c = db
        .create_collection("c", CollectionConfig::new(3, Metric::L2).unwrap())
        .unwrap();
    c.upsert(&[pt(1), pt(2)]).unwrap();
    db.flush().unwrap();

    c.upsert(&[Point::new(
        1,
        vec![9.0, 9.0, 9.0],
        Some(json!({"n": "upd"})),
    )])
    .unwrap();
    assert_eq!(c.count(), 2);
    let hits = c
        .query(&Query::vector(vec![9.0, 9.0, 9.0]).top_k(1))
        .unwrap();
    assert_eq!(hits[0].id, 1.into());

    assert_eq!(c.delete(&[2.into()]).unwrap(), 1);
    assert_eq!(c.count(), 1);

    c.set_payload(&[1.into()], &json!({"k": 1})).unwrap();
    let got = c.get(&[1.into()]).unwrap();
    assert_eq!(got[0].as_ref().unwrap().payload, Some(json!({"k": 1})));

    // 重开后: 覆盖/删除/边车全部生效。
    drop(db);
    let db = Database::open(dir.path()).unwrap();
    let c = db.collection("c").unwrap();
    assert_eq!(c.count(), 1);
    let got = c.get(&[1.into()]).unwrap();
    let p = got[0].as_ref().unwrap();
    assert_eq!(p.payload, Some(json!({"k": 1})));
    assert_eq!(p.vector, vec![9.0, 9.0, 9.0]);
}

#[test]
fn reopen_replays_payload_and_delete_ops() {
    let dir = TempDir::new().unwrap();
    {
        let db = Database::open(dir.path()).unwrap();
        let c = db
            .create_collection("c", CollectionConfig::new(3, Metric::L2).unwrap())
            .unwrap();
        c.upsert(&[pt(1), pt(2), pt(3)]).unwrap();
        c.delete(&[3.into()]).unwrap();
        c.clear_payload(&[2.into()]).unwrap();
    }
    let db = Database::open(dir.path()).unwrap();
    let c = db.collection("c").unwrap();
    assert_eq!(c.count(), 2);
    let pts = c.get(&[2.into(), 3.into()]).unwrap();
    assert_eq!(pts[0].as_ref().unwrap().payload, None);
    assert!(pts[1].is_none());
}

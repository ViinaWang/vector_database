#![allow(clippy::unwrap_used)]

//! 端到端 CRUD / 查询 / 持久化行为。

use serde_json::json;
use tempfile::TempDir;
use vectordb_core::{CollectionConfig, Condition, Database, Metric, Point, Query};

fn coll(db: &Database) -> vectordb_core::Collection {
    db.create_collection("docs", CollectionConfig::new(3, Metric::Cosine).unwrap())
        .unwrap()
}

fn pt(id: impl Into<vectordb_core::ExternalId>, v: Vec<f32>, lang: &str, year: i64) -> Point {
    Point::new(id, v, Some(json!({"lang": lang, "year": year})))
}

#[test]
fn crud_roundtrip() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path()).unwrap();
    let c = coll(&db);

    c.upsert(&[
        pt(1, vec![1.0, 0.0, 0.0], "en", 2023),
        pt(2, vec![0.0, 1.0, 0.0], "zh", 2022),
        pt("s3", vec![0.9, 0.1, 0.0], "en", 2024),
    ])
    .unwrap();

    assert_eq!(c.count(), 3);
    let got = c.get(&[1.into(), "s3".into(), 99.into()]).unwrap();
    assert!(got[0].is_some());
    assert!(got[1].is_some());
    assert!(got[2].is_none());
    assert_eq!(
        got[1].as_ref().unwrap().payload,
        Some(json!({"lang":"en","year":2024}))
    );

    // top-2 邻近
    let hits = c
        .query(&Query::vector(vec![1.0, 0.0, 0.0]).top_k(2))
        .unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].id, 1.into());
    assert!(hits[0].score > hits[1].score);

    // 覆盖写
    c.upsert(&[pt(1, vec![0.0, 0.0, 1.0], "en", 2023)]).unwrap();
    assert_eq!(c.count(), 3);
    let hits = c
        .query(&Query::vector(vec![0.0, 0.0, 1.0]).top_k(1))
        .unwrap();
    assert_eq!(hits[0].id, 1.into());

    // 删除
    assert_eq!(c.delete(&[2.into()]).unwrap(), 1);
    assert_eq!(c.delete(&[2.into()]).unwrap(), 0);
    assert_eq!(c.count(), 2);
    assert!(c.get(&[2.into()]).unwrap()[0].is_none());

    // payload 局部操作
    assert_eq!(
        c.set_payload(
            &[1.into()],
            &json!({"lang": "en", "year": 2025, "tag": "x"})
        )
        .unwrap(),
        1
    );
    let hits = c
        .query(&Query::vector(vec![0.0, 0.0, 1.0]).filter(Condition::range(
            "year",
            None,
            Some(2025.0),
            None,
            None,
        )))
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(c.clear_payload(&[1.into()]).unwrap(), 1);
    let hits = c
        .query(&Query::vector(vec![0.0, 0.0, 1.0]).filter(Condition::exists("lang")))
        .unwrap();
    assert!(hits.iter().all(|h| h.id != 1.into()));
}

#[test]
fn dimension_mismatch_rejected_wholesale() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path()).unwrap();
    let c = coll(&db);
    let err = c
        .upsert(&[
            pt(1, vec![1.0, 0.0, 0.0], "en", 2020),
            pt(2, vec![1.0], "en", 2020),
        ])
        .unwrap_err();
    assert!(matches!(
        err,
        vectordb_core::Error::DimensionMismatch { .. }
    ));
    assert_eq!(c.count(), 0);
}

#[test]
fn duplicate_collection_rejected() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path()).unwrap();
    coll(&db);
    let err = db
        .create_collection("docs", CollectionConfig::new(3, Metric::L2).unwrap())
        .unwrap_err();
    assert!(matches!(err, vectordb_core::Error::CollectionExists(_)));
}

#[test]
fn reopen_after_flush_and_after_wal_only() {
    let dir = TempDir::new().unwrap();

    // 路径 A: flush 后重开
    {
        let db = Database::open(dir.path()).unwrap();
        let c = coll(&db);
        c.upsert(&[pt(1, vec![1.0, 0.0, 0.0], "en", 2020)]).unwrap();
        db.flush().unwrap();
        c.upsert(&[pt(2, vec![0.0, 1.0, 0.0], "zh", 2021)]).unwrap();
    }
    {
        let db = Database::open(dir.path()).unwrap();
        let c = db.collection("docs").unwrap();
        assert_eq!(c.count(), 2);
        let hits = c
            .query(&Query::vector(vec![0.0, 1.0, 0.0]).top_k(1))
            .unwrap();
        assert_eq!(hits[0].id, 2.into());
    }

    // 路径 B: 纯 WAL（未 flush）重开
    let dir2 = TempDir::new().unwrap();
    {
        let db = Database::open(dir2.path()).unwrap();
        let c = coll(&db);
        c.upsert(&[pt(7, vec![1.0, 1.0, 0.0], "en", 2020)]).unwrap();
    }
    {
        let db = Database::open(dir2.path()).unwrap();
        assert_eq!(db.collection("docs").unwrap().count(), 1);
    }
}

#[test]
fn scroll_pages_and_filters() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path()).unwrap();
    let c = coll(&db);
    let mut pts = Vec::new();
    for i in 0..10 {
        pts.push(pt(
            i,
            vec![i as f32, 0.0, 1.0],
            if i % 2 == 0 { "en" } else { "zh" },
            2020,
        ));
    }
    c.upsert(&pts).unwrap();

    let mut seen = Vec::new();
    let mut off = 0;
    while let Some(next) = {
        let page = c.scroll(off, 3, None).unwrap();
        seen.extend(page.points.iter().map(|p| p.id.clone()));
        page.next_offset
    } {
        off = next;
    }
    assert_eq!(seen.len(), 10);

    let page = c
        .scroll(0, 100, Some(&Condition::matches("lang", "en")))
        .unwrap();
    assert_eq!(page.points.len(), 5);
    assert!(page.next_offset.is_none());
}

#[test]
fn compact_merges_and_purges() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path()).unwrap();
    let c = coll(&db);
    for i in 0..5 {
        c.upsert(&[pt(i, vec![i as f32, 1.0, 0.0], "en", 2020)])
            .unwrap();
        db.flush().unwrap();
    }
    c.delete(&[0.into(), 2.into()]).unwrap();
    assert_eq!(c.count(), 3);

    c.compact().unwrap();
    assert_eq!(c.count(), 3);
    let hits = c
        .query(&Query::vector(vec![0.0, 1.0, 0.0]).top_k(10))
        .unwrap();
    assert_eq!(hits.len(), 3);
    assert!(
        hits.iter()
            .all(|h| h.id.as_num() != Some(0) && h.id.as_num() != Some(2))
    );

    // 压实后重开仍然一致
    drop(db);
    let db = Database::open(dir.path()).unwrap();
    assert_eq!(db.collection("docs").unwrap().count(), 3);
}

#[test]
fn memory_database_smoke() {
    let db = Database::open_memory().unwrap();
    let c = db
        .create_collection("m", CollectionConfig::new(2, Metric::Dot).unwrap())
        .unwrap();
    c.upsert(&[Point::new(1, vec![1.0, 2.0], None)]).unwrap();
    assert_eq!(c.count(), 1);
    let hits = c.query(&Query::vector(vec![1.0, 2.0]).top_k(1)).unwrap();
    assert_eq!(hits[0].id, 1.into());
}

#[test]
fn drop_collection_removes_everything() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path()).unwrap();
    coll(&db);
    db.collection("docs")
        .unwrap()
        .upsert(&[pt(1, vec![1.0, 0.0, 0.0], "en", 2020)])
        .unwrap();
    db.flush().unwrap();
    db.drop_collection("docs").unwrap();
    assert!(db.collection("docs").is_none());
    assert!(db.collections().is_empty());
    drop(db);
    let db = Database::open(dir.path()).unwrap();
    assert!(db.collection("docs").is_none());
}

#[test]
fn second_open_of_same_dir_fails() {
    let dir = TempDir::new().unwrap();
    let _db = Database::open(dir.path()).unwrap();
    let err = Database::open(dir.path()).unwrap_err();
    assert!(matches!(err, vectordb_core::Error::AlreadyOpen(_)));
}

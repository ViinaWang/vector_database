#![allow(clippy::unwrap_used)]

//! 过滤检索语义: 过滤在算距离前应用，无召回损失。

use serde_json::json;
use tempfile::TempDir;
use vectordb_core::{CollectionConfig, Condition, Database, Metric, Point, Query};

#[test]
fn filters_narrow_and_order_correctly() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path()).unwrap();
    let c = db
        .create_collection("c", CollectionConfig::new(2, Metric::L2).unwrap())
        .unwrap();

    let mut pts = Vec::new();
    for i in 0..50i64 {
        let lang = if i % 3 == 0 { "en" } else { "zh" };
        pts.push(Point::new(
            i,
            vec![i as f32, (i % 7) as f32],
            Some(json!({"lang": lang, "year": 2000 + i, "tags": ["t1", "t2"]})),
        ));
    }
    c.upsert(&pts).unwrap();

    let q = vec![10.0, 3.0];

    // 精确对照: 手工过滤后算 L2。
    let expected: Vec<(i64, f32)> = {
        let mut all: Vec<(i64, f32)> = pts
            .iter()
            .filter(|p| p.payload.as_ref().unwrap()["lang"] == json!("en"))
            .map(|p| {
                let d = (p.vector[0] - q[0]).powi(2) + (p.vector[1] - q[1]).powi(2);
                (p.id.as_num().unwrap(), d)
            })
            .collect();
        all.sort_by(|a, b| a.1.total_cmp(&b.1));
        all
    };

    let hits = c
        .query(
            &Query::vector(q.clone())
                .top_k(5)
                .filter(Condition::matches("lang", "en")),
        )
        .unwrap();
    assert_eq!(hits.len(), 5.min(expected.len()));
    for (hit, exp) in hits.iter().zip(expected.iter()) {
        assert_eq!(hit.id.as_num().unwrap(), exp.0);
        assert!(
            (hit.score - exp.1).abs() < 1e-3,
            "{} vs {}",
            hit.score,
            exp.1
        );
    }

    // 范围 + 组合
    let hits = c
        .query(
            &Query::vector(q.clone())
                .top_k(50)
                .filter(Condition::all(vec![
                    Condition::range("year", None, Some(2010.0), None, Some(2030.0)),
                    Condition::matches("tags", "t1"),
                    !Condition::matches("lang", "zh"),
                ])),
        )
        .unwrap();
    // year in [2010,2030] ⇒ id in [10,30]，且 lang=en ⇒ id%3==0
    let expect_ids: Vec<i64> = (10..=30).filter(|i| i % 3 == 0).collect();
    let got_ids: Vec<i64> = hits.iter().filter_map(|h| h.id.as_num()).collect();
    assert_eq!(got_ids, expect_ids, "got {got_ids:?} want {expect_ids:?}");
    for h in &hits {
        assert!(
            expect_ids.contains(&h.id.as_num().unwrap()),
            "unexpected {}",
            h.id
        );
    }

    // has_id
    let hits = c
        .query(&Query::vector(q).top_k(3).filter(Condition::HasId {
            ids: vec![5.into(), 41.into()],
        }))
        .unwrap();
    assert_eq!(hits.len(), 2);
    for h in &hits {
        assert!(h.id.as_num() == Some(5) || h.id.as_num() == Some(41));
    }
}

#[test]
fn filtered_query_after_flush_and_delete() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path()).unwrap();
    let c = db
        .create_collection("c", CollectionConfig::new(2, Metric::Cosine).unwrap())
        .unwrap();
    let mut pts = Vec::new();
    for i in 0..10 {
        pts.push(Point::new(
            i,
            vec![1.0, i as f32],
            Some(json!({"g": if i < 5 { "a" } else { "b" }})),
        ));
    }
    c.upsert(&pts).unwrap();
    db.flush().unwrap();
    c.delete(&[0.into(), 5.into()]).unwrap();
    // 再写进可变段，混合两段过滤。
    c.upsert(&[Point::new(100, vec![1.0, 0.5], Some(json!({"g": "a"})))])
        .unwrap();

    let hits = c
        .query(
            &Query::vector(vec![1.0, 0.0])
                .top_k(10)
                .filter(Condition::matches("g", "a")),
        )
        .unwrap();
    let ids: Vec<i64> = hits.iter().filter_map(|h| h.id.as_num()).collect();
    assert!(!ids.contains(&0), "已删除点不应出现");
    assert_eq!(ids.len(), 5); // a 组: 1,2,3,4 + 100
    assert!(ids.contains(&100));
}

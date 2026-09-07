//! HNSW 引擎级召回门槛。正常用例控制在数千点/几十秒内（debug 模式），
//! 标 #[ignore] 的大规模用例用 `cargo test --release -- --ignored` 跑。
//!
//! 对照基准 = IndexKind::Flat 集合（精确扫描），同数据同删除。

#![allow(clippy::unwrap_used)]

use serde_json::json;
use tempfile::TempDir;
use vectordb_core::{CollectionConfig, Condition, Database, IndexKind, Metric, Point, Query};

const DIM: usize = 64;

fn lcg_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..dim)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) & 0xffff) as f32 / 65535.0) * 2.0 - 1.0
        })
        .collect()
}

/// 建好 h（hnsw）与 f（flat，对照）两个集合的库。
fn setup(dir: &TempDir, dim: usize) -> Database {
    let db = Database::open(dir.path()).unwrap();
    db.create_collection("h", CollectionConfig::new(dim, Metric::Cosine).unwrap())
        .unwrap();
    db.create_collection(
        "f",
        CollectionConfig::new(dim, Metric::Cosine)
            .unwrap()
            .with_index(IndexKind::Flat),
    )
    .unwrap();
    db
}

fn points(n: usize, dim: usize, group_every: usize) -> Vec<Point> {
    (0..n)
        .map(|i| {
            let payload = if group_every > 0 && i % group_every == 0 {
                json!({"g": "a"})
            } else {
                json!({"g": "b"})
            };
            Point::new(i as i64, lcg_vec(dim, i as u64 + 1), Some(payload))
        })
        .collect()
}

fn ids_of(hits: &[vectordb_core::ScoredPoint]) -> Vec<(i64, f32)> {
    hits.iter()
        .filter_map(|h| h.id.as_num().map(|n| (n, h.score)))
        .collect()
}

fn recall_of(got: &[(i64, f32)], truth: &[(i64, f32)], k: usize) -> f32 {
    let g: std::collections::HashSet<i64> = got.iter().map(|(id, _)| *id).take(k).collect();
    let t: std::collections::HashSet<i64> = truth.iter().map(|(id, _)| *id).take(k).collect();
    g.intersection(&t).count() as f32 / t.len().max(1) as f32
}

fn avg_recall(db: &Database, queries: &[Vec<f32>], k: usize, filter: Option<&Condition>) -> f32 {
    let h = db.collection("h").unwrap();
    let f = db.collection("f").unwrap();
    let mut total = 0.0f32;
    for q in queries {
        let mut qh = Query::vector(q.clone()).top_k(k);
        let mut qf = Query::vector(q.clone()).top_k(k);
        if let Some(c) = filter {
            qh = qh.filter(c.clone());
            qf = qf.filter(c.clone());
        }
        total += recall_of(
            &ids_of(&h.query(&qh).unwrap()),
            &ids_of(&f.query(&qf).unwrap()),
            k,
        );
    }
    total / queries.len() as f32
}

fn queries(n: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n).map(|i| lcg_vec(dim, 100_000 + i as u64)).collect()
}

#[test]
fn recall_threshold_random_data() {
    let dir = TempDir::new().unwrap();
    {
        let db = setup(&dir, DIM);
        let pts = points(3000, DIM, 0);
        db.collection("h").unwrap().upsert(&pts).unwrap();
        db.collection("f").unwrap().upsert(&pts).unwrap();
        db.flush().unwrap();
        let r = avg_recall(&db, &queries(30, DIM), 10, None);
        assert!(r >= 0.97, "recall@10 = {r:.3}");
    }
    // 重开后走 index.bin 载入的索引，召回不回退。
    let db = Database::open(dir.path()).unwrap();
    let r = avg_recall(&db, &queries(10, DIM), 10, None);
    assert!(r >= 0.97, "recall@10 after reopen = {r:.3}");
}

#[test]
fn incremental_batches_match_single_batch() {
    let dir = TempDir::new().unwrap();
    let db = setup(&dir, DIM);
    let pts = points(2400, DIM, 0);
    // 分 4 批，批间 flush → 4 个不可变段各建各的索引，跨段合并检索。
    for chunk in pts.chunks(600) {
        db.collection("h").unwrap().upsert(chunk).unwrap();
        db.collection("f").unwrap().upsert(chunk).unwrap();
        db.flush().unwrap();
    }
    let r = avg_recall(&db, &queries(20, DIM), 10, None);
    assert!(r >= 0.97, "multi-segment recall@10 = {r:.3}");
}

#[test]
fn delete_heavy_triggers_rebuild_and_keeps_recall() {
    let dir = TempDir::new().unwrap();
    {
        let db = setup(&dir, DIM);
        let pts = points(2000, DIM, 0);
        db.collection("h").unwrap().upsert(&pts).unwrap();
        db.collection("f").unwrap().upsert(&pts).unwrap();
        db.flush().unwrap();

        // 删 20%（400/2000，超过 0.30 需再删些——用 2/5=40% 才越阈值）。
        let victims: Vec<vectordb_core::ExternalId> = (0..2000)
            .step_by(5)
            .map(vectordb_core::ExternalId::Num)
            .collect();
        assert_eq!(victims.len(), 400); // 20% —— 低于阈值，不触发重建
        db.collection("h").unwrap().delete(&victims).unwrap();
        db.collection("f").unwrap().delete(&victims).unwrap();
        assert_eq!(db.collection("h").unwrap().count(), 1600);

        let r = avg_recall(&db, &queries(20, DIM), 10, None);
        assert!(r >= 0.95, "recall@10 after 20% delete = {r:.3}");
    }
    let db = Database::open(dir.path()).unwrap();
    assert_eq!(db.collection("h").unwrap().count(), 1600);
    let r = avg_recall(&db, &queries(10, DIM), 10, None);
    assert!(r >= 0.95, "recall@10 after reopen = {r:.3}");
}

#[test]
fn delete_over_threshold_rebuilds_segment() {
    // 墓碑越过 30% 阈值触发单段就地重建: 数据保持、段数量不膨胀。
    let dir = TempDir::new().unwrap();
    let db = setup(&dir, DIM);
    let pts = points(1000, DIM, 0);
    db.collection("h").unwrap().upsert(&pts).unwrap();
    db.flush().unwrap();

    let victims: Vec<vectordb_core::ExternalId> = (0..1000)
        .step_by(2)
        .map(vectordb_core::ExternalId::Num)
        .collect();
    assert_eq!(victims.len(), 500); // 50% 墓碑 → 重建
    db.collection("h").unwrap().delete(&victims).unwrap();
    assert_eq!(db.collection("h").unwrap().count(), 500);

    // 重建 = 新段目录替换旧的; 段目录里应恰好只有一个 seg-*。
    let segs: Vec<_> = std::fs::read_dir(dir.path().join("segments"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("seg-"))
        })
        .collect();
    assert_eq!(segs.len(), 1, "segments: {segs:?}");

    // 对照集合同批删除后召回。
    db.collection("f").unwrap().upsert(&pts).unwrap();
    db.flush().unwrap();
    db.collection("f").unwrap().delete(&victims).unwrap();
    let r = avg_recall(&db, &queries(20, DIM), 10, None);
    assert!(r >= 0.95, "recall@10 after 50% delete+rebuild = {r:.3}");
}

#[test]
fn filtered_recall_by_selectivity() {
    // g=="a" 占 1/10; g=="b" 占 9/10。
    let dir = TempDir::new().unwrap();
    let db = setup(&dir, DIM);
    let pts = points(3000, DIM, 10);
    db.collection("h").unwrap().upsert(&pts).unwrap();
    db.collection("f").unwrap().upsert(&pts).unwrap();
    db.flush().unwrap();

    let r10 = avg_recall(
        &db,
        &queries(20, DIM),
        10,
        Some(&Condition::matches("g", "b")),
    );
    assert!(r10 >= 0.95, "10% selectivity recall@10 = {r10:.3}");

    let r1 = avg_recall(
        &db,
        &queries(20, DIM),
        10,
        Some(&Condition::matches("g", "a")),
    );
    assert!(r1 >= 0.85, "1% selectivity recall@10 = {r1:.3}");
}

// 大规模用例: release 模式手动跑。cargo test --release --test hnsw -- --ignored
// 门槛按维度分设（默认 ef_search=512 的实测水平留有余量，
// 完整代价-召回矩阵见 index/hnsw/params.rs 文档）:
//   128 维（Cosine）≥ 0.98; 768 维（Cosine，均匀随机最难场景）≥ 0.92
#[test]
#[ignore = "large: run in release with --ignored"]
fn recall_10k_128_and_10k_768() {
    for (n, dim, threshold) in [(10_000usize, 128usize, 0.98f32), (10_000, 768, 0.92)] {
        let dir = TempDir::new().unwrap();
        let db = setup(&dir, dim);
        for s in (0..n).step_by(2000) {
            let chunk: Vec<Point> = (s..s + 2000)
                .map(|i| Point::new(i as i64, lcg_vec(dim, i as u64 + 7), None))
                .collect();
            db.collection("h").unwrap().upsert(&chunk).unwrap();
            db.collection("f").unwrap().upsert(&chunk).unwrap();
        }
        db.flush().unwrap();

        let qs = queries(30, dim);
        let mut total = 0.0f32;
        for q in &qs {
            let got = ids_of(
                &db.collection("h")
                    .unwrap()
                    .query(&Query::vector(q.clone()).top_k(10))
                    .unwrap(),
            );
            let want = ids_of(
                &db.collection("f")
                    .unwrap()
                    .query(&Query::vector(q.clone()).top_k(10))
                    .unwrap(),
            );
            total += recall_of(&got, &want, 10);
        }
        let avg = total / qs.len() as f32;
        assert!(avg >= threshold, "{n}x{dim} recall@10 = {avg:.3}");
    }
}

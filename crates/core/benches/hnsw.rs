//! HNSW 构建/查询基准，与 flat 精确扫描对照。
#![allow(clippy::unwrap_used)] // 基准代码允许 unwrap
//! perf/ 分支调参必须附本组前后数据（CONTRIBUTING）。

#![allow(missing_docs)] // criterion 宏生成的入口不需要文档

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::sync::Arc;
use vectordb_core::index::hnsw::{HnswIndex, HnswParams};
use vectordb_core::index::{AnnIndex, FlatIndex};
use vectordb_core::kernel::{self, Metric};
use vectordb_core::segment::core::SegmentCore;

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

fn core_of(n: usize, dim: usize) -> Arc<SegmentCore> {
    let mut vectors = Vec::with_capacity(n * dim);
    let mut norms = Vec::with_capacity(n);
    for i in 0..n {
        let v = lcg_vec(dim, i as u64 + 3);
        norms.push(kernel::norm(&v));
        vectors.extend_from_slice(&v);
    }
    Arc::new(SegmentCore {
        dim,
        vectors,
        norms,
        ids: (0..n as i64).map(vectordb_core::ExternalId::Num).collect(),
        payloads: vec![None; n],
    })
}

fn bench_build(c: &mut Criterion) {
    let mut g = c.benchmark_group("hnsw_build");
    g.sample_size(10);
    g.measurement_time(std::time::Duration::from_secs(20));
    for (n, dim) in [(5_000usize, 128usize), (10_000, 768)] {
        let core = core_of(n, dim);
        g.bench_with_input(
            BenchmarkId::new("build", format!("{n}x{dim}")),
            &core,
            |bench, core| {
                bench.iter(|| {
                    HnswIndex::build(
                        black_box(core.clone()),
                        Metric::Cosine,
                        HnswParams::default(),
                    )
                    .unwrap()
                })
            },
        );
    }
    g.finish();
}

fn bench_query(c: &mut Criterion) {
    // 高维（导航最难的均匀随机场景）与低维各一组，hnsw vs flat。
    for (n, dim) in [(10_000usize, 768usize), (10_000, 128)] {
        let core = core_of(n, dim);
        let hnsw = HnswIndex::build(core.clone(), Metric::Cosine, HnswParams::default()).unwrap();
        let flat = FlatIndex::new(core.clone(), Metric::Cosine);
        let queries: Vec<Vec<f32>> = (0..16).map(|i| lcg_vec(dim, 500_000 + i)).collect();

        let mut g = c.benchmark_group(format!("query/n{n}d{dim}/top10"));
        g.sample_size(50);
        g.bench_function("hnsw", |bench| {
            bench.iter(|| {
                let mut acc = 0f32;
                for q in &queries {
                    if let Some((_, s)) = hnsw.search(black_box(q), 10, None).first() {
                        acc += s;
                    }
                }
                acc
            })
        });
        g.bench_function("flat", |bench| {
            bench.iter(|| {
                let mut acc = 0f32;
                for q in &queries {
                    if let Some((_, s)) = flat.search(black_box(q), 10, None).first() {
                        acc += s;
                    }
                }
                acc
            })
        });
        if dim == 768 {
            // 高选择性过滤（1/16 命中）下的 hnsw 查询。
            let pass = |i: u32| i % 16 == 0;
            g.bench_function("hnsw_filtered_6pct", |bench| {
                bench.iter(|| {
                    let mut acc = 0f32;
                    for q in &queries {
                        if let Some((_, s)) = hnsw.search(black_box(q), 10, Some(&pass)).first() {
                            acc += s;
                        }
                    }
                    acc
                })
            });
        }
        g.finish();
    }
}

criterion_group!(benches, bench_build, bench_query);
criterion_main!(benches);

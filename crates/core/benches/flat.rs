#![allow(missing_docs)] // criterion 宏生成的入口不需要文档
//! 精确扫描基准: 不同规模 × 维度的单查询延迟。

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use vectordb_core::index::flat;
use vectordb_core::kernel::{self, Metric};

fn gen_vec(dim: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..dim)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) & 0xffff) as f32 / 65535.0 * 2.0 - 1.0
        })
        .collect()
}

fn bench_flat(c: &mut Criterion) {
    for (n, dim) in [(10_000usize, 768usize), (100_000, 128), (50_000, 1536)] {
        let mut vectors = Vec::with_capacity(n * dim);
        for i in 0..n {
            vectors.extend_from_slice(&gen_vec(dim, i as u32));
        }
        let norms: Vec<f32> = vectors.chunks(dim).map(kernel::norm).collect();
        let q = gen_vec(dim, 999_999);
        let mut g = c.benchmark_group(format!("flat/n{n}d{dim}"));
        g.sample_size(30);
        for (metric, name) in [(Metric::L2, "l2"), (Metric::Cosine, "cosine")] {
            g.bench_with_input(
                BenchmarkId::new(name, format!("{n}x{dim}")),
                &metric,
                |bench, &m| {
                    bench.iter(|| {
                        flat::search(
                            m,
                            black_box(&vectors),
                            black_box(&norms),
                            dim,
                            black_box(&q),
                            None,
                            10,
                        )
                    })
                },
            );
        }
        g.finish();
    }
}

criterion_group!(benches, bench_flat);
criterion_main!(benches);

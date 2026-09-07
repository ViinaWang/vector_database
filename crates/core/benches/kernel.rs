#![allow(missing_docs)] // criterion 宏生成的入口不需要文档
//! 距离内核基准。热路径守护（CONTRIBUTING: perf/ PR 必须附此组数据）。

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use vectordb_core::kernel::{self, Metric};

fn gen_vec(dim: usize, seed: u32) -> Vec<f32> {
    // 线性同余，够稳定即可。
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..dim)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) & 0xffff) as f32 / 65535.0 * 2.0 - 1.0
        })
        .collect()
}

fn bench_kernel(c: &mut Criterion) {
    for dim in [128usize, 768, 1536] {
        let a = gen_vec(dim, 1);
        let b = gen_vec(dim, 2);
        let bn = kernel::norm(&b);
        let mut g = c.benchmark_group(format!("kernel/dim{dim}"));
        g.throughput(criterion::Throughput::Bytes((dim * 4 * 2) as u64));
        g.bench_with_input(BenchmarkId::new("dot", dim), &dim, |bench, _| {
            bench.iter(|| kernel::dot(black_box(&a), black_box(&b)))
        });
        g.bench_with_input(BenchmarkId::new("l2_sq", dim), &dim, |bench, _| {
            bench.iter(|| kernel::l2_sq(black_box(&a), black_box(&b)))
        });
        g.bench_with_input(BenchmarkId::new("cosine", dim), &dim, |bench, _| {
            bench.iter(|| kernel::score(Metric::Cosine, black_box(&a), black_box(&b), bn))
        });
        g.finish();
    }
}

criterion_group!(benches, bench_kernel);
criterion_main!(benches);

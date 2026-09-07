# ADR 0003: 距离内核边界与 unsafe 政策

日期: 2026-09-07
状态: 已接受

## 决策

kernel 模块是唯一的距离计算入口（L2 / cosine / dot，向量分值语义: L2 返回平方距离、越小越好，
其余越大越好）。当前实现为标量 + 手工分块循环，依赖编译器自动向量化。

workspace 目前 `unsafe_code = "forbid"`。将来引入 AVX2/NEON/SIMD128 内联汇编或
std::arch 内在函数时，仅允许在 kernel / quant / index / wal / storage 五个模块内
将 forbid 降级为 allow，并要求:

- 每个 unsafe 块附 `// SAFETY:` 注释说明不变量;
- `#![deny(unsafe_op_in_unsafe_fn)]`;
- CI 保留对这些模块的 miri 抽查。

## 理由

内核先以 trait 边界 + 基准（benches/kernel.rs）固定，SIMD 优化作为独立 perf/ 分支迭代，
替换实现不动调用方；召回率/正确性测试以标量实现为参照。

# ADR 0001: 所有 IO 走 StorageBackend / Fs 抽象

日期: 2026-09-07
状态: 已接受

## 背景

目标平台包含 wasm32（浏览器/Node），该环境没有 mmap、没有 std::fs 的完整语义。
主流向量库的性能设计普遍绑定 mmap + page cache，导致其 Web 端要么缺失、要么是另一套代码。
如果核心存储层直接调用 std::fs / memmap2，浏览器支持会在中期被迫重写。

## 决策

两层抽象，核心代码只依赖 trait：

- `StorageBackend`: 单文件的定位读写（read_at / write_at / truncate / sync / len）。
- `Fs`: 目录级操作（create / rename / remove / list），原子写由 "写 .tmp + rename" 在其上实现。

实现:

- native: std::fs + unix `read_at`/`write_at` 与 windows `seek_read`/`seek_write`。mmap 后端将来作为只读优化加入，不改变接口。
- memory: 全内存实现（HashMap of buffers），供测试、临时库与 wasm 前期使用。

## 后果

- 原子 rename 语义在 memory 实现里靠移动 map 条目模拟，Windows 上目录 rename 的限制由"segment id 单调递增、目标必不存在"规避。
- 代价是 native 路径多一层间接调用；热路径（WAL 追加、段扫描）在 benches/ 下单独测量，超过 5% 回归再考虑特化。

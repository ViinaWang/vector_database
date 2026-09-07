# ADR 0002: WAL 先行写入 + 不可变段 + 可重写边车

日期: 2026-09-07
状态: 已接受

## 决策

写路径: 每个变更先追加 WAL（CRC32 校验、fsync）再应用内存。崩溃恢复 = 载入 manifest，重放 lsn > last_flushed_lsn 的 WAL 记录，截断损坏尾部。

段模型: flush 将内存可变段写成不可变段目录（vectors.bin / norms.bin / ids.jsonl / payloads.jsonl），manifest 记录段列表与 last_flushed_lsn，整个 manifest 原子替换。flush 完成后截断 WAL。

不可变段上的删除与 payload 更新不重写大文件，写两个小的可重写边车:
`dels.bin`（roaring bitmap）、`payloads.overlay.jsonl`。边车同样 tmp+rename 原子替换。
压实把所有段 + 边车合并为一个新段后原子换 manifest。

## 后果

- 崩溃窗口分析: 段目录 rename 之后、manifest 更新之前崩溃 → 段目录成为孤儿，数据仍在 WAL，恢复后重放到内存，孤儿目录在打开时 GC；manifest 更新之后、WAL 截断之前崩溃 → 重放按 lsn 跳过已落盘记录。
- WAL 记录当前用 JSON 编码，可读性优先；二进制编码是已知的性能候选，接口不变。
- upsert 已存在于不可变段的点 = 该段打墓碑 + 写入可变段，全局"一个 id 至多一个活点"由写入单路径维持。

# vectordb

嵌入式向量数据库，Rust 编写。目标是"SQLite 式"的使用体验: 一个目录就是一个库，
进程内直接调用，也可以起一个 HTTP 服务或加载 WebAssembly 模块。

状态: **pre-alpha**。API 和磁盘格式都会破坏性变更，别拿重要数据喂它。

## 现在有什么

- 集合（固定维度 + 距离度量 L2/Cosine/Dot）+ JSON payload
- CRUD: upsert / get / delete / scroll 分页 / payload 覆盖与清空
- 检索: HNSW 近似索引（不可变段，墓碑过滤在图遍历内生效）+ 精确扫描（可变段与对照）
- 过滤条件: 字段相等/数组包含、数值范围、存在性、AND/OR/NOT、ID 集合
- 持久化: WAL 先行写、不可变段 + manifest 原子切换、崩溃后自动恢复（尾部损坏自动截断）
- compact: 合并段、清墓碑、固化 payload 覆盖层; 墓碑超 30% 的段自动单段重建
- 三种接入: Rust 库 / REST 服务 / wasm（内存模式）

检索质量与代价（10k 均匀随机数据，recall@10，默认参数）:
128 维 ≈ 0.99，768 维 ≈ 0.93–0.96; ef_search 可调，完整矩阵见
`crates/core/src/index/hnsw/params.rs`。真实嵌入数据（有簇结构）显著更好。

## 现在没有什么（不做的都写在计划里）

量化、混合检索/全文、字段倒排索引、分布式、多租户、RBAC、GPU。
这是刻意的——先把嵌入式 CRUD、持久化与 ANN 语义做扎实。

## Quick start

Rust:

```rust
use vectordb_core::{CollectionConfig, Database, Metric, Point, Query};

let db = Database::open("./data")?;
let coll = db.create_collection("docs", CollectionConfig::new(384, Metric::Cosine)?)?;
coll.upsert(&[Point::new(1, vec![0.1; 384], Some(serde_json::json!({"lang": "en"})))])?;

for hit in coll.query(&Query::vector(vec![0.1; 384]).top_k(5))? {
    println!("{} score={}", hit.id, hit.score);
}
```

分数语义: L2 返回平方距离（越小越好），Cosine/Dot 返回相似度（越大越好）。

HTTP:

```sh
vectordb-server                          # VDB_PATH=./data VDB_ADDR=127.0.0.1:7280
curl -X POST :7280/collections -H 'content-type: application/json' \
     -d '{"name":"docs","dim":384,"metric":"cosine"}'
curl -X PUT :7280/collections/docs/points -H 'content-type: application/json' \
     -d '[{"id":1,"vector":[...],"payload":{"lang":"en"}}]'
curl -X POST :7280/collections/docs/query -H 'content-type: application/json' \
     -d '{"vector":[...],"top_k":5,"filter":{"field":"lang","op":{"kind":"match","value":"en"}}}'
```

CLI:

```sh
vectordb-cli --path ./data create docs 384 --metric cosine
vectordb-cli --path ./data upsert docs points.jsonl   # 每行一个 {"id":..,"vector":[..],"payload":{}}
vectordb-cli --path ./data query docs --vector 0.1,0.2,... --top-k 5
```

## 平台

| 形态 | 状态 |
|---|---|
| Linux / macOS / Windows（x86_64, aarch64） | 支持，CI 矩阵覆盖 |
| wasm32（Node / 浏览器） | 内存模式可用（`Database::open_memory` / `vectordb-wasm`）; 距离内核走 SIMD128，产物要求 Chrome 91+ / Firefox 89+ / Safari 16.4+ / Node 16+ |
| 浏览器持久化（OPFS + Worker） | 计划中 |

wasm 当前为纯内存库: 数据生命周期 = 实例生命周期，持久化用 scroll 导出 / upsert 导回。

## 磁盘布局

```
data/
  manifest.json            # 权威状态: 集合列表、段列表、last_flushed_lsn
  wal.log                  # 未落段的变更（CRC 校验，尾部损坏自动截断）
  segments/seg-N/          # 不可变段: vectors.bin / norms.bin / ids.jsonl / payloads.jsonl
    dels.bin               # 删除墓碑（roaring bitmap，可重写边车）
    payloads.overlay.jsonl # payload 覆盖层（可重写边车）
    index.bin              # HNSW 图（按需）
```

设计决策见 `docs/adr/`。

## 开发

```sh
cargo test --workspace                        # 全部测试（含故障注入、崩溃恢复）
cargo clippy --workspace --all-targets -- -D warnings
cargo bench -p vectordb-core                  # kernel / flat 热路径基准
cargo build -p vectordb-wasm --target wasm32-unknown-unknown
```

分支、提交与 PR 规则见 [CONTRIBUTING.md](CONTRIBUTING.md)。性能热路径是
kernel / quant / index / wal / storage 五个模块，改动必须附基准数据。

## License

MIT OR Apache-2.0 双许可，见 [LICENSE-MIT](LICENSE-MIT) 与 [LICENSE-APACHE](LICENSE-APACHE)。

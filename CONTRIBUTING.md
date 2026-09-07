# Contributing

## Branches & commits

- 长期分支只有 `main`（保护: PR 合入、CI 全绿、禁止 force-push）。
- 工作分支命名: `feat/<module>-<what>` / `fix/<issue>-<what>` / `perf/<hotpath>-<what>`。
- 提交信息走 Conventional Commits: `feat(wal): ...`、`perf(kernel): ...`。scope 用模块名
  （kernel / quant / index / wal / storage / segment / filter / engine / server / wasm / cli / ci）。

## Pull requests

- 单 PR 单一意图，一般不超过 ~400 行有效改动，超出就拆。
- PR 描述三件事: 动机、方案、性能影响（无影响就写"无"）。
- `perf/` PR 必须附 criterion 前后数据; 性能热路径 = kernel / quant / index / wal / storage，
  这些模块的改动在 PR 模板里勾选声明。
- squash merge，保持 main 线性。

## Code style

- `cargo fmt` 默认配置，不争论风格。
- `cargo clippy --workspace --all-targets -- -D warnings` 必须干净。
- 全仓库 `unsafe_code = "forbid"`；需要在热路径开洞时先读 docs/adr/0003。
- 公共 API 写 rustdoc，说语义（复杂度/并发/错误条件），别复述代码。
- 设计决策进 docs/adr/，一篇一个决定。

## Tests

- 新功能带测试; 持久化路径的改动补故障注入用例（tests/fault_injection.rs 有现成 FaultFs）。
- 召回率/性能门槛测试失败即 block，不允许"先合后修"。

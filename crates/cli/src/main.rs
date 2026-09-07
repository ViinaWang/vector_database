//! vectordb-cli: 直接操作数据库目录的命令行。
//!
//! upsert/query 接受 JSONL（每行一个 Point: {"id":..,"vector":[..],"payload":{..}}）。

use clap::{Parser, Subcommand};
use std::io::{BufRead, Write};
use vdb::{CollectionConfig, Database, ExternalId, Metric, Point, Query};
use vectordb_core as vdb;

#[derive(Parser)]
#[command(name = "vectordb", version, about = "embedded vector database cli")]
struct Cli {
    /// 数据库目录
    #[arg(long, default_value = "./data")]
    path: String,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// 列出集合与点数
    Info,
    /// 建集合
    Create {
        name: String,
        dim: usize,
        #[arg(long, default_value = "cosine")]
        metric: String,
    },
    /// 删集合
    Drop { name: String },
    /// 批量写入 JSONL（文件路径，省略或 - 表示 stdin）
    Upsert { name: String, file: Option<String> },
    /// 按 ID 批量取点
    Get { name: String, ids: Vec<String> },
    /// 按 ID 删除
    Delete { name: String, ids: Vec<String> },
    /// 邻近查询
    Query {
        name: String,
        /// 逗号分隔的查询向量，如 "0.1,0.2,0.3"
        #[arg(long, value_delimiter = ',')]
        vector: Option<Vec<f32>>,
        /// 用已有点的 ID 作查询
        #[arg(long)]
        from_id: Option<String>,
        #[arg(long, default_value = "10")]
        top_k: usize,
    },
    /// 分页浏览
    Scroll {
        name: String,
        #[arg(long, default_value = "0")]
        offset: u64,
        #[arg(long, default_value = "20")]
        limit: usize,
    },
    /// 合并压实
    Compact { name: String },
    /// WAL 落段
    Flush,
}

fn parse_metric(s: &str) -> Result<Metric, String> {
    match s.to_ascii_lowercase().as_str() {
        "l2" => Ok(Metric::L2),
        "cosine" => Ok(Metric::Cosine),
        "dot" => Ok(Metric::Dot),
        other => Err(format!("unknown metric {other:?}")),
    }
}

// "123" → Num，"abc" → Str，"\"abc\"" → Str(abc)
fn parse_id(s: &str) -> ExternalId {
    if let Ok(n) = s.parse::<i64>() {
        ExternalId::Num(n)
    } else if let Some(stripped) = s.strip_prefix('"').and_then(|x| x.strip_suffix('"')) {
        ExternalId::Str(stripped.into())
    } else {
        ExternalId::Str(s.into())
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let db = Database::open(&cli.path)?;

    match cli.cmd {
        Cmd::Info => {
            for name in db.collections() {
                let c = db.collection(&name).ok_or("collection vanished")?;
                println!(
                    "{}  dim={} metric={:?} points={}",
                    c.name(),
                    c.config().dim,
                    c.config().metric,
                    c.count()
                );
            }
        }
        Cmd::Create { name, dim, metric } => {
            let metric = parse_metric(&metric)?;
            db.create_collection(&name, CollectionConfig::new(dim, metric)?)?;
            println!("created {name}");
        }
        Cmd::Drop { name } => {
            db.drop_collection(&name)?;
            println!("dropped {name}");
        }
        Cmd::Upsert { name, file } => {
            let c = db
                .collection(&name)
                .ok_or_else(|| vdb::Error::CollectionNotFound(name.clone()))?;
            let reader: Box<dyn BufRead> = match file.as_deref() {
                None | Some("-") => Box::new(std::io::stdin().lock()),
                Some(p) => Box::new(std::io::BufReader::new(std::fs::File::open(p)?)),
            };
            let mut batch = Vec::new();
            let mut total = 0usize;
            for (i, line) in reader.lines().enumerate() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let p: Point =
                    serde_json::from_str(&line).map_err(|e| format!("line {}: {e}", i + 1))?;
                batch.push(p);
                if batch.len() >= 1000 {
                    let n = batch.len();
                    c.upsert(&batch)?;
                    total += n;
                    batch.clear();
                }
            }
            if !batch.is_empty() {
                total += batch.len();
                c.upsert(&batch)?;
            }
            println!("upserted {total}");
        }
        Cmd::Get { name, ids } => {
            let c = db
                .collection(&name)
                .ok_or_else(|| vdb::Error::CollectionNotFound(name.clone()))?;
            let ids: Vec<ExternalId> = ids.iter().map(|s| parse_id(s)).collect();
            for p in c.get(&ids)?.into_iter().flatten() {
                println!("{}", serde_json::to_string(&p)?);
            }
        }
        Cmd::Delete { name, ids } => {
            let c = db
                .collection(&name)
                .ok_or_else(|| vdb::Error::CollectionNotFound(name.clone()))?;
            let ids: Vec<ExternalId> = ids.iter().map(|s| parse_id(s)).collect();
            println!("deleted {}", c.delete(&ids)?);
        }
        Cmd::Query {
            name,
            vector,
            from_id,
            top_k,
        } => {
            let c = db
                .collection(&name)
                .ok_or_else(|| vdb::Error::CollectionNotFound(name.clone()))?;
            let q = match (vector, from_id) {
                (Some(v), None) => Query::vector(v).top_k(top_k),
                (None, Some(id)) => Query::nearest_to(parse_id(&id)).top_k(top_k),
                _ => return Err("need exactly one of --vector or --from-id".into()),
            };
            for h in c.query(&q)? {
                println!("{}", serde_json::to_string(&h)?);
            }
        }
        Cmd::Scroll {
            name,
            offset,
            limit,
        } => {
            let c = db
                .collection(&name)
                .ok_or_else(|| vdb::Error::CollectionNotFound(name.clone()))?;
            let page = c.scroll(offset, limit, None)?;
            for p in page.points {
                println!("{}", serde_json::to_string(&p)?);
            }
            if let Some(next) = page.next_offset {
                let _ = std::io::stdout().flush();
                eprintln!("next: --offset {next}");
            }
        }
        Cmd::Compact { name } => {
            let c = db
                .collection(&name)
                .ok_or_else(|| vdb::Error::CollectionNotFound(name.clone()))?;
            c.compact()?;
            println!("compacted {name}");
        }
        Cmd::Flush => {
            db.flush()?;
            println!("flushed");
        }
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

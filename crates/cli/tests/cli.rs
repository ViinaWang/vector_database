#![allow(clippy::unwrap_used)]

//! CLI 集成测试: 真子进程，stdin 传 JSONL，stdout 逐行解析。

use std::io::Write;
use std::process::{Command, Output, Stdio};

use serde_json::{Value, json};
use tempfile::TempDir;

const EXE: &str = env!("CARGO_BIN_EXE_vectordb-cli");

fn run(db: &str, args: &[&str], stdin: Option<&str>) -> Output {
    let mut child = Command::new(EXE)
        .arg("--path")
        .arg(db)
        .args(args)
        .stdin(if stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    if let Some(input) = stdin {
        let mut s = child.stdin.take().unwrap();
        s.write_all(input.as_bytes()).unwrap();
        drop(s);
    }
    child.wait_with_output().unwrap()
}

fn stdout_string(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn json_lines(out: &Output) -> Vec<Value> {
    stdout_string(out)
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

// ---------------------------------------------------------------- 测试

#[test]
fn end_to_end_flow() {
    let dir = TempDir::new().unwrap();
    let db = dir.path().join("db");
    let db = db.to_str().unwrap();

    let out = run(db, &["create", "demo", "3", "--index", "flat"], None);
    assert!(out.status.success());
    assert_eq!(stdout_string(&out), "created demo\n");

    let jsonl = concat!(
        r#"{"id":1,"vector":[1.0,0.0,0.0],"payload":{"lang":"en"}}"#,
        "\n",
        r#"{"id":2,"vector":[0.0,1.0,0.0],"payload":{"lang":"zh"}}"#,
        "\n",
        r#"{"id":"s3","vector":[0.9,0.1,0.0],"payload":{"lang":"en"}}"#,
        "\n",
    );
    let out = run(db, &["upsert", "demo"], Some(jsonl));
    assert!(out.status.success());
    assert_eq!(stdout_string(&out), "upserted 3\n");

    let out = run(
        db,
        &["query", "demo", "--vector", "1.0,0.0,0.0", "--top-k", "2"],
        None,
    );
    assert!(out.status.success());
    let hits = json_lines(&out);
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0]["id"], json!(1));
    assert_eq!(hits[0]["payload"], json!({"lang": "en"}));
    assert_eq!(hits[0]["vector"], json!(null)); // with_vector 缺省 false
    assert!(hits[0]["score"].as_f64().unwrap() >= hits[1]["score"].as_f64().unwrap());

    let out = run(db, &["info"], None);
    assert!(out.status.success());
    assert_eq!(
        stdout_string(&out),
        "demo  dim=3 metric=Cosine index=flat points=3\n"
    );

    let out = run(db, &["delete", "demo", "1"], None);
    assert!(out.status.success());
    assert_eq!(stdout_string(&out), "deleted 1\n");

    let out = run(db, &["compact", "demo"], None);
    assert!(out.status.success());
    assert_eq!(stdout_string(&out), "compacted demo\n");

    // 重开（新进程，同一 --path）后数据仍在
    let out = run(db, &["info"], None);
    assert!(out.status.success());
    assert_eq!(
        stdout_string(&out),
        "demo  dim=3 metric=Cosine index=flat points=2\n"
    );

    let out = run(
        db,
        &["query", "demo", "--vector", "1.0,0.0,0.0", "--top-k", "10"],
        None,
    );
    assert!(out.status.success());
    let ids: Vec<Value> = json_lines(&out)
        .into_iter()
        .map(|h| h["id"].clone())
        .collect();
    assert_eq!(ids, vec![json!("s3"), json!(2)]);
}

#[test]
fn missing_collection_exits_nonzero() {
    let dir = TempDir::new().unwrap();
    let db = dir.path().join("db");
    let db = db.to_str().unwrap();

    let out = run(db, &["get", "nope", "1"], None);
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("error:"));
}

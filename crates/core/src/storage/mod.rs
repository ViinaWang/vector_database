//! IO 抽象层（ADR 0001）。内核不直接触碰 std::fs / mmap。
//!
//! - [`StorageBackend`] — 单文件定位读写
//! - [`Fs`] — 目录操作与文件创建，原子写由 `atomic_write` 统一实现
//!
//! 实现: [`file`]（native，cfg 非 wasm）、[`memory`]（全内存，测试/wasm 前期）。

#[cfg(not(target_arch = "wasm32"))]
pub mod file;
pub mod memory;

use std::io;
use std::path::{Path, PathBuf};

use crate::error::Result;

/// 单文件的定位读写视图。
pub trait StorageBackend: Send + Sync {
    /// 从 offset 起读满 buf；不足则返回 UnexpectedEof。
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()>;
    /// 从 offset 起覆写 buf（文件按需扩展）。
    fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<()>;
    /// 当前文件长度。
    fn len(&self) -> io::Result<u64>;
    /// 是否为空文件。
    fn is_empty(&self) -> io::Result<bool> {
        Ok(self.len()? == 0)
    }
    /// 截断到 new_len。
    fn truncate(&self, new_len: u64) -> io::Result<()>;
    /// 刷盘（native = fsync）。
    fn sync(&self) -> io::Result<()>;
    /// 追加写，返回起始 offset。
    fn append(&self, buf: &[u8]) -> io::Result<u64> {
        let off = self.len()?;
        self.write_at(buf, off)?;
        Ok(off)
    }
}

/// 目录级文件系统视图。
pub trait Fs: Send + Sync {
    /// 创建（或截断）文件并返回读写视图。
    fn create(&self, path: &Path) -> io::Result<Box<dyn StorageBackend>>;
    /// 打开已存在文件（不截断）。
    fn open_rw(&self, path: &Path) -> io::Result<Box<dyn StorageBackend>>;
    /// 重命名/移动。目标必须不存在。
    fn rename(&self, from: &Path, to: &Path) -> io::Result<()>;
    /// 删除文件。
    fn remove_file(&self, path: &Path) -> io::Result<()>;
    /// 递归删除目录。
    fn remove_dir_all(&self, path: &Path) -> io::Result<()>;
    /// 递归创建目录。
    fn create_dir_all(&self, path: &Path) -> io::Result<()>;
    /// 文件或目录是否存在。
    fn exists(&self, path: &Path) -> bool;
    /// 列出直接子项。
    fn list_dir(&self, path: &Path) -> io::Result<Vec<PathBuf>>;
}

/// 原子写: 写 `<path>.tmp` + sync + rename。崩溃后要么旧内容要么完整新内容。
pub fn atomic_write(fs: &dyn Fs, path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    {
        let f = fs.create(&tmp)?;
        f.write_at(bytes, 0)?;
        f.truncate(bytes.len() as u64)?;
        f.sync()?;
    }
    fs.rename(&tmp, path)?;
    Ok(())
}

/// 读整个文件。
pub fn read_all(b: &dyn StorageBackend) -> io::Result<Vec<u8>> {
    let len = b.len()?;
    let mut buf = vec![0u8; len as usize];
    b.read_at(&mut buf, 0)?;
    Ok(buf)
}

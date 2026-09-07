//! 全内存 Fs。测试、临时库与 wasm（OPFS 接入前）使用。
//! rename/remove_dir_all 按"整目录搬移/清理"语义模拟，不追求 POSIX 细节。

use std::collections::{HashMap, HashSet};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use super::{Fs, StorageBackend};

struct MemoryFile {
    data: RwLock<Vec<u8>>,
}

/// 内存文件视图，可克隆共享同一份数据。
pub struct MemoryBackend {
    file: Arc<MemoryFile>,
}

impl StorageBackend for MemoryBackend {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        let data = self.file.data.read().map_err(poison)?;
        let start = offset as usize;
        let end = start.checked_add(buf.len()).ok_or_else(eof)?;
        if end > data.len() {
            return Err(eof());
        }
        buf.copy_from_slice(&data[start..end]);
        Ok(())
    }

    fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<()> {
        let mut data = self.file.data.write().map_err(poison)?;
        let start = offset as usize;
        let end = start + buf.len();
        if end > data.len() {
            data.resize(end, 0);
        }
        data[start..end].copy_from_slice(buf);
        Ok(())
    }

    fn len(&self) -> io::Result<u64> {
        let data = self.file.data.read().map_err(poison)?;
        Ok(data.len() as u64)
    }

    fn truncate(&self, new_len: u64) -> io::Result<()> {
        let mut data = self.file.data.write().map_err(poison)?;
        data.resize(new_len as usize, 0);
        Ok(())
    }

    fn sync(&self) -> io::Result<()> {
        Ok(())
    }
}

fn eof() -> io::Error {
    io::Error::new(io::ErrorKind::UnexpectedEof, "unexpected eof")
}

fn poison<T>(e: std::sync::PoisonError<T>) -> io::Error {
    io::Error::other(format!("lock poisoned: {e}"))
}

#[derive(Default)]
struct FsState {
    files: HashMap<PathBuf, Arc<MemoryFile>>,
    dirs: HashSet<PathBuf>,
}

/// 内存文件系统。
#[derive(Default)]
pub struct MemoryFs {
    state: Mutex<FsState>,
}

impl MemoryFs {
    /// 新建空内存文件系统。
    pub fn new() -> Self {
        Self::default()
    }

    fn with_state<T>(&self, f: impl FnOnce(&mut FsState) -> io::Result<T>) -> io::Result<T> {
        let mut st = self.state.lock().map_err(poison)?;
        f(&mut st)
    }
}

impl Fs for MemoryFs {
    fn create(&self, path: &Path) -> io::Result<Box<dyn StorageBackend>> {
        self.with_state(|st| {
            let file = Arc::new(MemoryFile {
                data: RwLock::new(Vec::new()),
            });
            st.files.insert(path.to_path_buf(), file.clone());
            Ok(Box::new(MemoryBackend { file }) as Box<dyn StorageBackend>)
        })
    }

    fn open_rw(&self, path: &Path) -> io::Result<Box<dyn StorageBackend>> {
        self.with_state(|st| {
            let file = st.files.get(path).cloned().ok_or_else(not_found)?;
            Ok(Box::new(MemoryBackend { file }) as Box<dyn StorageBackend>)
        })
    }

    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        self.with_state(|st| {
            if let Some(file) = st.files.remove(from) {
                st.files.insert(to.to_path_buf(), file);
                return Ok(());
            }
            if !st.dirs.remove(from) {
                return Err(not_found());
            }
            // 目录搬移: 重挂所有以 from 为前缀的条目。
            let children: Vec<PathBuf> = st
                .files
                .keys()
                .chain(st.dirs.iter())
                .filter(|p| p.starts_with(from))
                .filter(|p| *p != from)
                .cloned()
                .collect();
            st.dirs.insert(to.to_path_buf());
            for c in children {
                let target = match c.strip_prefix(from) {
                    Ok(rel) => to.join(rel),
                    Err(_) => continue,
                };
                if let Some(f) = st.files.remove(&c) {
                    st.files.insert(target, f);
                } else if st.dirs.remove(&c) {
                    st.dirs.insert(target);
                }
            }
            Ok(())
        })
    }

    fn remove_file(&self, path: &Path) -> io::Result<()> {
        self.with_state(|st| st.files.remove(path).map(|_| ()).ok_or_else(not_found))
    }

    fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
        self.with_state(|st| {
            st.files.retain(|p, _| !p.starts_with(path));
            st.dirs.retain(|p| p != path && !p.starts_with(path));
            Ok(())
        })
    }

    fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        self.with_state(|st| {
            st.dirs.insert(path.to_path_buf());
            Ok(())
        })
    }

    fn exists(&self, path: &Path) -> bool {
        self.state
            .lock()
            .map(|st| st.files.contains_key(path) || st.dirs.contains(path))
            .unwrap_or(false)
    }

    fn list_dir(&self, path: &Path) -> io::Result<Vec<PathBuf>> {
        self.with_state(|st| {
            let mut out: Vec<PathBuf> = st
                .files
                .keys()
                .chain(st.dirs.iter())
                .filter(|p| p.parent() == Some(path))
                .cloned()
                .collect();
            out.sort();
            Ok(out)
        })
    }
}

fn not_found() -> io::Error {
    io::Error::new(io::ErrorKind::NotFound, "no such file or directory")
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::atomic_write;

    #[test]
    fn atomic_write_via_rename() {
        let fs = MemoryFs::new();
        let p = Path::new("/db/manifest.json");
        atomic_write(&fs, p, b"v1").unwrap();
        assert_eq!(fs.open_rw(p).unwrap().len().unwrap(), 2);
        atomic_write(&fs, p, b"v2-longer").unwrap();
        let mut buf = vec![0u8; 9];
        fs.open_rw(p).unwrap().read_at(&mut buf, 0).unwrap();
        assert_eq!(buf, b"v2-longer");
        assert!(!fs.exists(&p.with_extension("tmp")));
    }

    #[test]
    fn dir_rename_moves_children() {
        let fs = MemoryFs::new();
        fs.create_dir_all(Path::new("/db/seg-1.tmp")).unwrap();
        let f = fs.create(Path::new("/db/seg-1.tmp/data.bin")).unwrap();
        f.write_at(b"abc", 0).unwrap();
        fs.rename(Path::new("/db/seg-1.tmp"), Path::new("/db/seg-1"))
            .unwrap();
        assert!(fs.exists(Path::new("/db/seg-1/data.bin")));
        assert!(!fs.exists(Path::new("/db/seg-1.tmp")));
    }

    #[test]
    fn partial_read_is_eof() {
        let fs = MemoryFs::new();
        let f = fs.create(Path::new("/f")).unwrap();
        f.write_at(b"ab", 0).unwrap();
        let mut buf = [0u8; 4];
        assert!(f.read_at(&mut buf, 1).is_err());
    }
}

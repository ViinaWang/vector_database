//! native 文件系统实现（Linux/macOS/Windows; wasm32 下不编译）。

use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};

use super::{Fs, StorageBackend};

/// 基于单个 `std::fs::File` 的后端，unix/windows 各自的定位读写 API。
pub struct FileBackend {
    file: File,
}

impl FileBackend {
    fn open(path: &Path, truncate: bool) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(truncate)
            .open(path)?;
        Ok(Self { file })
    }
}

impl StorageBackend for FileBackend {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        read_exact_at(&self.file, buf, offset)
    }

    fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<()> {
        write_all_at(&self.file, buf, offset)
    }

    fn len(&self) -> io::Result<u64> {
        Ok(self.file.metadata()?.len())
    }

    fn truncate(&self, new_len: u64) -> io::Result<()> {
        self.file.set_len(new_len)
    }

    fn sync(&self) -> io::Result<()> {
        self.file.sync_data()
    }
}

#[cfg(unix)]
fn read_exact_at(f: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    f.read_exact_at(buf, offset)
}

#[cfg(unix)]
fn write_all_at(f: &File, buf: &[u8], offset: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    f.write_all_at(buf, offset)
}

#[cfg(windows)]
fn read_exact_at(f: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut done = 0;
    while done < buf.len() {
        let n = f.seek_read(&mut buf[done..], offset + done as u64)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "unexpected eof",
            ));
        }
        done += n;
    }
    Ok(())
}

#[cfg(windows)]
fn write_all_at(f: &File, buf: &[u8], offset: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut done = 0;
    while done < buf.len() {
        let n = f.seek_write(&buf[done..], offset + done as u64)?;
        if n == 0 {
            return Err(io::Error::new(io::ErrorKind::WriteZero, "write returned 0"));
        }
        done += n;
    }
    Ok(())
}

/// native 目录 Fs。
pub struct NativeFs;

impl Fs for NativeFs {
    fn create(&self, path: &Path) -> io::Result<Box<dyn StorageBackend>> {
        Ok(Box::new(FileBackend::open(path, true)?))
    }

    fn open_rw(&self, path: &Path) -> io::Result<Box<dyn StorageBackend>> {
        Ok(Box::new(FileBackend::open(path, false)?))
    }

    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        std::fs::rename(from, to)
    }

    fn remove_file(&self, path: &Path) -> io::Result<()> {
        std::fs::remove_file(path)
    }

    fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
        match std::fs::remove_dir_all(path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e),
        }
    }

    fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        std::fs::create_dir_all(path)
    }

    fn exists(&self, path: &Path) -> bool {
        path.exists()
    }

    fn list_dir(&self, path: &Path) -> io::Result<Vec<PathBuf>> {
        let mut out = Vec::new();
        for entry in std::fs::read_dir(path)? {
            out.push(entry?.path());
        }
        out.sort();
        Ok(out)
    }
}

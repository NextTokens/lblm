//! CLI behaviour: safety properties found in review (no data loss, no clobbering, metadata kept,
//! concatenated streams, stdin/-o), exercised through the real binary.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_blmz"))
}

struct Dir(PathBuf);
impl Dir {
    fn new(tag: &str) -> Dir {
        let p = std::env::temp_dir().join(format!("blmz-cli-{}-{}", tag, std::process::id()));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        Dir(p)
    }
    fn p(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn run(cmd: &mut Command) -> Output {
    cmd.stdin(Stdio::null()).output().unwrap()
}

fn sample() -> Vec<u8> {
    let mut v = Vec::new();
    for i in 0..400u32 {
        v.extend_from_slice(format!("line {} of the sample, with some repeated words words words\n", i % 37).as_bytes());
    }
    v
}

fn no_partials(dir: &Path) {
    for e in fs::read_dir(dir).unwrap() {
        let name = e.unwrap().file_name();
        assert!(!name.to_string_lossy().contains("blmz-partial"), "leftover temp file {:?}", name);
    }
}

#[test]
fn compress_decompress_keep_and_force() {
    let d = Dir::new("basic");
    let f = d.p("a.txt");
    fs::write(&f, sample()).unwrap();
    assert!(run(bin().args(["-1", "-q"]).arg(&f)).status.success());
    assert!(f.exists(), "input kept by default");
    let z = d.p("a.txt.blz");
    assert!(z.exists());
    // refuses to overwrite without -f
    let o = run(bin().args(["-1", "-q"]).arg(&f));
    assert!(!o.status.success());
    fs::remove_file(&f).unwrap();
    assert!(run(bin().args(["-d", "-q"]).arg(&z)).status.success());
    assert_eq!(fs::read(&f).unwrap(), sample());
    assert!(run(bin().args(["-t", "-q"]).arg(&z)).status.success());
    no_partials(&d.0);
}

#[test]
fn rm_never_deletes_when_output_is_the_input() {
    let d = Dir::new("rm");
    let f = d.p("x.bin");
    fs::write(&f, sample()).unwrap();
    let o = run(bin().args(["-1", "-q", "-f", "--rm", "-o"]).arg(&f).arg(&f));
    assert!(!o.status.success(), "must refuse to write over the input");
    assert_eq!(fs::read(&f).unwrap(), sample(), "input intact");
    // --rm with a real output removes the input only after success
    let z = d.p("y.blz");
    assert!(run(bin().args(["-1", "-q", "--rm", "-o"]).arg(&z).arg(&f)).status.success());
    assert!(!f.exists());
    assert!(run(bin().args(["-t", "-q"]).arg(&z)).status.success());
    no_partials(&d.0);
}

#[test]
fn concatenated_streams_decode_in_sequence() {
    let d = Dir::new("cat");
    let (a, b) = (d.p("a"), d.p("b"));
    fs::write(&a, b"first part\n").unwrap();
    fs::write(&b, b"second part\n").unwrap();
    let o = run(bin().args(["-1", "-c"]).arg(&a).arg(&b));
    assert!(o.status.success());
    let both = d.p("both.blz");
    fs::write(&both, &o.stdout).unwrap();
    let o = run(bin().args(["-d", "-c"]).arg(&both));
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(o.stdout, b"first part\nsecond part\n");
    assert!(run(bin().args(["-t", "-q"]).arg(&both)).status.success());
}

#[test]
fn stdin_with_output_file() {
    let d = Dir::new("stdin");
    let z = d.p("s.blz");
    let mut child = bin()
        .args(["-1", "-o"])
        .arg(&z)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    {
        use std::io::Write;
        child.stdin.take().unwrap().write_all(&sample()).unwrap();
    }
    let o = child.wait_with_output().unwrap();
    assert!(o.status.success());
    assert!(o.stdout.is_empty(), "-o must write the file, not stdout");
    let o = run(bin().args(["-d", "-c"]).arg(&z));
    assert_eq!(o.stdout, sample());
}

#[test]
fn corrupt_input_fails_and_leaves_nothing_behind() {
    let d = Dir::new("corrupt");
    let f = d.p("c.txt");
    fs::write(&f, sample()).unwrap();
    assert!(run(bin().args(["-1", "-q"]).arg(&f)).status.success());
    let z = d.p("c.txt.blz");
    let mut bytes = fs::read(&z).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0x10;
    fs::write(&z, &bytes).unwrap();
    fs::remove_file(&f).unwrap();
    let o = run(bin().args(["-d", "-q"]).arg(&z));
    assert!(!o.status.success());
    assert!(!f.exists(), "no output under the real name after a failed decode");
    no_partials(&d.0);
}

#[test]
fn max_output_stops_decoding() {
    let d = Dir::new("maxout");
    let f = d.p("m.txt");
    fs::write(&f, sample()).unwrap();
    assert!(run(bin().args(["-1", "-q"]).arg(&f)).status.success());
    let o = run(bin().args(["-d", "-c", "--max-output", "1K"]).arg(d.p("m.txt.blz")));
    assert!(!o.status.success());
}

#[test]
fn memlimit_refuses_large_levels() {
    let d = Dir::new("memlimit");
    let f = d.p("l.txt");
    fs::write(&f, sample()).unwrap();
    let o = run(bin().args(["-9", "-q", "--memlimit", "64"]).arg(&f));
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("memlimit"));
    assert!(!d.p("l.txt.blz").exists());
}

#[cfg(unix)]
#[test]
fn permissions_and_mtime_follow_the_input() {
    use std::os::unix::fs::PermissionsExt;
    let d = Dir::new("perm");
    let f = d.p("secret.txt");
    fs::write(&f, sample()).unwrap();
    fs::set_permissions(&f, fs::Permissions::from_mode(0o600)).unwrap();
    let old = std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000_000);
    fs::File::options().write(true).open(&f).unwrap().set_modified(old).unwrap();
    assert!(run(bin().args(["-1", "-q"]).arg(&f)).status.success());
    let m = fs::metadata(d.p("secret.txt.blz")).unwrap();
    assert_eq!(m.permissions().mode() & 0o777, 0o600);
    assert_eq!(m.modified().unwrap(), old);
}

#[cfg(unix)]
#[test]
fn planted_symlink_at_a_temp_name_is_not_followed() {
    let d = Dir::new("symlink");
    let f = d.p("in.txt");
    fs::write(&f, sample()).unwrap();
    let victim = d.p("victim");
    fs::write(&victim, b"precious").unwrap();
    // plant links at every name an old build would have used, and at the new pattern's first slot
    for name in [
        "in.txt.blz.blmz-partial".to_string(),
        format!("in.txt.blz.{}-0.blmz-partial", std::process::id()),
    ] {
        std::os::unix::fs::symlink(&victim, d.p(&name)).unwrap();
    }
    assert!(run(bin().args(["-1", "-q"]).arg(&f)).status.success());
    assert_eq!(fs::read(&victim).unwrap(), b"precious");
    assert!(run(bin().args(["-t", "-q"]).arg(d.p("in.txt.blz"))).status.success());
}

#[cfg(unix)]
#[test]
fn non_utf8_file_names() {
    use std::os::unix::ffi::OsStrExt;
    let d = Dir::new("nonutf8");
    let name = std::ffi::OsStr::from_bytes(b"caf\xe9.txt");
    let f = d.0.join(name);
    fs::write(&f, sample()).unwrap();
    let o = run(bin().args(["-1", "-q"]).arg(&f));
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    fs::remove_file(&f).unwrap();
    let mut z = f.as_os_str().to_owned();
    z.push(".blz");
    assert!(run(bin().args(["-d", "-q"]).arg(&z)).status.success());
    assert_eq!(fs::read(&f).unwrap(), sample());
}

#[test]
fn usage_errors_exit_2() {
    assert_eq!(run(bin().arg("--no-such-flag")).status.code(), Some(2));
    assert_eq!(run(bin().args(["-o", "a", "b", "c"])).status.code(), Some(2));
}

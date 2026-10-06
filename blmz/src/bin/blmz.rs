//! blmz command-line tool.

use std::ffi::{OsStr, OsString};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

const USAGE: &str = "\
blmz - LBLM bit-native context-mixing compressor

USAGE:
  blmz [OPTIONS] [FILE...]          compress FILE to FILE.blz (stdin -> stdout if no FILE or '-')
  blmz -d [OPTIONS] [FILE.blz...]   decompress (concatenated streams are decoded in sequence)
  blmz -t FILE.blz...               verify integrity (decode, check length and CRC-32)
  blmz -l FILE.blz...               show header information
  blmz bench FILE...                compress + decompress in memory, report ratio and speed

OPTIONS:
  -1 .. -9            level (memory/ratio trade-off, default -6)
  -d, --decompress    decompress
  -c, --stdout        write to standard output
  -o, --output F      output file ('-' = stdout; with a single input)
  -f, --force         overwrite existing outputs; allow binary output to a terminal
      --rm            delete the input after the output is safely written (default: keep)
      --memlimit M    use at most M MiB for model tables + history (compress: refuse the
                      level; decompress: refuse the stream). Default 4096.
      --max-output N  decompress: stop after N bytes of output (K/M/G suffixes). Decoding is
                      as slow as compression, so this also bounds CPU time on untrusted input.
  -q, --quiet         no summary on stderr
  -h, --help          this help
  -V, --version       version

Combined short flags work (-dc, -9f). Exit status: 0 ok, 1 error, 2 usage.
";

#[derive(PartialEq, Clone, Copy)]
enum Mode {
    Compress,
    Decompress,
    Test,
    List,
    Bench,
}

struct Cli {
    mode: Mode,
    level: u8,
    stdout: bool,
    output: Option<PathBuf>,
    force: bool,
    rm: bool,
    memlimit: u64,
    max_output: u64,
    quiet: bool,
    files: Vec<OsString>,
}

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn parse_size(s: &str) -> Result<u64, String> {
    let (num, mult) = match s.chars().last() {
        Some('K' | 'k') => (&s[..s.len() - 1], 1u64 << 10),
        Some('M' | 'm') => (&s[..s.len() - 1], 1 << 20),
        Some('G' | 'g') => (&s[..s.len() - 1], 1 << 30),
        _ => (s, 1),
    };
    num.parse::<u64>()
        .map(|n| n.saturating_mul(mult))
        .map_err(|_| format!("bad size '{}'", s))
}

fn parse() -> Result<Cli, String> {
    let mut cli = Cli {
        mode: Mode::Compress,
        level: blmz::DEFAULT_LEVEL,
        stdout: false,
        output: None,
        force: false,
        rm: false,
        memlimit: 4096,
        max_output: u64::MAX,
        quiet: false,
        files: vec![],
    };
    let mut args: Vec<OsString> = std::env::args_os().skip(1).collect();
    if args.first().map(|a| a == "bench").unwrap_or(false) {
        args.remove(0);
        cli.mode = Mode::Bench;
    }
    let mut it = args.into_iter();
    let mut only_files = false;
    let value = |it: &mut std::vec::IntoIter<OsString>, opt: &str| -> Result<OsString, String> {
        it.next().ok_or_else(|| format!("{} needs a value", opt))
    };
    while let Some(a) = it.next() {
        let s = match a.to_str() {
            Some(s) if !only_files && s.starts_with('-') && s != "-" => s.to_string(),
            _ => {
                cli.files.push(a);
                continue;
            }
        };
        // expand combined short flags (-dc, -9f); options with a value must be given alone
        let flags: Vec<String> = if !s.starts_with("--") && s.len() > 2 {
            s[1..].chars().map(|c| format!("-{}", c)).collect()
        } else {
            vec![s.clone()]
        };
        for f in flags {
            match f.as_str() {
                "--" => only_files = true,
                "-d" | "--decompress" => cli.mode = Mode::Decompress,
                "-t" | "--test" => cli.mode = Mode::Test,
                "-l" | "--list" => cli.mode = Mode::List,
                "-c" | "--stdout" => cli.stdout = true,
                "-f" | "--force" => cli.force = true,
                "--rm" => cli.rm = true,
                "-q" | "--quiet" => cli.quiet = true,
                "-o" | "--output" if s == f => cli.output = Some(PathBuf::from(value(&mut it, "-o")?)),
                "--memlimit" => {
                    let v = value(&mut it, "--memlimit")?;
                    cli.memlimit = v.to_str().and_then(|v| v.parse().ok()).ok_or("bad --memlimit (MiB)")?
                }
                "--max-output" => {
                    let v = value(&mut it, "--max-output")?;
                    cli.max_output = parse_size(v.to_str().ok_or("bad --max-output")?)?
                }
                "-h" | "--help" => {
                    print!("{}", USAGE);
                    std::process::exit(0);
                }
                "-V" | "--version" => {
                    let _ = writeln!(
                        io::stdout(),
                        "blmz {} (format {}, model {:#x})",
                        env!("CARGO_PKG_VERSION"),
                        blmz::format::VERSION,
                        blmz::format::MODEL_ID
                    );
                    std::process::exit(0);
                }
                d if d.len() == 2 && (b'1'..=b'9').contains(&d.as_bytes()[1]) => cli.level = d.as_bytes()[1] - b'0',
                other => return Err(format!("unknown option {} (in {})", other, s)),
            }
        }
    }
    if cli.output.is_some() && cli.files.len() > 1 {
        return Err("-o works with a single input".into());
    }
    if cli.output.as_deref() == Some(Path::new("-")) {
        cli.output = None;
        cli.stdout = true;
    }
    if cli.stdout && cli.rm {
        return Err("--rm cannot be combined with -c".into());
    }
    Ok(cli)
}

fn main() -> ExitCode {
    let cli = match parse() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("blmz: {}\nTry 'blmz --help'.", e);
            return ExitCode::from(2);
        }
    };
    let inputs: Vec<OsString> = if cli.files.is_empty() {
        vec![OsString::from("-")]
    } else {
        cli.files.clone()
    };
    let mut failed = false;
    for f in &inputs {
        let r = match (cli.mode, f == OsStr::new("-")) {
            (Mode::List | Mode::Bench, true) => Err("needs a file argument".into()),
            (Mode::Compress | Mode::Decompress | Mode::Test, true) => run_stdin(&cli),
            (Mode::Compress | Mode::Decompress, false) => run_file(&cli, Path::new(f)),
            (Mode::Test, false) => test_file(&cli, Path::new(f)),
            (Mode::List, false) => list_file(Path::new(f)),
            (Mode::Bench, false) => bench_file(&cli, Path::new(f)),
        };
        if let Err(e) = r {
            if is_broken_pipe(&*e) {
                return ExitCode::from(1);
            }
            eprintln!("blmz: {}: {}", Path::new(f).display(), e);
            failed = true;
        }
    }
    if failed {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

fn is_broken_pipe(e: &(dyn std::error::Error + 'static)) -> bool {
    let io = e.downcast_ref::<io::Error>().or_else(|| match e.downcast_ref::<blmz::Error>() {
        Some(blmz::Error::Io(x)) => Some(x),
        _ => None,
    });
    io.map(|x| x.kind() == io::ErrorKind::BrokenPipe).unwrap_or(false)
}

fn limits(cli: &Cli) -> blmz::Limits {
    blmz::Limits::default()
        .max_memory(cli.memlimit.saturating_mul(1 << 20))
        .max_output(cli.max_output)
}

fn options(cli: &Cli, content_length: Option<u64>) -> Res<blmz::Options> {
    let opts = blmz::Options::level(cli.level);
    let (_, params) = opts.resolve()?;
    let need = params.total_memory(content_length);
    let limit = cli.memlimit.saturating_mul(1 << 20);
    if need > limit {
        return Err(format!(
            "level {} needs {} MiB, over --memlimit {} MiB (choose a lower level)",
            cli.level,
            need.div_ceil(1 << 20),
            cli.memlimit
        )
        .into());
    }
    Ok(opts)
}

/// Decode every stream in `input` (concatenated .blz streams decode in sequence, like gzip).
fn decode_all<R: BufRead, W: Write>(input: &mut R, mut out: W, limits: &blmz::Limits) -> Result<u64, blmz::Error> {
    let mut total = 0u64;
    loop {
        let remaining = blmz::Limits::default()
            .max_memory(limits.max_memory)
            .max_output(limits.max_output.saturating_sub(total));
        total += blmz::decompress_one(input, &mut out, &remaining)?;
        if input.fill_buf()?.is_empty() {
            return Ok(total);
        }
    }
}

fn run_stdin(cli: &Cli) -> Res<()> {
    let stdin = io::stdin();
    if cli.mode != Mode::Compress && stdin.is_terminal() && !cli.force {
        return Err("refusing to read compressed data from a terminal (use -f)".into());
    }
    let mut input = BufReader::new(stdin.lock());
    if let (Some(path), Mode::Compress | Mode::Decompress) = (&cli.output, cli.mode) {
        let out = Output::create(cli, path, None)?;
        return out.finish_with(cli, None, |w| match cli.mode {
            Mode::Compress => Ok(blmz::compress_stream(&mut input, w, &options(cli, None)?, None)?),
            _ => Ok(decode_all(&mut input, w, &limits(cli))?),
        });
    }
    match cli.mode {
        Mode::Compress => {
            if io::stdout().is_terminal() && !cli.force {
                return Err("refusing to write compressed data to a terminal (use -f, or redirect)".into());
            }
            blmz::compress_stream(input, BufWriter::new(io::stdout().lock()), &options(cli, None)?, None)?;
        }
        Mode::Decompress => {
            let mut w = BufWriter::new(io::stdout().lock());
            decode_all(&mut input, &mut w, &limits(cli))?;
            w.flush()?;
        }
        _ => {
            let n = decode_all(&mut input, io::sink(), &limits(cli))?;
            if !cli.quiet {
                eprintln!("(stdin): OK ({} bytes)", n);
            }
        }
    }
    Ok(())
}

fn output_path(cli: &Cli, input: &Path) -> Res<PathBuf> {
    if let Some(o) = &cli.output {
        return Ok(o.clone());
    }
    let os = input.as_os_str();
    match cli.mode {
        Mode::Compress => {
            let mut s = os.to_owned();
            s.push(".blz");
            Ok(PathBuf::from(s))
        }
        _ => {
            let bytes = os.as_encoded_bytes();
            match bytes.strip_suffix(b".blz") {
                Some(stem) if !stem.is_empty() && !stem.ends_with(b"/") => {
                    // SAFETY: `stem` is a prefix of a valid OsStr, cut immediately before an ASCII '.'
                    Ok(PathBuf::from(unsafe { OsStr::from_encoded_bytes_unchecked(stem) }))
                }
                _ => Err("unknown suffix (expected .blz); use -o or -c".into()),
            }
        }
    }
}

/// True if `a` and `b` name the same existing file.
fn same_file(a: &Path, b: &Path) -> bool {
    match (fs::metadata(a), fs::metadata(b)) {
        #[cfg(unix)]
        (Ok(x), Ok(y)) => {
            use std::os::unix::fs::MetadataExt;
            x.dev() == y.dev() && x.ino() == y.ino()
        }
        #[cfg(not(unix))]
        (Ok(_), Ok(_)) => match (fs::canonicalize(a), fs::canonicalize(b)) {
            (Ok(x), Ok(y)) => x == y,
            _ => false,
        },
        _ => false,
    }
}

/// An output file written under a unique temporary name, created exclusively (never follows or
/// clobbers anything already there) with owner-only permissions, and renamed into place only
/// after a complete, synced write. Dropping it unfinished removes the temporary.
struct Output {
    tmp: PathBuf,
    dst: PathBuf,
    file: Option<File>,
}

impl Output {
    fn create(cli: &Cli, dst: &Path, input: Option<&Path>) -> Res<Output> {
        if let Some(input) = input {
            if same_file(input, dst) {
                return Err(format!("output {} is the input file", dst.display()).into());
            }
        }
        if fs::symlink_metadata(dst).is_ok() && !cli.force {
            return Err(format!("{} exists (use -f to overwrite)", dst.display()).into());
        }
        let pid = std::process::id();
        for n in 0u32..1000 {
            let mut s = dst.as_os_str().to_owned();
            s.push(format!(".{}-{}.blmz-partial", pid, n));
            let tmp = PathBuf::from(s);
            let mut oo = OpenOptions::new();
            oo.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                oo.mode(0o600);
            }
            match oo.open(&tmp) {
                Ok(file) => {
                    return Ok(Output {
                        tmp,
                        dst: dst.to_path_buf(),
                        file: Some(file),
                    })
                }
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(format!("cannot create {}: {}", tmp.display(), e).into()),
            }
        }
        Err("could not create a unique temporary file".into())
    }

    /// Run `body` against the buffered output, then copy metadata from `meta_from`, sync, and
    /// rename into place.
    fn finish_with<F>(mut self, cli: &Cli, meta_from: Option<&fs::Metadata>, body: F) -> Res<()>
    where
        F: FnOnce(&mut BufWriter<&File>) -> Res<u64>,
    {
        let file = self.file.take().expect("open output");
        {
            let mut w = BufWriter::with_capacity(1 << 16, &file);
            body(&mut w)?;
            w.flush()?;
        }
        if let Some(m) = meta_from {
            // permissions and modification time follow the input (gzip/xz behaviour)
            fs::set_permissions(&self.tmp, m.permissions())?;
            if let Ok(t) = m.modified() {
                let _ = file.set_modified(t);
            }
        }
        file.sync_all()?;
        drop(file);
        if fs::symlink_metadata(&self.dst).is_ok() && !cli.force {
            return Err(format!("{} appeared while writing (use -f to overwrite)", self.dst.display()).into());
        }
        fs::rename(&self.tmp, &self.dst).map_err(|e| format!("cannot rename to {}: {}", self.dst.display(), e))?;
        self.tmp = PathBuf::new(); // committed: nothing to clean up
        sync_parent(&self.dst);
        Ok(())
    }
}

impl Drop for Output {
    fn drop(&mut self) {
        if !self.tmp.as_os_str().is_empty() {
            let _ = fs::remove_file(&self.tmp);
        }
    }
}

fn sync_parent(path: &Path) {
    #[cfg(unix)]
    {
        let dir = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
        if let Ok(d) = File::open(dir) {
            let _ = d.sync_all();
        }
    }
    #[cfg(not(unix))]
    let _ = path;
}

fn run_file(cli: &Cli, input: &Path) -> Res<()> {
    let meta = fs::metadata(input)?;
    if meta.is_dir() {
        return Err("is a directory".into());
    }
    if !meta.is_file() && !cli.stdout {
        return Err("not a regular file (use -c to stream it)".into());
    }
    // only a regular file has a trustworthy length (procfs/sysfs report 0, pipes have none)
    let len = (meta.is_file() && meta.len() > 0).then_some(meta.len());
    let t0 = Instant::now();
    let mut src = BufReader::new(File::open(input)?);
    let compressing = cli.mode == Mode::Compress;
    let run = |w: &mut dyn Write, src: &mut BufReader<File>| -> Res<u64> {
        if compressing {
            match blmz::compress_stream(src, w, &options(cli, len)?, len) {
                Err(blmz::Error::LengthMismatch { declared, actual }) => {
                    Err(format!("input changed size while being compressed ({} -> {} bytes)", declared, actual).into())
                }
                r => Ok(r?),
            }
        } else {
            Ok(decode_all(src, w, &limits(cli))?)
        }
    };
    if cli.stdout {
        let stdout = io::stdout();
        if compressing && stdout.is_terminal() && !cli.force {
            return Err("refusing to write compressed data to a terminal (use -f, or redirect)".into());
        }
        let mut w = BufWriter::new(stdout.lock());
        run(&mut w, &mut src)?;
        w.flush()?;
        return Ok(());
    }
    let dst = output_path(cli, input)?;
    let out = Output::create(cli, &dst, Some(input))?;
    let mut n = 0u64;
    out.finish_with(cli, Some(&meta), |w| {
        n = run(w, &mut src)?;
        Ok(n)
    })?;
    if cli.rm {
        fs::remove_file(input)?;
        sync_parent(input);
    }
    if !cli.quiet {
        let secs = t0.elapsed().as_secs_f64();
        let out_len = fs::metadata(&dst)?.len();
        let (raw, packed) = if compressing { (n, out_len) } else { (n, meta.len()) };
        eprintln!(
            "{} -> {}: {} -> {} bytes ({:.3} bits/byte, {:.2}%), {:.1}s, {:.1} KB/s",
            input.display(),
            dst.display(),
            if compressing { raw } else { packed },
            if compressing { packed } else { raw },
            if raw > 0 { packed as f64 * 8.0 / raw as f64 } else { 0.0 },
            if raw > 0 { packed as f64 * 100.0 / raw as f64 } else { 0.0 },
            secs,
            raw as f64 / 1024.0 / secs.max(1e-9)
        );
    }
    Ok(())
}

fn test_file(cli: &Cli, input: &Path) -> Res<()> {
    let n = decode_all(&mut BufReader::new(File::open(input)?), io::sink(), &limits(cli))?;
    if !cli.quiet {
        eprintln!("{}: OK ({} bytes)", input.display(), n);
    }
    Ok(())
}

fn list_file(input: &Path) -> Res<()> {
    let h = blmz::read_header(BufReader::new(File::open(input)?))?;
    let size = fs::metadata(input)?.len();
    let p = h.params;
    let mut o = io::stdout().lock();
    writeln!(o, "{}", input.display())?;
    writeln!(o, "  format {}  model {:#x}  level {}", h.version, h.model_id, h.level)?;
    writeln!(
        o,
        "  tables: obits {} hbits {} mbits {} sbits {} selvbits {} selubits {}  window 2^{}",
        p.obits, p.hbits, p.mbits, p.sbits, p.selvbits, p.selubits, p.window_log
    )?;
    writeln!(
        o,
        "  memory to decode: {:.1} MiB ({:.1} MiB tables + {:.1} MiB history)",
        p.total_memory(h.content_length) as f64 / 1048576.0,
        p.memory_bytes() as f64 / 1048576.0,
        p.history_bytes(h.content_length) as f64 / 1048576.0
    )?;
    match h.content_length {
        Some(n) => writeln!(
            o,
            "  content {} bytes, stored {} bytes ({:.3} bits/byte)",
            n,
            size,
            if n > 0 { size as f64 * 8.0 / n as f64 } else { 0.0 }
        )?,
        None => writeln!(o, "  content length not recorded (streamed), stored {} bytes", size)?,
    }
    Ok(())
}

fn bench_file(cli: &Cli, input: &Path) -> Res<()> {
    let mut data = Vec::new();
    File::open(input)?.read_to_end(&mut data)?;
    let opts = options(cli, Some(data.len() as u64))?;
    let t0 = Instant::now();
    let packed = blmz::compress(&data, &opts)?;
    let tc = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let back = blmz::decompress_with_limits(&packed, &limits(cli))?;
    let td = t1.elapsed().as_secs_f64();
    if back != data {
        return Err("ROUND TRIP MISMATCH".into());
    }
    let kb = data.len() as f64 / 1024.0;
    writeln!(
        io::stdout().lock(),
        "{}\tlevel {}\t{} -> {} bytes\t{:.4} bits/byte\tcomp {:.1} KB/s\tdecomp {:.1} KB/s\tround-trip OK",
        input.display(),
        cli.level,
        data.len(),
        packed.len(),
        if data.is_empty() {
            0.0
        } else {
            packed.len() as f64 * 8.0 / data.len() as f64
        },
        kb / tc.max(1e-9),
        kb / td.max(1e-9)
    )?;
    Ok(())
}

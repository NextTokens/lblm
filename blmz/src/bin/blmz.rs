//! blmz command-line tool.

use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

const USAGE: &str = "\
blmz - LBLM bit-native context-mixing compressor

USAGE:
  blmz [OPTIONS] [FILE...]          compress FILE to FILE.blz (stdin -> stdout if no FILE)
  blmz -d [OPTIONS] [FILE.blz...]   decompress
  blmz -t FILE.blz...               verify integrity (decode, check length and CRC-32)
  blmz -l FILE.blz...               show header information
  blmz bench FILE...                compress + decompress in memory, report ratio and speed

OPTIONS:
  -1 .. -9          level (memory/ratio trade-off, default -6); see `blmz -l` / ROADMAP.md
  -d, --decompress  decompress
  -c, --stdout      write to standard output
  -o, --output F    output file (single input only)
  -f, --force       overwrite existing output files
      --rm          delete the input after success (default: keep)
      --memlimit M  refuse streams needing more than M MiB of model memory (default 8192)
  -q, --quiet       no progress/summary on stderr
  -h, --help        this help
  -V, --version     version
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
    quiet: bool,
    files: Vec<PathBuf>,
}

fn parse() -> Result<Cli, String> {
    let mut cli = Cli {
        mode: Mode::Compress,
        level: blmz::DEFAULT_LEVEL,
        stdout: false,
        output: None,
        force: false,
        rm: false,
        memlimit: 8192,
        quiet: false,
        files: vec![],
    };
    let mut args = std::env::args().skip(1).peekable();
    if args.peek().map(|s| s == "bench").unwrap_or(false) {
        args.next();
        cli.mode = Mode::Bench;
    }
    let mut only_files = false;
    while let Some(a) = args.next() {
        if only_files || !a.starts_with('-') || a == "-" {
            cli.files.push(PathBuf::from(a));
            continue;
        }
        match a.as_str() {
            "--" => only_files = true,
            "-d" | "--decompress" => cli.mode = Mode::Decompress,
            "-t" | "--test" => cli.mode = Mode::Test,
            "-l" | "--list" => cli.mode = Mode::List,
            "-c" | "--stdout" => cli.stdout = true,
            "-f" | "--force" => cli.force = true,
            "--rm" => cli.rm = true,
            "-q" | "--quiet" => cli.quiet = true,
            "-o" | "--output" => cli.output = Some(PathBuf::from(args.next().ok_or("-o needs a file")?)),
            "--memlimit" => {
                cli.memlimit = args
                    .next()
                    .ok_or("--memlimit needs a value")?
                    .parse()
                    .map_err(|_| "bad --memlimit")?
            }
            "-h" | "--help" => {
                print!("{}", USAGE);
                std::process::exit(0);
            }
            "-V" | "--version" => {
                println!(
                    "blmz {} (format {}, model {:#x})",
                    env!("CARGO_PKG_VERSION"),
                    blmz::format::VERSION,
                    blmz::format::MODEL_ID
                );
                std::process::exit(0);
            }
            s if s.len() == 2 && s.as_bytes()[1].is_ascii_digit() && s != "-0" => cli.level = s.as_bytes()[1] - b'0',
            s => return Err(format!("unknown option {}", s)),
        }
    }
    if cli.output.is_some() && cli.files.len() > 1 {
        return Err("-o works with a single input".into());
    }
    Ok(cli)
}

fn main() -> ExitCode {
    let cli = match parse() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("blmz: {}\n\n{}", e, USAGE);
            return ExitCode::from(2);
        }
    };
    let mut failed = false;
    if cli.files.is_empty() || cli.files.iter().all(|f| f.as_os_str() == "-") {
        if let Err(e) = run_stdio(&cli) {
            eprintln!("blmz: {}", e);
            failed = true;
        }
    } else {
        for f in &cli.files {
            let r = match cli.mode {
                Mode::Compress | Mode::Decompress => run_file(&cli, f),
                Mode::Test => test_file(&cli, f),
                Mode::List => list_file(f),
                Mode::Bench => bench_file(&cli, f),
            };
            if let Err(e) = r {
                eprintln!("blmz: {}: {}", f.display(), e);
                failed = true;
            }
        }
    }
    if failed {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

fn limits(cli: &Cli) -> blmz::Limits {
    blmz::Limits {
        max_memory: cli.memlimit.saturating_mul(1 << 20),
        ..Default::default()
    }
}

fn run_stdio(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin().lock();
    let stdout = BufWriter::new(io::stdout().lock());
    match cli.mode {
        Mode::Compress => {
            blmz::compress_stream(stdin, stdout, &blmz::Options::level(cli.level), None)?;
        }
        Mode::Decompress => {
            blmz::decompress_stream(BufReader::new(stdin), stdout, &limits(cli))?;
        }
        Mode::Test => {
            blmz::decompress_stream(BufReader::new(stdin), io::sink(), &limits(cli))?;
        }
        Mode::List | Mode::Bench => return Err("needs a file argument".into()),
    }
    Ok(())
}

fn output_path(cli: &Cli, input: &Path) -> Result<PathBuf, String> {
    if let Some(o) = &cli.output {
        return Ok(o.clone());
    }
    match cli.mode {
        Mode::Compress => {
            let mut s = input.as_os_str().to_owned();
            s.push(".blz");
            Ok(PathBuf::from(s))
        }
        _ => {
            let name = input.to_string_lossy();
            match name.strip_suffix(".blz") {
                Some(stem) if !stem.is_empty() => Ok(PathBuf::from(stem)),
                _ => Err("unknown suffix (expected .blz); use -o or -c".into()),
            }
        }
    }
}

fn run_file(cli: &Cli, input: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let meta = fs::metadata(input)?;
    if !meta.is_file() {
        return Err("not a regular file".into());
    }
    let t0 = Instant::now();
    let src = BufReader::new(File::open(input)?);
    if cli.stdout {
        let out = BufWriter::new(io::stdout().lock());
        match cli.mode {
            Mode::Compress => blmz::compress_stream(src, out, &blmz::Options::level(cli.level), Some(meta.len()))?,
            _ => blmz::decompress_stream(src, out, &limits(cli))?,
        };
        return Ok(());
    }
    let dst = output_path(cli, input)?;
    if dst.exists() && !cli.force {
        return Err(format!("{} exists (use -f to overwrite)", dst.display()).into());
    }
    // write to a temporary sibling, then rename: never leave a half-written output under the real name
    let tmp = {
        let mut s = dst.as_os_str().to_owned();
        s.push(".blmz-partial");
        PathBuf::from(s)
    };
    let result = (|| -> Result<u64, Box<dyn std::error::Error>> {
        let mut out = BufWriter::new(File::create(&tmp)?);
        let n = match cli.mode {
            Mode::Compress => blmz::compress_stream(src, &mut out, &blmz::Options::level(cli.level), Some(meta.len()))?,
            _ => blmz::decompress_stream(src, &mut out, &limits(cli))?,
        };
        out.flush()?;
        out.into_inner().map_err(|e| e.into_error())?.sync_all()?;
        Ok(n)
    })();
    let n = match result {
        Ok(n) => n,
        Err(e) => {
            let _ = fs::remove_file(&tmp);
            return Err(e);
        }
    };
    fs::rename(&tmp, &dst)?;
    if cli.rm {
        fs::remove_file(input)?;
    }
    if !cli.quiet {
        let secs = t0.elapsed().as_secs_f64();
        let out_len = fs::metadata(&dst)?.len();
        let (raw, packed) = if cli.mode == Mode::Compress {
            (n, out_len)
        } else {
            (out_len, meta.len())
        };
        eprintln!(
            "{} -> {}: {} -> {} bytes ({:.3} bits/byte, {:.2}%), {:.1}s, {:.1} KB/s",
            input.display(),
            dst.display(),
            if cli.mode == Mode::Compress { raw } else { packed },
            if cli.mode == Mode::Compress { packed } else { raw },
            if raw > 0 { packed as f64 * 8.0 / raw as f64 } else { 0.0 },
            if raw > 0 { packed as f64 * 100.0 / raw as f64 } else { 0.0 },
            secs,
            raw as f64 / 1024.0 / secs.max(1e-9)
        );
    }
    Ok(())
}

fn test_file(cli: &Cli, input: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let n = blmz::decompress_stream(BufReader::new(File::open(input)?), io::sink(), &limits(cli))?;
    if !cli.quiet {
        eprintln!("{}: OK ({} bytes)", input.display(), n);
    }
    Ok(())
}

fn list_file(input: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let h = blmz::read_header(BufReader::new(File::open(input)?))?;
    let size = fs::metadata(input)?.len();
    let p = h.params;
    println!("{}", input.display());
    println!("  format {}  model {:#x}  level {}", h.version, h.model_id, h.level);
    println!(
        "  tables: obits {} hbits {} mbits {} sbits {} selvbits {} selubits {}  window 2^{}",
        p.obits, p.hbits, p.mbits, p.sbits, p.selvbits, p.selubits, p.window_log
    );
    println!("  model memory {:.1} MiB", p.memory_bytes() as f64 / 1048576.0);
    match h.content_length {
        Some(n) => println!(
            "  content {} bytes, stored {} bytes ({:.3} bits/byte)",
            n,
            size,
            if n > 0 { size as f64 * 8.0 / n as f64 } else { 0.0 }
        ),
        None => println!("  content length not recorded (streamed), stored {} bytes", size),
    }
    Ok(())
}

fn bench_file(cli: &Cli, input: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut data = Vec::new();
    File::open(input)?.read_to_end(&mut data)?;
    let opts = blmz::Options::level(cli.level);
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
    println!(
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
    );
    Ok(())
}

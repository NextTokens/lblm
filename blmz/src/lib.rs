//! `blmz` — the LBLM bit-native context-mixing compressor, packaged for production use.
//!
//! The predictor is a faithful port of the research engine's adopted default configuration
//! (`blmrs/src/bin/strong.rs`); this crate adds what the research engine never had: a real
//! arithmetic coder, a versioned container with checksums, a decoder, portable (platform-
//! independent) math so files decode on any machine, bounded memory, and resource limits for
//! untrusted input.
//!
//! ```no_run
//! let data = std::fs::read("input").unwrap();
//! let packed = blmz::compress(&data, &blmz::Options::default());
//! let back = blmz::decompress(&packed).unwrap();
//! assert_eq!(back, data);
//! ```
//!
//! Compression and decompression are symmetric and slow (tens of KB/s): every bit of the input
//! runs the full model. See ROADMAP.md.

// The model mirrors blmrs/src/bin/strong.rs line by line (index loops included) so parity
// reviews can diff the two; keep that shape.
#![allow(clippy::needless_range_loop)]

pub mod coder;
pub mod crc32;
pub mod format;
pub mod math;
pub mod model;

use coder::{quantize, ByteSource, Decoder, Encoder};
use format::{Header, MODEL_ID, P_MORE, TRAILER_LEN, VERSION};
pub use model::{Model, Params};
use std::io::{self, Read, Write};

/// Compression level presets: table sizes per level (see `Params`). Level 0 is invalid.
/// The parameters are written into every stream header, so presets may be retuned in later
/// releases without breaking old files.
pub fn level_params(level: u8) -> Option<Params> {
    // obits, hbits, mbits, sbits, selvbits, selubits, window_log
    let t: [u8; 7] = match level {
        1 => [16, 16, 16, 14, 16, 16, 24],
        2 => [17, 17, 17, 15, 17, 17, 25],
        3 => [18, 18, 18, 16, 18, 18, 26],
        4 => [19, 19, 19, 17, 19, 19, 27],
        5 => [20, 20, 20, 18, 20, 20, 28],
        6 => [21, 21, 21, 19, 21, 21, 29],
        7 => [22, 22, 22, 20, 22, 22, 30],
        8 => [23, 22, 22, 22, 23, 23, 31],
        9 => [24, 24, 24, 22, 24, 24, 32],
        _ => return None,
    };
    Some(Params {
        obits: t[0],
        hbits: t[1],
        mbits: t[2],
        sbits: t[3],
        selvbits: t[4],
        selubits: t[5],
        window_log: t[6],
    })
}

pub const DEFAULT_LEVEL: u8 = 6;

#[derive(Clone, Debug)]
pub struct Options {
    /// 1..=9; ignored when `params` is set.
    pub level: u8,
    /// Explicit table sizes (overrides `level`; the header records level 0).
    pub params: Option<Params>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            level: DEFAULT_LEVEL,
            params: None,
        }
    }
}

impl Options {
    pub fn level(level: u8) -> Self {
        Options { level, params: None }
    }
    fn resolve(&self) -> Result<(u8, Params), Error> {
        match self.params {
            Some(p) => {
                p.validate().map_err(Error::BadParams)?;
                Ok((0, p))
            }
            None => level_params(self.level).map(|p| (self.level, p)).ok_or(Error::BadLevel(self.level)),
        }
    }
}

/// Resource limits applied when DEcompressing (input may be hostile).
#[derive(Clone, Debug)]
pub struct Limits {
    /// Refuse streams whose model needs more than this many bytes of tables (excl. history).
    pub max_memory: u64,
    /// Stop with `Error::OutputLimit` after this many output bytes.
    pub max_output: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_memory: 8 << 30,
            max_output: u64::MAX,
        }
    }
}

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    /// Input does not start with the blz magic.
    NotBlz,
    UnsupportedVersion(u8),
    /// The stream was written by a different predictor (or a non-portable parity build).
    UnsupportedModel(u8),
    CorruptHeader(&'static str),
    BadParams(String),
    BadLevel(u8),
    MemoryLimit {
        need: u64,
        limit: u64,
    },
    OutputLimit(u64),
    /// The payload or trailer ended early.
    Truncated,
    /// Decoded data disagrees with the stored length or checksum.
    Corrupt(&'static str),
    /// Bytes follow the end of the stream.
    TrailingData,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {}", e),
            Error::NotBlz => write!(f, "not a blz stream (bad magic)"),
            Error::UnsupportedVersion(v) => write!(f, "unsupported format version {} (this build reads {})", v, VERSION),
            Error::UnsupportedModel(m) => write!(f, "unsupported model id {:#x} (this build reads {:#x})", m, MODEL_ID),
            Error::CorruptHeader(s) => write!(f, "corrupt header: {}", s),
            Error::BadParams(s) => write!(f, "invalid model parameters: {}", s),
            Error::BadLevel(l) => write!(f, "invalid level {} (1..=9)", l),
            Error::MemoryLimit { need, limit } => {
                write!(f, "stream needs {} MiB of model memory, limit is {} MiB", need >> 20, limit >> 20)
            }
            Error::OutputLimit(n) => write!(f, "output exceeds the limit of {} bytes", n),
            Error::Truncated => write!(f, "stream is truncated"),
            Error::Corrupt(s) => write!(f, "stream is corrupt: {}", s),
            Error::TrailingData => write!(f, "unexpected data after the end of the stream"),
        }
    }
}

impl std::error::Error for Error {}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<Error> for io::Error {
    fn from(e: Error) -> Self {
        match e {
            Error::Io(e) => e,
            other => io::Error::new(io::ErrorKind::InvalidData, other),
        }
    }
}

/// Codes bytes into a payload. Usually driven through `compress` / `compress_stream`.
pub struct Compressor {
    model: Model,
    enc: Encoder,
    crc: crc32::Crc32,
    len: u64,
}

impl Compressor {
    pub fn new(params: Params) -> Self {
        Compressor {
            model: Model::new(params),
            enc: Encoder::new(),
            crc: crc32::Crc32::new(),
            len: 0,
        }
    }

    #[inline]
    pub fn push_byte(&mut self, b: u8) {
        self.enc.encode(1, P_MORE);
        for i in (0..8).rev() {
            let bit = ((b >> i) & 1) as u32;
            let p = self.model.p();
            self.enc.encode(bit, quantize(p));
            self.model.update(bit);
        }
        self.len += 1;
    }

    pub fn push(&mut self, data: &[u8]) {
        self.crc.update(data);
        for &b in data {
            self.push_byte(b);
        }
    }

    /// Payload bytes that are final (may be written out while compressing).
    pub fn take_output(&mut self) -> Vec<u8> {
        self.enc.take_output()
    }

    /// End the payload; returns the remaining payload bytes followed by the trailer.
    pub fn finish(mut self) -> Vec<u8> {
        self.enc.encode(0, P_MORE);
        let mut out = self.enc.finish();
        out.extend_from_slice(&self.len.to_le_bytes());
        out.extend_from_slice(&self.crc.finish().to_le_bytes());
        out
    }
}

/// Compress `data` in memory.
pub fn compress(data: &[u8], opts: &Options) -> Vec<u8> {
    let (level, params) = opts.resolve().expect("invalid options");
    let header = Header {
        version: VERSION,
        model_id: MODEL_ID,
        level,
        params,
        content_length: Some(data.len() as u64),
    };
    let mut out = header.to_bytes();
    let mut c = Compressor::new(params);
    c.push(data);
    out.extend_from_slice(&c.take_output());
    out.extend_from_slice(&c.finish());
    out
}

/// Compress from a reader to a writer. `content_length`, if known, is recorded in the header
/// (it lets the decoder detect corruption early and bound its output); it must be exact.
/// Returns the number of input bytes.
pub fn compress_stream<R: Read, W: Write>(mut input: R, mut output: W, opts: &Options, content_length: Option<u64>) -> Result<u64, Error> {
    let (level, params) = opts.resolve()?;
    let header = Header {
        version: VERSION,
        model_id: MODEL_ID,
        level,
        params,
        content_length,
    };
    output.write_all(&header.to_bytes())?;
    let mut c = Compressor::new(params);
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = match input.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        };
        c.push(&buf[..n]);
        output.write_all(&c.take_output())?;
    }
    if let Some(n) = content_length {
        if n != c.len {
            return Err(Error::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("declared content length {} but read {} bytes", n, c.len),
            )));
        }
    }
    let n = c.len;
    output.write_all(&c.finish())?;
    output.flush()?;
    Ok(n)
}

/// A buffered pull source over any reader, remembering the first I/O error.
struct ReadSource<R: Read> {
    r: R,
    buf: Box<[u8]>,
    pos: usize,
    len: usize,
    err: Option<io::Error>,
    eof: bool,
}

impl<R: Read> ReadSource<R> {
    fn new(r: R) -> Self {
        ReadSource {
            r,
            buf: vec![0u8; 1 << 16].into_boxed_slice(),
            pos: 0,
            len: 0,
            err: None,
            eof: false,
        }
    }
    fn fill(&mut self) -> bool {
        if self.eof || self.err.is_some() {
            return false;
        }
        loop {
            match self.r.read(&mut self.buf) {
                Ok(0) => {
                    self.eof = true;
                    return false;
                }
                Ok(n) => {
                    self.pos = 0;
                    self.len = n;
                    return true;
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    self.err = Some(e);
                    return false;
                }
            }
        }
    }
    fn read_exact_bytes(&mut self, out: &mut [u8]) -> Result<(), Error> {
        for o in out.iter_mut() {
            match self.next_byte() {
                Some(b) => *o = b,
                None => return Err(self.take_err().map(Error::Io).unwrap_or(Error::Truncated)),
            }
        }
        Ok(())
    }
    fn take_err(&mut self) -> Option<io::Error> {
        self.err.take()
    }
}

impl<R: Read> ByteSource for ReadSource<R> {
    #[inline]
    fn next_byte(&mut self) -> Option<u8> {
        if self.pos == self.len && !self.fill() {
            return None;
        }
        let b = self.buf[self.pos];
        self.pos += 1;
        Some(b)
    }
}

/// Read only the header (cheap; does not run the model).
pub fn read_header<R: Read>(mut input: R) -> Result<Header, Error> {
    Header::read_from(&mut input)
}

/// Decompress from a reader to a writer, verifying length and checksum. Returns the number of
/// bytes written. On error, the writer may have received partial (unverified) output.
pub fn decompress_stream<R: Read, W: Write>(mut input: R, mut output: W, limits: &Limits) -> Result<u64, Error> {
    let header = Header::read_from(&mut input)?;
    let params = header.params;
    let need = params.memory_bytes();
    if need > limits.max_memory {
        return Err(Error::MemoryLimit {
            need,
            limit: limits.max_memory,
        });
    }
    let max_out = match header.content_length {
        Some(n) if n > limits.max_output => return Err(Error::OutputLimit(limits.max_output)),
        Some(n) => n,
        None => limits.max_output,
    };
    let mut model = Model::new(params);
    let mut dec = Decoder::new(ReadSource::new(input));
    let mut crc = crc32::Crc32::new();
    let mut out = Vec::with_capacity(1 << 16);
    let mut n: u64 = 0;
    loop {
        if dec.overrun() != 0 {
            return Err(Error::Truncated);
        }
        if dec.decode(P_MORE) == 0 {
            break;
        }
        if n >= max_out {
            return Err(if header.content_length.is_some() {
                Error::Corrupt("longer than the declared length")
            } else {
                Error::OutputLimit(max_out)
            });
        }
        let mut b = 0u32;
        for _ in 0..8 {
            let bit = dec.decode(quantize(model.p()));
            model.update(bit);
            b = (b << 1) | bit;
        }
        out.push(b as u8);
        n += 1;
        if out.len() >= 1 << 16 {
            crc.update(&out);
            output.write_all(&out)?;
            out.clear();
        }
    }
    if dec.overrun() != 0 {
        return Err(Error::Truncated);
    }
    crc.update(&out);
    output.write_all(&out)?;
    output.flush()?;
    let mut src = dec.into_source();
    if let Some(e) = src.take_err() {
        return Err(Error::Io(e));
    }
    let mut trailer = [0u8; TRAILER_LEN];
    src.read_exact_bytes(&mut trailer)?;
    let len = u64::from_le_bytes(trailer[0..8].try_into().unwrap());
    let sum = u32::from_le_bytes(trailer[8..12].try_into().unwrap());
    if len != n || header.content_length.is_some_and(|h| h != n) {
        return Err(Error::Corrupt("length mismatch"));
    }
    if sum != crc.finish() {
        return Err(Error::Corrupt("checksum mismatch"));
    }
    if src.next_byte().is_some() {
        return Err(Error::TrailingData);
    }
    if let Some(e) = src.take_err() {
        return Err(Error::Io(e));
    }
    Ok(n)
}

/// Decompress an in-memory stream with default limits.
pub fn decompress(data: &[u8]) -> Result<Vec<u8>, Error> {
    decompress_with_limits(data, &Limits::default())
}

pub fn decompress_with_limits(data: &[u8], limits: &Limits) -> Result<Vec<u8>, Error> {
    let mut out = Vec::new();
    decompress_stream(data, &mut out, limits)?;
    Ok(out)
}

/// The model's ideal code length for `data` in bits (no coder): sum of -log2 p(bit). Useful for
/// research comparisons; matches the `whole-stream` figure of strong.rs times 8 * len.
pub fn ideal_bits(data: &[u8], params: Params) -> f64 {
    let mut m = Model::new(params);
    let mut tot = 0.0;
    for &b in data {
        for i in (0..8).rev() {
            let bit = ((b >> i) & 1) as u32;
            let p = m.p();
            tot -= (if bit == 1 { p } else { 1.0 - p }).log2();
            m.update(bit);
        }
    }
    tot
}

/// FNV-1a over the IEEE bits of every probability the model emits for `data`, plus the total
/// cost (bits). Two predictors are bit-identical on `data` iff these agree. Matches the `phash`
/// line of the research engine patched for parity testing.
pub fn probability_trace_hash(data: &[u8], params: Params) -> (u64, f64) {
    let mut m = Model::new(params);
    let mut h: u64 = 0xcbf29ce484222325;
    let mut tot = 0.0;
    for &b in data {
        for i in (0..8).rev() {
            let bit = ((b >> i) & 1) as u32;
            let p = m.p();
            for byte in p.to_bits().to_le_bytes() {
                h = (h ^ byte as u64).wrapping_mul(0x100000001b3);
            }
            tot -= (if bit == 1 { p } else { 1.0 - p }).log2();
            m.update(bit);
        }
    }
    (h, tot)
}

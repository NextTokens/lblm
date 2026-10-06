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
//! let packed = blmz::compress(&data, &blmz::Options::default()).unwrap();
//! let back = blmz::decompress(&packed).unwrap();
//! assert_eq!(back, data);
//! ```
//!
//! Compression and decompression are symmetric and slow (tens of KB/s): every bit of the input
//! runs the full model, so on untrusted input the output limit is also the CPU-time limit — see
//! [`Limits`]. Roadmap: `ROADMAP.md`; byte format: `FORMAT.md`.

// The model mirrors blmrs/src/bin/strong.rs line by line (index loops included) so parity
// reviews can diff the two; keep that shape.
#![allow(clippy::needless_range_loop)]

// x87 arithmetic (32-bit x86 without SSE2) keeps 80-bit intermediates: probabilities would differ
// from every other platform and streams would not decode across machines.
#[cfg(all(target_arch = "x86", not(target_feature = "sse2")))]
compile_error!("blmz requires SSE2 floating point on 32-bit x86 (x87 arithmetic is not reproducible)");

#[doc(hidden)]
pub mod coder;
pub mod crc32;
pub mod format;
#[doc(hidden)]
pub mod math;
pub mod model;

use coder::{quantize, ByteSource, Decoder, Encoder};
use format::{Header, MODEL_ID, P_MORE, TRAILER_LEN, VERSION};
pub use model::{AllocError, Model, Params};
use std::io::{self, BufRead, BufReader, Read, Write};

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
        6 => [21, 21, 21, 19, 21, 21, 28],
        7 => [22, 22, 22, 20, 22, 22, 29],
        8 => [23, 22, 22, 22, 23, 23, 30],
        9 => [24, 24, 24, 22, 24, 24, 31],
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

/// Compression options.
#[derive(Clone, Debug)]
#[non_exhaustive]
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
    pub fn with_params(params: Params) -> Self {
        Options {
            level: 0,
            params: Some(params),
        }
    }
    /// The (level, params) this resolves to, or BadLevel / BadParams.
    pub fn resolve(&self) -> Result<(u8, Params), Error> {
        match self.params {
            Some(p) => {
                p.validate().map_err(Error::BadParams)?;
                Ok((0, p))
            }
            None => level_params(self.level).map(|p| (self.level, p)).ok_or(Error::BadLevel(self.level)),
        }
    }
}

/// Resource limits for DEcompression (the input may be hostile).
///
/// * `max_memory` bounds tables + history (see [`Params::total_memory`]); checked from the header
///   before anything is allocated. Allocation failure below the limit is `Error::OutOfMemory`.
/// * `max_output` bounds the decoded size. Decoding costs about as much CPU per output byte as
///   compression, and a small stream can legitimately expand by 10^5 or more (e.g. long runs of
///   zeros), so for untrusted input this is also the CPU-time limit. A declared content length in
///   the header is enforced too, but the sender chooses it.
///
/// The defaults are safe for untrusted input of ordinary size: enough memory for any level, and
/// at most 1 GiB of output (hours of CPU at worst). For larger trusted archives use
/// [`Limits::unbounded`] or raise `max_output`; the CLI decodes without an output limit unless
/// `--max-output` is given.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Limits {
    pub max_memory: u64,
    pub max_output: u64,
}

impl Default for Limits {
    fn default() -> Self {
        // level 9 with an unknown length: ~1.7 GiB of tables + a 2 GiB window
        Limits {
            max_memory: 4 << 30,
            max_output: 1 << 30,
        }
    }
}

impl Limits {
    /// No output limit (trusted input only); memory as the default.
    pub fn unbounded() -> Self {
        Limits::default().max_output(u64::MAX)
    }
    pub fn max_memory(mut self, bytes: u64) -> Self {
        self.max_memory = bytes;
        self
    }
    pub fn max_output(mut self, bytes: u64) -> Self {
        self.max_output = bytes;
        self
    }
}

#[derive(Debug)]
#[non_exhaustive]
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
    /// The stream needs more memory than `Limits::max_memory`.
    MemoryLimit {
        need: u64,
        limit: u64,
    },
    /// Allocating the model failed (address space or allocator refused).
    OutOfMemory {
        need: u64,
    },
    OutputLimit(u64),
    /// The payload or trailer ended early.
    Truncated,
    /// Decoded data disagrees with the stored length or checksum.
    Corrupt(&'static str),
    /// Bytes follow the end of the stream.
    TrailingData,
    /// The input did not have the declared length (e.g. a file changed while being compressed).
    LengthMismatch {
        declared: u64,
        actual: u64,
    },
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
            Error::MemoryLimit { need, limit } => write!(
                f,
                "stream needs {} MiB of memory, limit is {} MiB",
                need.div_ceil(1 << 20),
                limit >> 20
            ),
            Error::OutOfMemory { need } => write!(f, "could not allocate {} MiB for the model", need.div_ceil(1 << 20)),
            Error::OutputLimit(n) => write!(f, "output exceeds the limit of {} bytes", n),
            Error::Truncated => write!(f, "stream is truncated"),
            Error::Corrupt(s) => write!(f, "stream is corrupt: {}", s),
            Error::TrailingData => write!(f, "unexpected data after the end of the stream"),
            Error::LengthMismatch { declared, actual } => {
                write!(f, "input length changed: expected {} bytes, read {}", declared, actual)
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

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

fn alloc_err(e: AllocError) -> Error {
    Error::OutOfMemory { need: e.bytes }
}

/// Codes bytes into a payload. Usually driven through `compress` / `compress_stream`; use it
/// directly to push data incrementally into your own sink.
pub struct Compressor {
    model: Model,
    enc: Encoder,
    crc: crc32::Crc32,
    len: u64,
}

impl Compressor {
    /// `expected_len` (if known) pre-sizes the history buffer; it is not written anywhere.
    pub fn new(params: Params, expected_len: Option<u64>) -> Result<Self, Error> {
        params.validate().map_err(Error::BadParams)?;
        Ok(Compressor {
            model: Model::try_new(params, expected_len).map_err(alloc_err)?,
            enc: Encoder::new(),
            crc: crc32::Crc32::new(),
            len: 0,
        })
    }

    #[inline]
    fn push_byte(&mut self, b: u8) {
        self.enc.encode(1, P_MORE);
        for i in (0..8).rev() {
            let bit = ((b >> i) & 1) as u32;
            let p = self.model.p();
            self.enc.encode(bit, quantize(p));
            self.model.update(bit);
        }
        self.len += 1;
    }

    /// Code `data` (checksummed and counted).
    pub fn push(&mut self, data: &[u8]) -> Result<(), Error> {
        self.crc.update(data);
        for &b in data {
            self.push_byte(b);
        }
        if self.model.alloc_failed() {
            return Err(Error::OutOfMemory {
                need: self.model.params().total_memory(Some(self.len)),
            });
        }
        Ok(())
    }

    /// Bytes coded so far.
    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
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

fn header_for(level: u8, params: Params, content_length: Option<u64>) -> Header {
    Header {
        version: VERSION,
        model_id: MODEL_ID,
        level,
        params,
        content_length,
    }
}

/// Compress `data` in memory. Fails only on invalid options or allocation failure.
pub fn compress(data: &[u8], opts: &Options) -> Result<Vec<u8>, Error> {
    let (level, params) = opts.resolve()?;
    let mut out = header_for(level, params, Some(data.len() as u64)).to_bytes();
    let mut c = Compressor::new(params, Some(data.len() as u64))?;
    c.push(data)?;
    out.extend_from_slice(&c.take_output());
    out.extend_from_slice(&c.finish());
    Ok(out)
}

/// Compress from a reader to a writer. `content_length`, if known, is recorded in the header
/// (it lets the decoder detect corruption early and bound its output); if the reader then yields
/// a different number of bytes the result is `Error::LengthMismatch` (the output is unusable).
/// Returns the number of input bytes.
pub fn compress_stream<R: Read, W: Write>(mut input: R, mut output: W, opts: &Options, content_length: Option<u64>) -> Result<u64, Error> {
    let (level, params) = opts.resolve()?;
    let mut c = Compressor::new(params, content_length)?;
    output.write_all(&header_for(level, params, content_length).to_bytes())?;
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = match input.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        };
        if let Some(declared) = content_length {
            if c.len() + n as u64 > declared {
                return Err(Error::LengthMismatch {
                    declared,
                    actual: c.len() + n as u64,
                });
            }
        }
        c.push(&buf[..n])?;
        output.write_all(&c.take_output())?;
    }
    if let Some(declared) = content_length {
        if declared != c.len() {
            return Err(Error::LengthMismatch { declared, actual: c.len() });
        }
    }
    let n = c.len();
    output.write_all(&c.finish())?;
    output.flush()?;
    Ok(n)
}

/// Pull bytes one at a time from a BufRead without reading past what is consumed, remembering
/// the first I/O error (so it can be reported instead of a generic truncation).
struct BufSource<'a, R: BufRead> {
    r: &'a mut R,
    err: Option<io::Error>,
}

impl<R: BufRead> ByteSource for BufSource<'_, R> {
    #[inline]
    fn next_byte(&mut self) -> Option<u8> {
        if self.err.is_some() {
            return None;
        }
        loop {
            match self.r.fill_buf() {
                Ok([]) => return None,
                Ok(buf) => {
                    let b = buf[0];
                    self.r.consume(1);
                    return Some(b);
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    self.err = Some(e);
                    return None;
                }
            }
        }
    }
}

/// Read only the header (cheap; does not run the model).
pub fn read_header<R: Read>(mut input: R) -> Result<Header, Error> {
    Header::read_from(&mut input)
}

/// Check a header against `limits` before any allocation: returns the memory it needs.
pub fn check_limits(header: &Header, limits: &Limits) -> Result<u64, Error> {
    let need = header.params.total_memory(header.content_length);
    if need > limits.max_memory {
        return Err(Error::MemoryLimit {
            need,
            limit: limits.max_memory,
        });
    }
    if let Some(n) = header.content_length {
        if n > limits.max_output {
            return Err(Error::OutputLimit(limits.max_output));
        }
    }
    Ok(need)
}

/// Decompress exactly one stream from `input`, leaving the reader positioned right after it
/// (so it works on pipes and sockets that stay open, and on concatenated streams). Verifies
/// length and checksum. Returns the number of bytes written. On error, `output` may hold partial
/// (unverified) data.
pub fn decompress_one<R: BufRead, W: Write>(input: &mut R, mut output: W, limits: &Limits) -> Result<u64, Error> {
    let header = Header::read_from(input)?;
    check_limits(&header, limits)?;
    let max_out = header.content_length.unwrap_or(limits.max_output);
    let mut model = Model::try_new(header.params, header.content_length).map_err(alloc_err)?;
    let mut dec = Decoder::new(BufSource { r: input, err: None });
    let mut crc = crc32::Crc32::new();
    let mut out = Vec::with_capacity(1 << 16);
    let mut n: u64 = 0;
    fn fail<R: BufRead>(dec: &mut Decoder<BufSource<R>>) -> Error {
        match dec.source_mut().err.take() {
            Some(e) => Error::Io(e),
            None => Error::Truncated,
        }
    }
    loop {
        if dec.overrun() != 0 {
            return Err(fail(&mut dec));
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
            if model.alloc_failed() {
                return Err(Error::OutOfMemory {
                    need: header.params.total_memory(Some(n)),
                });
            }
            crc.update(&out);
            output.write_all(&out)?;
            out.clear();
        }
    }
    if dec.overrun() != 0 {
        return Err(fail(&mut dec));
    }
    if model.alloc_failed() {
        return Err(Error::OutOfMemory {
            need: header.params.total_memory(Some(n)),
        });
    }
    crc.update(&out);
    output.write_all(&out)?;
    output.flush()?;
    let mut src = dec.into_source();
    let mut trailer = [0u8; TRAILER_LEN];
    for t in trailer.iter_mut() {
        *t = match src.next_byte() {
            Some(b) => b,
            None => return Err(src.err.take().map(Error::Io).unwrap_or(Error::Truncated)),
        };
    }
    let len = u64::from_le_bytes(trailer[0..8].try_into().unwrap());
    let sum = u32::from_le_bytes(trailer[8..12].try_into().unwrap());
    if len != n || header.content_length.is_some_and(|h| h != n) {
        return Err(Error::Corrupt("length mismatch"));
    }
    if sum != crc.finish() {
        return Err(Error::Corrupt("checksum mismatch"));
    }
    Ok(n)
}

/// Decompress one stream from a reader to a writer and require end of input after it
/// (`Error::TrailingData` otherwise). Returns the number of bytes written. On error, the writer
/// may have received partial (unverified) output.
pub fn decompress_stream<R: Read, W: Write>(input: R, output: W, limits: &Limits) -> Result<u64, Error> {
    let mut input = BufReader::new(input);
    let n = decompress_one(&mut input, output, limits)?;
    if !input.fill_buf()?.is_empty() {
        return Err(Error::TrailingData);
    }
    Ok(n)
}

/// Decompress an in-memory stream with default limits.
pub fn decompress(data: &[u8]) -> Result<Vec<u8>, Error> {
    decompress_with_limits(data, &Limits::default())
}

pub fn decompress_with_limits(mut data: &[u8], limits: &Limits) -> Result<Vec<u8>, Error> {
    let mut out = Vec::new();
    decompress_one(&mut data, &mut out, limits)?;
    if !data.is_empty() {
        return Err(Error::TrailingData);
    }
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
/// line of the research engine patched for parity testing (tools/parity.sh).
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

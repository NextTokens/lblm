//! Round-trip, robustness and format-stability tests. Level 1 (20 MiB of tables) keeps them fast.

use blmz::{compress, compress_stream, decompress, decompress_stream, decompress_with_limits, Error, Limits, Options};

fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *s >> 33
}

/// Deterministic English-like text (words from a small vocabulary, punctuation, newlines).
fn text(n: usize, seed: u64) -> Vec<u8> {
    const WORDS: &[&str] = &[
        "the",
        "of",
        "and",
        "compression",
        "model",
        "bit",
        "context",
        "mixing",
        "predicts",
        "next",
        "a",
        "is",
        "memory",
        "which",
        "Wikipedia",
        "data",
        "stream",
        "with",
        "for",
        "in",
        "London",
        "1879",
        "was",
        "it",
    ];
    let mut s = seed;
    let mut out = Vec::with_capacity(n + 16);
    while out.len() < n {
        let w = WORDS[(lcg(&mut s) % WORDS.len() as u64) as usize];
        out.extend_from_slice(w.as_bytes());
        out.push(match lcg(&mut s) % 13 {
            0 => b'.',
            1 => b',',
            2 => b'\n',
            _ => b' ',
        });
    }
    out.truncate(n);
    out
}

fn random(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    (0..n).map(|_| lcg(&mut s) as u8).collect()
}

/// Little-endian records with slowly varying fields (database/binary-like).
fn records(n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    let mut i = 0u32;
    while out.len() < n {
        out.extend_from_slice(&i.to_le_bytes());
        out.extend_from_slice(&(i.wrapping_mul(7) % 1000).to_le_bytes());
        out.extend_from_slice(b"rec\0");
        i += 1;
    }
    out.truncate(n);
    out
}

fn l1() -> Options {
    Options::level(1)
}

fn corpus() -> Vec<(&'static str, Vec<u8>)> {
    let mut all: Vec<u8> = (0..=255u8).collect();
    all.extend((0..=255u8).rev());
    vec![
        ("empty", vec![]),
        ("one-byte", vec![b'x']),
        ("one-zero-byte", vec![0]),
        ("one-ff-byte", vec![0xFF]),
        ("all-bytes", all),
        ("zeros", vec![0u8; 20_000]),
        ("ones", vec![0xFFu8; 20_000]),
        ("text", text(40_000, 1)),
        ("random", random(20_000, 2)),
        ("records", records(30_000)),
        ("period-7", (0..25_000u32).map(|i| b"abcdefg"[(i % 7) as usize]).collect()),
        ("mixed", {
            let mut v = text(10_000, 3);
            v.extend(random(5_000, 4));
            v.extend(text(10_000, 3));
            v
        }),
    ]
}

#[test]
fn roundtrip_corpus() {
    for (name, data) in corpus() {
        let packed = compress(&data, &l1()).unwrap();
        let back = decompress(&packed).unwrap_or_else(|e| panic!("{}: {}", name, e));
        assert_eq!(back, data, "{}", name);
    }
}

#[test]
fn compresses_what_it_should() {
    let t = text(40_000, 1);
    let p = compress(&t, &l1()).unwrap();
    assert!(p.len() * 4 < t.len(), "text should compress >4x, got {} -> {}", t.len(), p.len());
    let z = vec![0u8; 20_000];
    assert!(compress(&z, &l1()).unwrap().len() < 120, "zeros should be nearly free");
    let r = random(20_000, 2);
    let pr = compress(&r, &l1()).unwrap();
    // incompressible data: bounded expansion (model overhead on random bytes is small)
    assert!(pr.len() < r.len() + r.len() / 50 + 64, "random expanded too much: {}", pr.len());
}

#[test]
fn every_level_roundtrips() {
    let data = text(3_000, 9);
    for level in 1..=9 {
        let p = compress(&data, &Options::level(level)).unwrap();
        assert_eq!(decompress(&p).unwrap(), data, "level {}", level);
        assert_eq!(blmz::read_header(&p[..]).unwrap().level, level);
    }
}

#[test]
fn streaming_matches_one_shot_and_handles_unknown_length() {
    let data = text(30_000, 5);
    let one = compress(&data, &l1()).unwrap();
    let mut streamed = Vec::new();
    // tiny reads exercise chunk boundaries
    struct Trickle<'a>(&'a [u8]);
    impl std::io::Read for Trickle<'_> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let n = buf.len().min(self.0.len()).min(7);
            buf[..n].copy_from_slice(&self.0[..n]);
            self.0 = &self.0[n..];
            Ok(n)
        }
    }
    compress_stream(Trickle(&data), &mut streamed, &l1(), Some(data.len() as u64)).unwrap();
    assert_eq!(streamed, one, "streaming with a known length must equal the one-shot output");

    let mut unknown = Vec::new();
    compress_stream(&data[..], &mut unknown, &l1(), None).unwrap();
    assert!(blmz::read_header(&unknown[..]).unwrap().content_length.is_none());
    let mut back = Vec::new();
    decompress_stream(Trickle(&unknown), &mut back, &Limits::default()).unwrap();
    assert_eq!(back, data);
}

#[test]
fn wrong_declared_length_is_an_error() {
    let data = text(1000, 1);
    let mut out = Vec::new();
    assert!(compress_stream(&data[..], &mut out, &l1(), Some(999)).is_err());
}

#[test]
fn truncation_never_panics_and_always_errors() {
    let data = text(5_000, 11);
    let p = compress(&data, &l1()).unwrap();
    for cut in (0..p.len()).step_by(37).chain([p.len() - 1, p.len() - 12, p.len() - 13]) {
        let r = decompress(&p[..cut]);
        assert!(r.is_err(), "truncated at {} of {} decoded successfully", cut, p.len());
    }
}

#[test]
fn bit_flips_are_detected() {
    let data = text(5_000, 12);
    let p = compress(&data, &l1()).unwrap();
    let mut s = 99u64;
    for _ in 0..200 {
        let mut q = p.clone();
        let at = (lcg(&mut s) as usize) % q.len();
        q[at] ^= 1 << (lcg(&mut s) % 8);
        if let Ok(back) = decompress(&q) {
            panic!("flip at {} accepted ({} bytes out, equal = {})", at, back.len(), back == data);
        }
    }
}

#[test]
fn trailing_garbage_and_concatenation_rejected() {
    let p = compress(b"hello hello hello", &l1()).unwrap();
    let mut q = p.clone();
    q.push(0);
    assert!(matches!(decompress(&q), Err(Error::TrailingData)));
    let mut cat = p.clone();
    cat.extend_from_slice(&p);
    assert!(matches!(decompress(&cat), Err(Error::TrailingData)));
}

#[test]
fn header_validation() {
    let p = compress(b"abc", &l1()).unwrap();
    assert!(matches!(decompress(b"not a blz file at all"), Err(Error::NotBlz)));
    assert!(matches!(decompress(&[]), Err(Error::NotBlz)));
    let mut v = p.clone();
    v[4] = 99;
    assert!(matches!(decompress(&v), Err(Error::UnsupportedVersion(99))));
    let mut m = p.clone();
    m[8] = 31; // obits out of range, but header CRC now wrong too
    assert!(matches!(decompress(&m), Err(Error::CorruptHeader(_))));
}

#[test]
fn hostile_header_params_are_refused_before_allocating() {
    // a well-formed header (valid CRC) asking for 2^30-slot tables
    let mut h = blmz::format::Header {
        version: blmz::format::VERSION,
        model_id: blmz::format::MODEL_ID,
        level: 0,
        params: blmz::level_params(1).unwrap(),
        content_length: Some(10),
    };
    h.params.obits = 30;
    h.params.selvbits = 30;
    let bytes = h.to_bytes();
    let r = decompress_with_limits(
        &bytes,
        &Limits {
            max_memory: 1 << 30,
            max_output: u64::MAX,
        },
    );
    assert!(matches!(r, Err(Error::MemoryLimit { .. })), "{:?}", r.err());
    h.params.obits = 31;
    assert!(matches!(decompress(&h.to_bytes()), Err(Error::BadParams(_))));
}

#[test]
fn output_limit() {
    let data = vec![b'a'; 5000];
    let p = compress(&data, &l1()).unwrap();
    let r = decompress_with_limits(
        &p,
        &Limits {
            max_output: 100,
            ..Default::default()
        },
    );
    assert!(matches!(r, Err(Error::OutputLimit(100))));
}

/// Format stability + cross-platform determinism: these exact bytes must come out on every OS and
/// CPU. If this fails after an intentional model/format change, bump MODEL_ID/VERSION and re-pin;
/// if it fails on one platform only, decoding is not portable there — a release blocker.
#[test]
fn golden_streams() {
    let cases: [(&str, Vec<u8>, u8); 4] = [
        ("text-l1", text(20_000, 21), 1),
        ("records-l1", records(12_000), 1),
        ("random-l1", random(4_000, 22), 1),
        ("text-l3", text(6_000, 23), 3),
    ];
    let mut got = Vec::new();
    for (name, data, level) in cases.iter() {
        let p = compress(data, &Options::level(*level)).unwrap();
        got.push(format!("{}:{}:{:08x}", name, p.len(), blmz::crc32::crc32(&p)));
        assert_eq!(&decompress(&p).unwrap(), data);
    }
    let got = got.join(" ");
    // std-math (parity-only) builds use the platform libm: not portable, so not pinned
    if cfg!(not(feature = "std-math")) {
        assert_eq!(got, GOLDEN, "golden streams changed:\n got  {}\n want {}", got, GOLDEN);
    }
}

const GOLDEN: &str = "text-l1:2824:9cb32860 records-l1:1534:4e936fe3 random-l1:4056:b15bb47c text-l3:1018:d705d7df";

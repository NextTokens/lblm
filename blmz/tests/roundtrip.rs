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
    let r = decompress_with_limits(&bytes, &Limits::default().max_memory(1 << 30));
    // 64-bit: over the memory limit; 32-bit: does not even fit the address space
    assert!(matches!(r, Err(Error::MemoryLimit { .. } | Error::BadParams(_))), "{:?}", r.err());
    h.params.obits = 31;
    assert!(matches!(decompress(&h.to_bytes()), Err(Error::BadParams(_))));
}

#[test]
fn output_limit() {
    let data = vec![b'a'; 5000];
    let p = compress(&data, &l1()).unwrap();
    let r = decompress_with_limits(&p, &Limits::default().max_output(100));
    assert!(matches!(r, Err(Error::OutputLimit(100))));
}

fn tiny(window_log: u8) -> blmz::Params {
    blmz::Params {
        obits: 12,
        hbits: 12,
        mbits: 16,
        sbits: 10,
        selvbits: 12,
        selubits: 12,
        window_log,
    }
}

/// A header with no declared length followed by zero bytes decodes to long runs of 0xFF without
/// ever reaching end of stream (the "bomb" shape): max_output must stop it promptly.
#[test]
fn zero_payload_is_bounded_by_max_output() {
    let h = blmz::format::Header {
        version: blmz::format::VERSION,
        model_id: blmz::format::MODEL_ID,
        level: 0,
        params: tiny(16),
        content_length: None,
    };
    let mut bomb = h.to_bytes();
    bomb.extend_from_slice(&[0u8; 64]);
    let t = std::time::Instant::now();
    let r = decompress_with_limits(&bomb, &Limits::default().max_output(20_000));
    assert!(matches!(r, Err(Error::OutputLimit(20_000) | Error::Truncated)), "{:?}", r.err());
    assert!(t.elapsed().as_secs() < 30);
}

/// The history window counts against max_memory: tiny tables but a 4 GiB window and no length.
#[test]
fn memory_limit_counts_the_history_window() {
    let p = tiny(32);
    assert!(p.memory_bytes() < 64 << 20);
    assert_eq!(p.total_memory(None), p.memory_bytes() + (1 << 32));
    assert_eq!(p.total_memory(Some(1000)), p.memory_bytes() + 1000);
    let h = blmz::format::Header {
        version: blmz::format::VERSION,
        model_id: blmz::format::MODEL_ID,
        level: 0,
        params: p,
        content_length: None,
    };
    let r = decompress_with_limits(&h.to_bytes(), &Limits::default().max_memory(1 << 30));
    assert!(matches!(r, Err(Error::MemoryLimit { .. } | Error::BadParams(_))), "{:?}", r.err());
}

/// Inputs longer than the history window: a near-duplicate exactly one window (2^16) back must
/// neither break the round trip nor lock the match model onto its own freshly written byte.
#[test]
fn window_wrap_with_match_at_exactly_the_window_distance() {
    // random bytes: the ONLY repeats are the copy exactly one window back (text would give the
    // match model nearer candidates and hide the defect)
    let a = random(1 << 16, 77);
    let mut data = a.clone();
    let mut b = a.clone();
    b.remove(30_000); // copy with one byte deleted: the match drops out of sync there
    data.extend_from_slice(&b);
    data.extend_from_slice(&text(20_000, 78));
    let small = compress(&data, &Options::with_params(tiny(16))).unwrap();
    let big = compress(&data, &Options::with_params(tiny(17))).unwrap();
    assert_eq!(decompress(&small).unwrap(), data);
    assert_eq!(decompress(&big).unwrap(), data);
    // the 2^16 window sees the copy at distance 2^16 - 0 / -1; it must compress about as well as
    // a window that holds everything (the defect cost +44% here)
    assert!(
        small.len() * 100 < big.len() * 105,
        "window 16: {} bytes, window 17: {} bytes",
        small.len(),
        big.len()
    );
}

#[test]
fn decompress_one_leaves_the_reader_after_the_stream() {
    let a = compress(b"first stream", &l1()).unwrap();
    let b = compress(b"second", &l1()).unwrap();
    let mut all = a.clone();
    all.extend_from_slice(&b);
    all.extend_from_slice(b"XYZ");
    let mut r = &all[..];
    let mut out = Vec::new();
    blmz::decompress_one(&mut r, &mut out, &Limits::default()).unwrap();
    assert_eq!(out, b"first stream");
    out.clear();
    blmz::decompress_one(&mut r, &mut out, &Limits::default()).unwrap();
    assert_eq!(out, b"second");
    assert_eq!(r, b"XYZ");
}

#[test]
fn io_errors_are_reported_as_io_errors() {
    struct Failing<'a>(&'a [u8]);
    impl std::io::Read for Failing<'_> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.0.is_empty() {
                return Err(std::io::Error::other("disk on fire"));
            }
            let n = buf.len().min(self.0.len());
            buf[..n].copy_from_slice(&self.0[..n]);
            self.0 = &self.0[n..];
            Ok(n)
        }
    }
    let p = compress(&text(5_000, 5), &l1()).unwrap();
    let r = decompress_stream(Failing(&p[..p.len() / 2]), std::io::sink(), &Limits::default());
    assert!(
        matches!(r, Err(Error::Io(ref e)) if e.to_string().contains("disk on fire")),
        "{:?}",
        r.err()
    );
}

#[test]
fn model_p_is_idempotent_and_update_is_safe_alone() {
    let mut a = blmz::Model::new(tiny(16));
    let mut b = blmz::Model::new(tiny(16));
    for &byte in b"hello world, hello model" {
        for i in (0..8).rev() {
            let bit = ((byte >> i) & 1) as u32;
            let p1 = a.p();
            assert_eq!(p1.to_bits(), a.p().to_bits(), "second p() must not re-evaluate");
            a.update(bit);
            b.update(bit); // without p(): must behave as if p() had been called
        }
    }
    assert_eq!(a.p().to_bits(), b.p().to_bits());
}

/// Streams written by earlier builds (tests/fixtures) must decode forever, bit-exactly.
/// Never regenerate or edit these files: a model or format change adds NEW fixtures.
#[test]
#[cfg_attr(blmz_std_math, ignore)]
fn fixtures_decode_forever() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    for (name, content) in fixture_contents() {
        let packed = std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("{}: {}", name, e));
        let back = decompress(&packed).unwrap_or_else(|e| panic!("{}: {}", name, e));
        assert_eq!(back, content, "{} decoded to different bytes", name);
    }
}

fn fixture_contents() -> Vec<(&'static str, Vec<u8>)> {
    let mut wrap = text(1 << 16, 77);
    wrap[100] = b'#';
    let mut b = wrap.clone();
    b.remove(30_000);
    wrap.extend_from_slice(&b);
    vec![
        ("v1-m1-text-l1.blz", text(20_000, 31)),
        ("v1-m1-records-l1-streamed.blz", records(8_000)),
        ("v1-m1-empty-l1.blz", vec![]),
        ("v1-m1-text-l3.blz", text(5_000, 32)),
        ("v1-m1-wrap-w16.blz", wrap),
    ]
}

/// Run once by hand (cargo test --release -- --ignored write_fixtures) when ADDING fixtures for a
/// new model id or format version; never to overwrite existing ones.
#[test]
#[ignore]
fn write_fixtures() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    std::fs::create_dir_all(&dir).unwrap();
    for (name, content) in fixture_contents() {
        let path = dir.join(name);
        if path.exists() {
            continue; // never overwrite
        }
        let packed = match name {
            "v1-m1-records-l1-streamed.blz" => {
                let mut v = Vec::new();
                compress_stream(&content[..], &mut v, &l1(), None).unwrap();
                v
            }
            "v1-m1-text-l3.blz" => compress(&content, &Options::level(3)).unwrap(),
            "v1-m1-wrap-w16.blz" => compress(&content, &Options::with_params(tiny(16))).unwrap(),
            _ => compress(&content, &l1()).unwrap(),
        };
        std::fs::write(path, packed).unwrap();
    }
}

/// Encoder stability + cross-platform determinism: these exact bytes must come out on every OS and
/// CPU. If this fails on one platform only, decoding is not portable there: a release blocker.
/// If it fails everywhere after an intentional model change, the change needs a NEW model id (old
/// streams must keep decoding, see fixtures_decode_forever); only then pin the new values.
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
    // blmz_std_math (parity-only) builds use the platform libm: not portable, so not pinned
    if cfg!(not(blmz_std_math)) {
        assert_eq!(got, GOLDEN, "golden streams changed:\n got  {}\n want {}", got, GOLDEN);
    }
}

const GOLDEN: &str = "text-l1:2824:9cb32860 records-l1:1534:4e936fe3 random-l1:4056:b15bb47c text-l3:1018:d705d7df";

/// Every hostile-header shape found in review (tables too big for 32-bit, or over the default
/// memory limit) must come back as an error, never a panic or an allocation abort. The payload
/// is empty, so nothing is touched beyond (lazily zeroed) allocation.
#[test]
fn hostile_headers_never_panic() {
    let base = tiny(16);
    let mut cases = Vec::new();
    for (field, v) in [
        ("obits", 27u8),
        ("hbits", 29),
        ("mbits", 29),
        ("sbits", 25),
        ("sbits", 26),
        ("selvbits", 27),
        ("selubits", 28),
    ] {
        let mut p = base;
        match field {
            "obits" => p.obits = v,
            "hbits" => p.hbits = v,
            "mbits" => p.mbits = v,
            "sbits" => p.sbits = v,
            "selvbits" => p.selvbits = v,
            _ => p.selubits = v,
        }
        cases.push(p);
    }
    for p in cases {
        for len in [None, Some(10)] {
            let h = blmz::format::Header {
                version: blmz::format::VERSION,
                model_id: blmz::format::MODEL_ID,
                level: 0,
                params: p,
                content_length: len,
            };
            let r = std::panic::catch_unwind(|| decompress(&h.to_bytes()));
            assert!(matches!(r, Ok(Err(_))), "{:?} len {:?} -> {:?}", p, len, r.map(|x| x.err()));
        }
    }
}

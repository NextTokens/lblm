# blmz — the LBLM compressor, packaged

`blmz` turns the research engine `blmrs/src/bin/strong.rs` (the bit-native context-mixing predictor
behind the project's enwik8 numbers) into a real codec. The research engine only measures cross-entropy;
`blmz` writes and reads actual files.

| | research engine (`strong.rs`) | `blmz` |
|---|---|---|
| output | a bits/bit number | a `.blz` file + decoder |
| predictor | ~30 env flags, adopted defaults | the adopted defaults, frozen (model id 1) |
| parity | — | **bit-identical** per-bit probabilities (`tools/parity.sh`) |
| math | platform libm (`ln`/`exp`/`log2`) | frozen portable implementation → decodes on any platform |
| config | env vars + argv, silent fallbacks | table sizes in a CRC-checked header; validated |
| memory | ~10× the input + ~1.5 GB of tables | levels 1–9: 36 MiB … 3.7 GiB worst case (tables + history window) |
| untrusted input | panics/aborts on bad params | typed errors; memory and output limits checked before allocating |
| speed (same tables) | 1× | ~1.6–1.8× (same probabilities; cache-friendly layout, prefetching) |

## Results

Full Silesia corpus (211,938,580 bytes), level 6 (decode verification status: `ROADMAP.md`):

| | blmz -6 | xz -9e | bzip2 -9 | gzip -9 |
|---|---|---|---|---|
| total bytes | **42,037,924** | 48,456,004 | 54,506,769 | 67,631,990 |
| vs xz | **−13.2 %** | — | +12.5 % | +39.6 % |

It beats xz -9e on 11 of 12 files (text −22…−29 %, x-ray −18 %, osdb −20 %) and loses on `sao`
(+1.6 %, binary star records). Speed: ~54 KB/s per core in both directions (contended 4-core
Xeon), i.e. roughly 1 hour per GB. Per-file numbers: `ROADMAP.md`.

## Use

```sh
cd blmz && cargo build --release
./target/release/blmz -6 file              # -> file.blz (keeps file, copies its mode and mtime)
./target/release/blmz -d file.blz          # -> file (concatenated streams decode in sequence)
./target/release/blmz -t file.blz          # verify (decode + length + CRC-32)
./target/release/blmz -l file.blz          # header info, memory needed to decode
tar c dir | ./target/release/blmz > dir.tar.blz     # streaming
./target/release/blmz -d --max-output 1G --memlimit 512 untrusted.blz
./target/release/blmz bench file           # ratio + speed + round-trip check, in memory
```

As a library:

```rust
let packed = blmz::compress(&data, &blmz::Options::level(6))?;
let data2 = blmz::decompress(&packed)?;                         // default limits
let strict = blmz::Limits::default().max_memory(512 << 20).max_output(1 << 30);
let data3 = blmz::decompress_with_limits(&packed, &strict)?;
blmz::compress_stream(reader, writer, &blmz::Options::default(), None)?;
blmz::decompress_stream(reader, writer, &blmz::Limits::default())?;    // requires EOF after
blmz::decompress_one(&mut bufreader, writer, &strict)?;                 // one stream, reader left after it
```

## Levels

Compression and decompression are symmetric: both run the whole model on every bit. Ratio and speed
measured on the first 300 KB of Silesia `dickens` (one core, Xeon @ 2.1 GHz, 260 MB L3; slower on
small-cache desktops). Memory: tables, plus history up to the window (only as much as the input).

| level | tables | window | worst case | bits/bit | KB/s |
|---|---|---|---|---|---|
| 1 | 20 MiB | 16 MiB | 36 MiB | 0.2673 | 83 |
| 4 | 66 MiB | 128 MiB | 194 MiB | 0.2500 | 57 |
| 6 (default) | 223 MiB | 256 MiB | 479 MiB | 0.2473 | 51 |
| 7 | 433 MiB | 512 MiB | 945 MiB | 0.2468 | 47 |
| 9 | 1.65 GiB | 2 GiB | 3.65 GiB | 0.2464 | 34 |

Larger inputs benefit more from larger tables (the research enwik8 runs used 2^27-slot tables).
Use `Options::with_params(Params { .. })` for custom table sizes; `cargo run --example levels`
prints every level.

## Guarantees

* **Lossless and verified**: every stream carries the content length and CRC-32; the decoder checks both.
* **Portable**: `+ - * / sqrt` are IEEE-exact with SSE2 / any non-x87 FPU, and Rust never fuses them into
  FMAs; 32-bit x86 without SSE2 is refused at compile time. The only transcendental functions (`ln`,
  `exp`, `log2`) are vendored (FreeBSD msun via `libm` 0.2.8) and pinned by `math::tests::frozen_values`.
  CI decodes committed fixture streams and checks pinned compressed bytes on Linux x86-64, macOS arm64,
  Windows x86-64 and 32-bit x86 (verified locally on x86-64 and i686).
* **Untrusted input**: header fields are range-checked and CRC-protected; `Limits::max_memory` (tables +
  history) is checked before anything is allocated, and allocation failure is `Error::OutOfMemory`, not an
  abort. Truncation, bit flips, trailing data and I/O errors are typed errors (fuzzed, and reviewed with
  ~1,900 crafted streams). **Decompression bombs are inherent** — a few bytes can legitimately expand by
  10^5+ — and decoding is as slow as encoding, so for untrusted data set `Limits::max_output` (CLI
  `--max-output`): it bounds both output and CPU time.
* **Format stability**: any change to the predictor's arithmetic is a new model id, any layout change a
  new format version, and decoders keep every old pair; `tests/fixtures/` holds streams that must decode
  forever. Until 1.0 the format is a *preview*: see `ROADMAP.md` for the planned model-2 changes.

## Format

[`FORMAT.md`](FORMAT.md) specifies every byte: header (20 or 28 bytes, CRC-32), the coder (carry-less
32-bit binary arithmetic coder, 32-bit probabilities, exact flush), the per-byte end flag, quantisation,
and the 12-byte trailer. Fixed overhead is 47 bytes for an empty input; the payload is within a few bytes
of the model's ideal code length.

## Parity with the research engine

`tools/parity.sh <file> [cap] [obits]` patches a copy of `strong.rs` to hash every per-bit probability
and compares it with `blmz` built with `RUSTFLAGS="--cfg blmz_std_math"` (platform libm, as the research
engine uses; such builds write model id 0x81, which normal builds refuse). CI runs it on every change to
`strong.rs` or `blmz/`. A model change that should ship must land in both, or ship as a new model id.

## Fuzzing

`cargo +nightly fuzz run {decompress,payload,roundtrip}` (see `fuzz/`). CI builds the targets and runs
each for 60 s; long-running fuzzing (OSS-Fuzz or a nightly job) is on the roadmap.

## Not yet

Speed (tens of KB/s), parallelism, block/seekable format, stored-block fallback for incompressible data,
interrupt cleanup of `.blmz-partial` files on SIGINT, and the model fixes found in the audit are on the
roadmap: [`ROADMAP.md`](ROADMAP.md).

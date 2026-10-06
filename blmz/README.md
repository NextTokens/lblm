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
| memory | ~10× the input + ~1.5 GB of tables | levels 1–9 (20 MiB … 1.7 GiB of tables) + bounded history window |
| untrusted input | panics on bad params | typed errors, memory/output limits, never panics on corrupt data |
| speed (same tables) | 1× | ~1.6–1.8× (same probabilities; cache-friendly layout, prefetching) |

## Use

```sh
cd blmz && cargo build --release
./target/release/blmz -6 file              # -> file.blz (keeps file)
./target/release/blmz -d file.blz          # -> file
./target/release/blmz -t file.blz          # verify (decode + length + CRC-32)
./target/release/blmz -l file.blz          # header info
tar c dir | ./target/release/blmz > dir.tar.blz     # streaming
./target/release/blmz bench file           # ratio + speed + round-trip check, in memory
```

As a library:

```rust
let packed = blmz::compress(&data, &blmz::Options::level(6))?;
let data2 = blmz::decompress(&packed)?;                       // default limits
let data3 = blmz::decompress_with_limits(&packed, &blmz::Limits { max_memory: 512 << 20, max_output: 1 << 30 })?;
blmz::compress_stream(reader, writer, &blmz::Options::default(), None)?;
blmz::decompress_stream(reader, writer, &blmz::Limits::default())?;
```

## Levels

Compression and decompression are symmetric: both run the whole model on every bit. Measured on the
first 300 KB of Silesia `dickens` (one core, Xeon @ 2.1 GHz, 260 MB L3; slower on small-cache desktops):

| level | model memory | bits/bit | KB/s |
|---|---|---|---|
| 1 | 20 MiB | 0.2673 | 83 |
| 4 | 66 MiB | 0.2500 | 57 |
| 6 (default) | 223 MiB | 0.2473 | 51 |
| 7 | 433 MiB | 0.2468 | 47 |
| 9 | 1.7 GiB | 0.2464 | 34 |

Larger inputs benefit more from larger tables (the research enwik8 runs used 2^27-slot tables).
Use `Options { params: Some(Params { .. }), .. }` for custom table sizes.

## Guarantees

* **Lossless and verified**: every stream carries the content length and CRC-32; the decoder checks both.
* **Portable**: `+ - * / sqrt` are IEEE-exact everywhere and Rust never fuses them into FMAs; the only
  transcendental functions (`ln`, `exp`, `log2`) are vendored (FreeBSD msun via `libm` 0.2.8) and pinned
  by a test (`math::tests::frozen_values`). CI checks pinned compressed bytes on Linux x86-64, macOS
  arm64, Windows x86-64 and 32-bit x86.
* **Safe on hostile input**: header fields are range-checked and CRC-protected before any allocation;
  `Limits::max_memory` refuses oversized models; a stream with a declared length cannot produce more
  output than declared; truncation, bit flips and trailing data are errors, not panics.
* **Format stability**: any change to the predictor's arithmetic is a new model id, any layout change a
  new format version; old decoders are kept. `tests/roundtrip.rs::golden_streams` pins exact bytes.
  Until 1.0 the format is a *preview*: see `ROADMAP.md` for the planned model-2 changes.

## Format (version 1)

See `src/format.rs` for the byte layout: a 20- or 28-byte header (magic `89 42 4C 5A`, version, model id,
flags, level, 7 table-size fields, optional length, header CRC-32), a binary arithmetic-coded payload
(a model-independent "more" flag per byte, P = 1 − 2^-24, then 8 model-coded bits), and a 12-byte trailer
(length, CRC-32). The coder is a carry-less 32-bit range coder with 32-bit probabilities; its overhead
against the model's ideal code length is under 4 bytes per stream.

## Parity with the research engine

`tools/parity.sh <file> [cap] [obits]` patches a copy of `strong.rs` to hash every per-bit probability
and compares it with `blmz` built with `--features std-math` (platform libm, as the research engine uses).
CI runs it on every change to `strong.rs` or `blmz/`. A model change that should ship must land in both,
or ship as a new model id.

## Not yet

Speed (tens of KB/s), parallelism, block/seekable format, stored-block fallback for incompressible data,
and the model fixes found in the audit are on the roadmap: [`ROADMAP.md`](ROADMAP.md).

# The `.blz` stream format — version 1, model id 1

Status: **preview**. Streams written today will always decode: every released decoder keeps
every (version, model id) pair it ever wrote, and `tests/fixtures/` pins real streams that CI
decodes on every platform. Until 1.0, new *encoders* may switch to a newer model id
(see `ROADMAP.md`).

All integers are little-endian unless stated otherwise. A stream is

```
header | payload | trailer
```

and streams may be concatenated (the CLI decodes them in sequence; the library's
`decompress_one` reads exactly one and leaves the reader after it).

## Header

| offset | size | field |
|---|---|---|
| 0 | 4 | magic `89 42 4C 5A` (`\x89BLZ`) |
| 4 | 1 | format version = 1. Readers check this **before** the CRC (it defines the layout) |
| 5 | 1 | model id: `1` = the predictor below with the portable math. `0x81` = the same predictor on the platform libm (parity builds only; never accepted by normal readers) |
| 6 | 1 | flags. bit 0: content length present. Other bits must be 0 |
| 7 | 1 | level used by the writer (informational; 0 = custom parameters) |
| 8 | 7 | `obits hbits mbits sbits selvbits selubits window_log` |
| 15 | 1 | reserved, 0 |
| 16 | 8 | content length (only if flags bit 0) |
| 16 or 24 | 4 | CRC-32 of all preceding header bytes |

Valid parameters: each table field in [10, 30], `window_log` in [16, 32]. Readers must reject
anything else, and must also be able to refuse a stream whose memory need
(`tables + min(2^window_log, content length)`, see `Params::total_memory`) exceeds a limit
*before* allocating.

CRC-32 everywhere is the IEEE 802.3 / zlib / PNG CRC: reflected, polynomial `0xEDB88320`,
initial value `0xFFFFFFFF`, final XOR `0xFFFFFFFF` (`crc32("123456789") = 0xCBF43926`).

## Payload

A binary arithmetic code. For every content byte, in order:

1. the **more flag** `1`, coded with `p1 = 0xFFFFFF00` (P(more) = 1 − 2^-24), independent of the model;
2. the byte's 8 bits, most significant first, each coded with `p1 = quantize(model.p())`, after
   which the model is updated with the actual bit.

After the last byte the flag `0` is coded with the same `p1 = 0xFFFFFF00`, then the coder is
flushed. An empty content is just the `0` flag and the flush.

### Coder

State: `x1 = 0`, `x2 = 0xFFFFFFFF` (u32). To code `bit` with `p1` in [1, 2^32 − 1]
(the probability that the bit is 1, in units of 2^-32):

```
xmid = x1 + (((x2 - x1) as u64 * p1 as u64) >> 32)      // u32 arithmetic, no overflow
if bit == 1 { x2 = xmid } else { x1 = xmid + 1 }
while (x1 ^ x2) & 0xFF000000 == 0 {                       // leading bytes equal: emit
    output(x2 >> 24); x1 = x1 << 8; x2 = (x2 << 8) | 0xFF
}
```

Flush: output the 4 bytes of `x1`, most significant first.

Decoder: `x` = the first 4 payload bytes (big-endian); the same `xmid`; `bit = (x <= xmid)`;
the same interval update; on every shift also `x = (x << 8) | next_byte`. The decoder consumes
exactly as many bytes as the encoder produced, so the trailer begins immediately after.
A decoder that needs a byte past the end of its input has a truncated or corrupt stream.

### Quantisation

`quantize(p)`: `q = p * 2^32` (f64); if `q >= 4294967295.0` then `0xFFFFFFFF`, else if `q >= 1.0`
then `q as u32` (truncation), else `1`. NaN maps to 1 (the model never produces NaN).

### The model (model id 1)

The probabilities come from the predictor in `src/model.rs` + `src/math.rs`, which **are the
normative definition** of model id 1: it is the adopted default configuration of the research
engine `blmrs/src/bin/strong.rs` (bit-identical per-bit probabilities on the same machine,
`tools/parity.sh`), sized by the header's table parameters, with two changes that matter only
for inputs longer than the history window (2^window_log bytes): history older than the window is
not kept, and a match candidate further back than the window is not taken.

All arithmetic is IEEE-754 binary64 with round-to-nearest-even and no fused multiply-add; the
only transcendental functions (`ln`, `exp`, `log2`) are the FreeBSD msun algorithms vendored in
`src/math.rs` (pinned by `math::tests::frozen_values`). An implementation in another language
must reproduce these operations exactly, in the same order.

## Trailer

| size | field |
|---|---|
| 8 | content length |
| 4 | CRC-32 of the content |

Readers must check both (and that a header length, if present, agrees).

## Overheads (measured)

Header 20 bytes (28 with a content length) + trailer 12 + coder flush 4 + end flag ~3 = **47 bytes
for an empty input** at level ≥ 1 with a length. The per-byte more flag costs 8.6·10^-8 bits.
Payload size is within a few bytes of the model's ideal code length (Σ −log2 p).

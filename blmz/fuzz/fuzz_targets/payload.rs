//! A valid header followed by fuzzer-controlled payload/trailer bytes: exercises the arithmetic
//! decoder, the model on garbage, the EOS flag, and length/CRC checks (the header CRC would
//! otherwise reject almost every mutated input before the payload is reached).
#![no_main]
use libfuzzer_sys::fuzz_target;
#[path = "common.rs"]
mod common;

fuzz_target!(|data: &[u8]| {
    let with_len = data.first().map(|b| b & 1 == 1).unwrap_or(false);
    let h = blmz::format::Header {
        version: blmz::format::VERSION,
        model_id: blmz::format::MODEL_ID,
        level: 0,
        params: common::tiny(),
        content_length: if with_len { Some(data.len() as u64 % 64) } else { None },
    };
    let mut stream = h.to_bytes();
    stream.extend_from_slice(data);
    let _ = blmz::decompress_with_limits(&stream, &common::limits());
});

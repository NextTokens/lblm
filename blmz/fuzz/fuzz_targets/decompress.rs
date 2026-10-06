//! Arbitrary bytes into the full decoder: must return Ok/Err, never panic, hang or over-allocate.
#![no_main]
use libfuzzer_sys::fuzz_target;
#[path = "common.rs"]
mod common;

fuzz_target!(|data: &[u8]| {
    let _ = blmz::decompress_with_limits(data, &common::limits());
});

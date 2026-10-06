//! compress -> decompress must reproduce any input exactly.
#![no_main]
use libfuzzer_sys::fuzz_target;
#[path = "common.rs"]
mod common;

fuzz_target!(|data: &[u8]| {
    let opts = blmz::Options::with_params(common::tiny());
    let packed = blmz::compress(data, &opts).expect("compress");
    let back = blmz::decompress(&packed).expect("decompress");
    assert_eq!(back, data);
});

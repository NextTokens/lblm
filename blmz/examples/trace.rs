//! Parity probe: prints the probability-trace hash and ideal cost of the predictor on a file,
//! in the same form as the research engine patched for parity (`phash`, `whole-stream`).
//!
//!   cargo run --release --features std-math --example trace -- <file> <byte_cap> <obits>
//!
//! With `std-math` the model uses the platform libm exactly like blmrs/src/bin/strong.rs, so the
//! hash must equal the research engine's on the same machine. Without it, the portable math is
//! used and the cost should differ only in the ~6th decimal.
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let path = &a[1];
    let cap: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let obits: u8 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(22);
    let mut data = std::fs::read(path).expect("read");
    if cap > 0 && data.len() > cap {
        data.truncate(cap);
    }
    let t = std::time::Instant::now();
    let (h, tot) = blmz::probability_trace_hash(&data, blmz::Params::research(obits));
    let n = data.len() as f64 * 8.0;
    println!("bytes={} obits={} model={:#x}", data.len(), obits, blmz::format::MODEL_ID);
    println!("  phash = {:016x}  totbits = {:.6}", h, tot);
    println!(
        "  whole-stream = {:.6} bits/bit  ideal bytes = {:.1}  [{:.1}s]",
        if n > 0.0 { tot / n } else { 0.0 },
        tot / 8.0,
        t.elapsed().as_secs_f64()
    );
}

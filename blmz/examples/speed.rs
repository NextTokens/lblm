//! Speed probe: model-only throughput at a level preset.  cargo run --release --example speed -- <file> <level>
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).expect("read");
    let level: u8 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(6);
    let p = blmz::level_params(level).expect("level");
    let t = std::time::Instant::now();
    let bits = blmz::ideal_bits(&data, p);
    let s = t.elapsed().as_secs_f64();
    println!(
        "level {} mem {:.0} MiB: {:.6} bits/bit, {:.1} KB/s",
        level,
        p.memory_bytes() as f64 / 1048576.0,
        bits / (8.0 * data.len() as f64),
        data.len() as f64 / 1024.0 / s
    );
}

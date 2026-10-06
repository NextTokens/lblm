//! Print the memory each level needs (tables, window, worst case).  cargo run --release --example levels
fn main() {
    for l in 1..=9 {
        let p = blmz::level_params(l).unwrap();
        let mib = |b: u64| b as f64 / 1048576.0;
        println!(
            "level {}: tables {:.0} MiB, window {:.0} MiB, worst case {:.0} MiB",
            l,
            mib(p.memory_bytes()),
            mib(p.history_bytes(None)),
            mib(p.total_memory(None))
        );
    }
}

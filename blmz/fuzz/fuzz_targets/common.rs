#![allow(dead_code)]
// Small tables so each iteration is cheap; the code paths are the same at every size.
pub fn tiny() -> blmz::Params {
    blmz::Params {
        obits: 10,
        hbits: 10,
        mbits: 10,
        sbits: 10,
        selvbits: 10,
        selubits: 10,
        window_log: 16,
    }
}

pub fn limits() -> blmz::Limits {
    blmz::Limits::default().max_memory(64 << 20).max_output(4096)
}

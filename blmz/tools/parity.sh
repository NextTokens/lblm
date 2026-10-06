#!/usr/bin/env bash
# Parity gate: prove the blmz predictor is bit-identical to the research engine
# (blmrs/src/bin/strong.rs, default flags) on the same machine.
#
#   blmz/tools/parity.sh <file> [byte_cap=200000] [obits=20]
#
# Builds (1) a copy of strong.rs patched ONLY to print an FNV-1a hash of every per-bit
# probability, and (2) blmz's trace example with the platform libm (--cfg blmz_std_math), then
# compares the two hashes. Any model change in strong.rs that should reach production must keep
# this gate green against a matching change in blmz (or be released as a new MODEL_ID).
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
repo="$(cd "$here/.." && pwd)"
file="${1:?usage: parity.sh <file> [byte_cap] [obits]}"
cap="${2:-200000}"
obits="${3:-20}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

python3 - "$repo/blmrs/src/bin/strong.rs" "$work/strong_ref.rs" <<'PY'
import sys
s = open(sys.argv[1]).read()
def sub(old, new):
    global s
    assert s.count(old) == 1, "strong.rs changed shape near: " + old[:60]
    s = s.replace(old, new)
sub("    let mut tailn = 0usize;\n",
    "    let mut tailn = 0usize;\n    let mut phash: u64 = 0xcbf29ce484222325;\n")
sub("        let y = bits[i];\n",
    "        for bb in p.to_bits().to_le_bytes() { phash = (phash ^ bb as u64).wrapping_mul(0x100000001b3); }\n        let y = bits[i];\n")
sub('    println!("  blmrs-strong  whole-stream',
    '    println!("  phash = {:016x}  totbits = {:.6}", phash, tot);\n    println!("  blmrs-strong  whole-stream')
open(sys.argv[2], "w").write(s)
PY
rustc -O --edition 2021 "$work/strong_ref.rs" -o "$work/strong_ref" 2>/dev/null
# separate target dir: a cfg'd build must never overwrite the portable binaries
parity_target="$here/target/parity"
(cd "$here" && RUSTFLAGS="--cfg blmz_std_math" CARGO_TARGET_DIR="$parity_target" cargo build --quiet --release --example trace)

ref="$("$work/strong_ref" "$file" "$cap" "$obits" | grep phash)"
got="$("$parity_target/release/examples/trace" "$file" "$cap" "$obits" | grep phash)"
echo "research engine: $ref"
echo "blmz (std math): $got"
if [ "$ref" = "$got" ]; then echo "PARITY OK"; else echo "PARITY FAILED"; exit 1; fi

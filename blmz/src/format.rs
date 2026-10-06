//! The `.blz` container, format version 1.
//!
//! ```text
//! offset size  field
//!  0     4     magic 0x89 'B' 'L' 'Z'
//!  4     1     format version (1)
//!  5     1     model id (MODEL_ID; identifies the exact predictor incl. its math)
//!  6     1     flags: bit 0 = content length present; all other bits must be 0
//!  7     1     level the stream was written with (informational; 0 = custom)
//!  8     7     model parameters: obits hbits mbits sbits selvbits selubits window_log
//! 15     1     reserved, must be 0
//! [16    8     content length, u64 little-endian]          only if flags bit 0
//!  ..    4     CRC-32 of all header bytes above, little-endian
//!  ..    ..    payload: binary arithmetic code. Before each byte, a "more" flag is coded with
//!              P(more) = 1 - 2^-24 (model-independent); then the byte's 8 bits, MSB first, with
//!              the model's probabilities. A 0 flag ends the payload. The coder's 4-byte flush
//!              follows; the decoder consumes exactly the payload bytes.
//!  ..    8     content length, u64 little-endian
//!  ..    4     CRC-32 of the content, little-endian
//! ```
//!
//! Compatibility rule: any change to the predictor's arithmetic, constants or update order is a
//! new MODEL_ID; any change to this layout is a new version. Decoders keep every old pair.

use crate::model::Params;

pub const MAGIC: [u8; 4] = [0x89, b'B', b'L', b'Z'];
pub const VERSION: u8 = 1;

/// Model id 1: the strong.rs adopted default path with the frozen portable math (math.rs).
#[cfg(not(blmz_std_math))]
pub const MODEL_ID: u8 = 1;
/// Model id 0x81: the same predictor on the PLATFORM libm. Not portable; parity testing only
/// (`RUSTFLAGS="--cfg blmz_std_math"`).
#[cfg(blmz_std_math)]
pub const MODEL_ID: u8 = 0x81;

pub const FLAG_LENGTH: u8 = 1;

/// P(another byte follows), 32-bit fixed point: 1 - 2^-24.
pub const P_MORE: u32 = 0xFFFF_FF00;

pub const TRAILER_LEN: usize = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub version: u8,
    pub model_id: u8,
    pub level: u8,
    pub params: Params,
    pub content_length: Option<u64>,
}

impl Header {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut h = Vec::with_capacity(28);
        h.extend_from_slice(&MAGIC);
        h.push(self.version);
        h.push(self.model_id);
        h.push(if self.content_length.is_some() { FLAG_LENGTH } else { 0 });
        h.push(self.level);
        let p = &self.params;
        h.extend_from_slice(&[p.obits, p.hbits, p.mbits, p.sbits, p.selvbits, p.selubits, p.window_log, 0]);
        if let Some(n) = self.content_length {
            h.extend_from_slice(&n.to_le_bytes());
        }
        let c = crate::crc32::crc32(&h);
        h.extend_from_slice(&c.to_le_bytes());
        h
    }

    /// Parse a header from the start of `r`. Returns the header; the reader is left at the payload.
    pub fn read_from<R: std::io::Read>(r: &mut R) -> Result<Header, crate::Error> {
        use crate::Error;
        let mut fixed = [0u8; 16];
        read_exact_or(r, &mut fixed[..4], Error::NotBlz)?;
        if fixed[0..4] != MAGIC {
            return Err(Error::NotBlz);
        }
        read_exact_or(r, &mut fixed[4..], Error::Truncated)?;
        // the version decides the layout, so it is read before the (layout-dependent) CRC
        let version = fixed[4];
        if version != VERSION {
            return Err(Error::UnsupportedVersion(version));
        }
        let model_id = fixed[5];
        let flags = fixed[6];
        let mut all = fixed.to_vec();
        let content_length = if flags & FLAG_LENGTH != 0 {
            let mut l = [0u8; 8];
            read_exact_or(r, &mut l, Error::Truncated)?;
            all.extend_from_slice(&l);
            Some(u64::from_le_bytes(l))
        } else {
            None
        };
        let mut c = [0u8; 4];
        read_exact_or(r, &mut c, Error::Truncated)?;
        if u32::from_le_bytes(c) != crate::crc32::crc32(&all) {
            return Err(Error::CorruptHeader("header checksum mismatch"));
        }
        if flags & !FLAG_LENGTH != 0 || fixed[15] != 0 {
            return Err(Error::CorruptHeader("reserved bits set"));
        }
        if model_id != MODEL_ID {
            return Err(Error::UnsupportedModel(model_id));
        }
        let params = Params {
            obits: fixed[8],
            hbits: fixed[9],
            mbits: fixed[10],
            sbits: fixed[11],
            selvbits: fixed[12],
            selubits: fixed[13],
            window_log: fixed[14],
        };
        params.validate().map_err(Error::BadParams)?;
        Ok(Header {
            version,
            model_id,
            level: fixed[7],
            params,
            content_length,
        })
    }
}

fn read_exact_or<R: std::io::Read>(r: &mut R, buf: &mut [u8], short: crate::Error) -> Result<(), crate::Error> {
    match r.read_exact(buf) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Err(short),
        Err(e) => Err(crate::Error::Io(e)),
    }
}

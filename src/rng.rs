use std::time::SystemTime;
use crate::vector::*;

/*

    khash function
    This function implements the khash algorithm, a simple and efficient hash function.
    It takes a 32-bit unsigned integer 'state' as input and returns a 32-bit unsigned integer as the hash value.
    The function applies a series of bitwise operations and multiplication with prime constants to scramble the input bits and produce the hash value.
    The khash algorithm consists of the following steps:
        XOR the 'state' with a constant value (2747636419).
        Multiply the result by another constant value (2654435769) using the wrapping multiplication to avoid overflow.
        XOR the result with the right-shifted version of itself by 16 bits.
        Repeat step 3 one more time.
        Multiply the final result by the same constant value (2654435769) using wrapping multiplication.
    The wrapping multiplication and XOR operations help ensure that the hash function produces a valid 32-bit unsigned integer without overflow issues.
    Although this hash function is not cryptographically secure, it is suitable for many non-cryptographic hash table implementations and other basic hashing needs.
    */

pub fn khash(mut state: u32) -> u32 {
    state = (state ^ 2747636419).wrapping_mul(2654435769);
    state = (state ^ (state >> 16)).wrapping_mul(2654435769);
    state = (state ^ (state >> 16)).wrapping_mul(2654435769);
    state
}
pub fn krand(seed: u32) -> f32 {
    khash(seed) as f32 / 4294967295.0
}

#[derive(Clone)]
pub struct Rng {
    seed: u32,
}

impl Rng {
    pub fn new_random() -> Self {
        Rng { seed: SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map(|x| x.as_nanos() as u32).unwrap_or(1) }
    }
    pub fn new_seeded(starting_seed: u32) -> Self {
        Rng { seed: starting_seed }
    }
    pub fn next_float(&mut self) -> f32 {
        self.seed = khash(self.seed) + 69;
        self.seed as f32 / 4294967295.0
    }
}
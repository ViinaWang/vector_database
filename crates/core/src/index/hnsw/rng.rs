//! 自带的小型 xorshift64* 生成器。不引外部依赖是为了 wasm 目标零摩擦，
//! 且构建结果确定（同数据 + 同参数 → 同图，便于测试对照）。

/// 确定性伪随机源。
pub struct Rng(u64);

impl Rng {
    /// 固定种子。
    pub fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    /// 下一个 u64。
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// (0, 1) 内的 f32，避免端点。
    pub fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32 & 0xFF_FFFF;
        (bits as f32 + 0.5) / 16_777_217.0
    }
}

#[cfg(test)]
mod tests {
    use super::Rng;

    #[test]
    fn deterministic_and_in_range() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        let mut r = Rng::new(7);
        for _ in 0..1000 {
            let v = r.next_f32();
            assert!((0.0..1.0).contains(&v));
        }
    }
}

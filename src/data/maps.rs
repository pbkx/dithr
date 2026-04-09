pub const BAYER_2X2: [[u8; 2]; 2] = [[0, 2], [3, 1]];

pub const BAYER_2X2_FLAT: [u16; 4] = [0, 2, 3, 1];

pub const BAYER_4X4: [[u8; 4]; 4] = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];

pub const BAYER_4X4_FLAT: [u16; 16] = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];

pub const BAYER_8X8: [[u8; 8]; 8] = [
    [0, 32, 8, 40, 2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
];

pub const BAYER_8X8_FLAT: [u16; 64] = [
    0, 32, 8, 40, 2, 34, 10, 42, 48, 16, 56, 24, 50, 18, 58, 26, 12, 44, 4, 36, 14, 46, 6, 38, 60,
    28, 52, 20, 62, 30, 54, 22, 3, 35, 11, 43, 1, 33, 9, 41, 51, 19, 59, 27, 49, 17, 57, 25, 15,
    47, 7, 39, 13, 45, 5, 37, 63, 31, 55, 23, 61, 29, 53, 21,
];

pub const CLUSTER_DOT_4X4: [[u8; 4]; 4] =
    [[12, 5, 6, 13], [4, 0, 1, 7], [11, 3, 2, 8], [15, 10, 9, 14]];

pub const CLUSTER_DOT_4X4_FLAT: [u16; 16] = [12, 5, 6, 13, 4, 0, 1, 7, 11, 3, 2, 8, 15, 10, 9, 14];

pub const CLUSTER_DOT_8X8: [[u8; 8]; 8] = [
    [24, 10, 12, 26, 35, 47, 49, 37],
    [8, 0, 2, 14, 45, 59, 61, 51],
    [22, 6, 4, 16, 43, 57, 63, 53],
    [30, 20, 18, 28, 33, 41, 55, 39],
    [34, 46, 48, 36, 25, 11, 13, 27],
    [44, 58, 60, 50, 9, 1, 3, 15],
    [42, 56, 62, 52, 23, 7, 5, 17],
    [32, 40, 54, 38, 31, 21, 19, 29],
];

pub const CLUSTER_DOT_8X8_FLAT: [u16; 64] = [
    24, 10, 12, 26, 35, 47, 49, 37, 8, 0, 2, 14, 45, 59, 61, 51, 22, 6, 4, 16, 43, 57, 63, 53, 30,
    20, 18, 28, 33, 41, 55, 39, 34, 46, 48, 36, 25, 11, 13, 27, 44, 58, 60, 50, 9, 1, 3, 15, 42,
    56, 62, 52, 23, 7, 5, 17, 32, 40, 54, 38, 31, 21, 19, 29,
];

#[must_use]
pub fn generate_bayer_16x16() -> [[u8; 16]; 16] {
    let mut out = [[0_u8; 16]; 16];

    for y in 0..16 {
        for x in 0..16 {
            let base = u16::from(BAYER_8X8[y % 8][x % 8]);
            let quadrant = u16::from(BAYER_2X2[y / 8][x / 8]);
            out[y][x] = (base * 4 + quadrant) as u8;
        }
    }

    out
}

#[must_use]
pub fn generate_bayer_16x16_flat() -> [u16; 256] {
    let map = generate_bayer_16x16();
    let mut flat = [0_u16; 256];

    for (y, row) in map.iter().enumerate() {
        for (x, value) in row.iter().enumerate() {
            flat[y * 16 + x] = u16::from(*value);
        }
    }

    flat
}

const VOID_CLUSTER_SIDE: usize = 64;
const VOID_CLUSTER_LEN: usize = VOID_CLUSTER_SIDE * VOID_CLUSTER_SIDE;
const VOID_CLUSTER_RADIUS: isize = 7;
const VOID_CLUSTER_INITIAL_ONES: usize = VOID_CLUSTER_LEN / 8;

#[must_use]
pub fn generate_void_and_cluster_64x64_flat() -> [u16; VOID_CLUSTER_LEN] {
    let kernel = void_cluster_kernel();
    let mut pattern = void_cluster_initial_pattern();
    void_cluster_relax(&mut pattern, &kernel);

    let mut ranks = [u16::MAX; VOID_CLUSTER_LEN];
    let mut assigned = [false; VOID_CLUSTER_LEN];

    let mut phase1_pattern = pattern;
    let mut phase1_field = void_cluster_field(&phase1_pattern, &kernel);
    let ones = phase1_pattern.iter().filter(|&&value| value).count();
    for rank in (0..ones).rev() {
        if let Some(idx) = void_cluster_find_cluster(&phase1_pattern, &phase1_field) {
            ranks[idx] = rank as u16;
            assigned[idx] = true;
            void_cluster_set(&mut phase1_pattern, &mut phase1_field, &kernel, idx, false);
        }
    }

    let mut phase2_pattern = pattern;
    let mut phase2_field = void_cluster_field(&phase2_pattern, &kernel);
    for rank in ones..VOID_CLUSTER_LEN {
        if let Some(idx) = void_cluster_find_void(&phase2_pattern, &phase2_field) {
            ranks[idx] = rank as u16;
            assigned[idx] = true;
            void_cluster_set(&mut phase2_pattern, &mut phase2_field, &kernel, idx, true);
        }
    }

    let mut used = [false; VOID_CLUSTER_LEN];
    for &rank in ranks.iter() {
        if rank != u16::MAX {
            used[usize::from(rank)] = true;
        }
    }

    let mut next = 0_usize;
    for (idx, rank) in ranks.iter_mut().enumerate() {
        if assigned[idx] {
            continue;
        }
        while next < VOID_CLUSTER_LEN && used[next] {
            next += 1;
        }
        if next < VOID_CLUSTER_LEN {
            *rank = next as u16;
            used[next] = true;
        }
    }

    ranks
}

fn void_cluster_initial_pattern() -> [bool; VOID_CLUSTER_LEN] {
    let mut ranked = Vec::with_capacity(VOID_CLUSTER_LEN);
    for idx in 0..VOID_CLUSTER_LEN {
        ranked.push((void_cluster_hash(idx), idx));
    }
    ranked.sort_unstable_by_key(|&(hash, idx)| (hash, idx));

    let mut pattern = [false; VOID_CLUSTER_LEN];
    for &(_, idx) in ranked.iter().take(VOID_CLUSTER_INITIAL_ONES) {
        pattern[idx] = true;
    }

    pattern
}

fn void_cluster_hash(idx: usize) -> u64 {
    let x = (idx % VOID_CLUSTER_SIDE) as u64;
    let y = (idx / VOID_CLUSTER_SIDE) as u64;
    let mixed = x
        .wrapping_mul(0x9e37_79b1_85eb_ca87_u64)
        .wrapping_add(y.wrapping_mul(0xc2b2_ae3d_27d4_eb4f_u64))
        .wrapping_add(0x94d0_49bb_1331_11eb_u64);
    mix_u64(mixed)
}

fn mix_u64(mut value: u64) -> u64 {
    value ^= value >> 33;
    value = value.wrapping_mul(0xff51_afd7_ed55_8ccd_u64);
    value ^= value >> 33;
    value = value.wrapping_mul(0xc4ce_b9fe_1a85_ec53_u64);
    value ^ (value >> 33)
}

fn void_cluster_kernel() -> Vec<(isize, isize, i64)> {
    let mut kernel = Vec::new();
    let radius_sq = VOID_CLUSTER_RADIUS * VOID_CLUSTER_RADIUS;
    for dy in -VOID_CLUSTER_RADIUS..=VOID_CLUSTER_RADIUS {
        for dx in -VOID_CLUSTER_RADIUS..=VOID_CLUSTER_RADIUS {
            let distance_sq = dx * dx + dy * dy;
            if distance_sq > radius_sq {
                continue;
            }
            let weight = 1024_i64 / i64::from(1 + distance_sq as i32);
            kernel.push((dx, dy, weight.max(1)));
        }
    }
    kernel
}

fn void_cluster_field(
    pattern: &[bool; VOID_CLUSTER_LEN],
    kernel: &[(isize, isize, i64)],
) -> [i64; VOID_CLUSTER_LEN] {
    let mut field = [0_i64; VOID_CLUSTER_LEN];
    for (idx, &value) in pattern.iter().enumerate() {
        if value {
            void_cluster_accumulate(&mut field, kernel, idx, 1);
        }
    }
    field
}

fn void_cluster_relax(pattern: &mut [bool; VOID_CLUSTER_LEN], kernel: &[(isize, isize, i64)]) {
    let mut field = void_cluster_field(pattern, kernel);
    for _ in 0..(VOID_CLUSTER_LEN * 4) {
        let cluster = match void_cluster_find_cluster(pattern, &field) {
            Some(value) => value,
            None => break,
        };
        let void = match void_cluster_find_void(pattern, &field) {
            Some(value) => value,
            None => break,
        };
        if field[cluster] <= field[void] {
            break;
        }
        void_cluster_set(pattern, &mut field, kernel, cluster, false);
        void_cluster_set(pattern, &mut field, kernel, void, true);
    }
}

fn void_cluster_set(
    pattern: &mut [bool; VOID_CLUSTER_LEN],
    field: &mut [i64; VOID_CLUSTER_LEN],
    kernel: &[(isize, isize, i64)],
    idx: usize,
    value: bool,
) {
    if pattern[idx] == value {
        return;
    }
    pattern[idx] = value;
    let delta = if value { 1 } else { -1 };
    void_cluster_accumulate(field, kernel, idx, delta);
}

fn void_cluster_accumulate(
    field: &mut [i64; VOID_CLUSTER_LEN],
    kernel: &[(isize, isize, i64)],
    idx: usize,
    delta: i64,
) {
    let x = (idx % VOID_CLUSTER_SIDE) as isize;
    let y = (idx / VOID_CLUSTER_SIDE) as isize;
    for &(dx, dy, weight) in kernel {
        let nx = void_cluster_wrap(x + dx);
        let ny = void_cluster_wrap(y + dy);
        let target = ny * VOID_CLUSTER_SIDE + nx;
        field[target] += delta * weight;
    }
}

fn void_cluster_find_cluster(
    pattern: &[bool; VOID_CLUSTER_LEN],
    field: &[i64; VOID_CLUSTER_LEN],
) -> Option<usize> {
    let mut best_idx = None;
    let mut best_value = i64::MIN;
    for (idx, &value) in pattern.iter().enumerate() {
        if !value {
            continue;
        }
        let score = field[idx];
        let tie_break = match best_idx {
            Some(current) => idx < current,
            None => true,
        };
        if score > best_value || (score == best_value && tie_break) {
            best_value = score;
            best_idx = Some(idx);
        }
    }
    best_idx
}

fn void_cluster_find_void(
    pattern: &[bool; VOID_CLUSTER_LEN],
    field: &[i64; VOID_CLUSTER_LEN],
) -> Option<usize> {
    let mut best_idx = None;
    let mut best_value = i64::MAX;
    for (idx, &value) in pattern.iter().enumerate() {
        if value {
            continue;
        }
        let score = field[idx];
        let tie_break = match best_idx {
            Some(current) => idx < current,
            None => true,
        };
        if score < best_value || (score == best_value && tie_break) {
            best_value = score;
            best_idx = Some(idx);
        }
    }
    best_idx
}

fn void_cluster_wrap(value: isize) -> usize {
    let side = VOID_CLUSTER_SIDE as isize;
    value.rem_euclid(side) as usize
}

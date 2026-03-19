mod common;

use common::{
    bench_gray_case, bench_rgb_case, gray_ramp, mode_gray_1, mode_palette_cga, rgb_gradient,
    set_gray_throughput, set_rgb_throughput, touch_common,
};
use criterion::{criterion_group, criterion_main, Criterion};
use dithr::{
    direct_binary_search_in_place, electrostatic_halftoning_in_place, knuth_dot_diffusion_in_place,
    lattice_boltzmann_in_place, riemersma_in_place,
};

fn bench_advanced(c: &mut Criterion) {
    touch_common();

    let width = 256;
    let height = 256;
    let fixture = gray_ramp(width, height);

    let mut group = c.benchmark_group("advanced_gray_riemersma_dotdiff_256");
    group.sample_size(10);
    set_gray_throughput(&mut group, width, height);

    bench_gray_case(
        &mut group,
        "riemersma_gray1_256",
        &fixture,
        width,
        height,
        |buffer| riemersma_in_place(buffer, mode_gray_1()),
    );
    bench_gray_case(
        &mut group,
        "knuth_dot_diffusion_gray1_256",
        &fixture,
        width,
        height,
        |buffer| knuth_dot_diffusion_in_place(buffer, mode_gray_1()),
    );
    group.finish();

    let fixture = rgb_gradient(width, height);
    let mut group = c.benchmark_group("advanced_rgb_riemersma_dotdiff_256");
    group.sample_size(10);
    set_rgb_throughput(&mut group, width, height);

    bench_rgb_case(
        &mut group,
        "riemersma_rgb_palette_cga_256",
        &fixture,
        width,
        height,
        |buffer| riemersma_in_place(buffer, mode_palette_cga()),
    );
    bench_rgb_case(
        &mut group,
        "knuth_dot_diffusion_rgb_palette_cga_256",
        &fixture,
        width,
        height,
        |buffer| knuth_dot_diffusion_in_place(buffer, mode_palette_cga()),
    );
    group.finish();

    let width = 128;
    let height = 128;
    let fixture = gray_ramp(width, height);
    let mut group = c.benchmark_group("advanced_gray_dbs_128");
    group.sample_size(10);
    set_gray_throughput(&mut group, width, height);

    bench_gray_case(
        &mut group,
        "direct_binary_search_gray_128_iters8",
        &fixture,
        width,
        height,
        |buffer| direct_binary_search_in_place(buffer, 8),
    );
    bench_gray_case(
        &mut group,
        "lattice_boltzmann_gray_128_steps8",
        &fixture,
        width,
        height,
        |buffer| lattice_boltzmann_in_place(buffer, 8),
    );
    bench_gray_case(
        &mut group,
        "electrostatic_halftoning_gray_128_steps8",
        &fixture,
        width,
        height,
        |buffer| electrostatic_halftoning_in_place(buffer, 8),
    );
    group.finish();
}

criterion_group!(benches, bench_advanced);
criterion_main!(benches);

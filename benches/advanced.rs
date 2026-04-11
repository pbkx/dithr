mod common;

use common::{
    bench_gray_case, bench_rgb_case, gray_ramp, mode_gray_1, mode_palette_cga, rgb_gradient,
    set_gray_throughput, set_rgb_throughput, touch_common,
};
use criterion::{criterion_group, criterion_main, Criterion};
use dithr::dbs::{
    direct_binary_search_in_place, electrostatic_halftoning_in_place, lattice_boltzmann_in_place,
    least_squares_model_based_in_place, model_based_med_in_place,
};
use dithr::dot_diffusion::{knuth_dot_diffusion_in_place, optimized_dot_diffusion_in_place};
use dithr::riemersma::riemersma_in_place;
use std::time::Duration;

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
    bench_gray_case(
        &mut group,
        "optimized_dot_diffusion_gray1_256",
        &fixture,
        width,
        height,
        |buffer| optimized_dot_diffusion_in_place(buffer, mode_gray_1()),
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
    bench_rgb_case(
        &mut group,
        "optimized_dot_diffusion_rgb_palette_cga_256",
        &fixture,
        width,
        height,
        |buffer| optimized_dot_diffusion_in_place(buffer, mode_palette_cga()),
    );
    group.finish();

    let width = 64;
    let height = 64;
    let fixture = gray_ramp(width, height);
    let mut group = c.benchmark_group("advanced_gray_physical_64");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    set_gray_throughput(&mut group, width, height);

    bench_gray_case(
        &mut group,
        "lattice_boltzmann_gray_64_steps6",
        &fixture,
        width,
        height,
        |buffer| lattice_boltzmann_in_place(buffer, 6),
    );
    bench_gray_case(
        &mut group,
        "electrostatic_halftoning_gray_64_steps4",
        &fixture,
        width,
        height,
        |buffer| electrostatic_halftoning_in_place(buffer, 4),
    );
    bench_gray_case(
        &mut group,
        "model_based_med_gray_64",
        &fixture,
        width,
        height,
        model_based_med_in_place,
    );
    bench_gray_case(
        &mut group,
        "least_squares_model_based_gray_64_iters2",
        &fixture,
        width,
        height,
        |buffer| least_squares_model_based_in_place(buffer, 2),
    );
    group.finish();

    let width = 32;
    let height = 32;
    let fixture = gray_ramp(width, height);
    let mut group = c.benchmark_group("advanced_gray_dbs_32");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    set_gray_throughput(&mut group, width, height);

    bench_gray_case(
        &mut group,
        "direct_binary_search_gray_32_iters4",
        &fixture,
        width,
        height,
        |buffer| direct_binary_search_in_place(buffer, 4),
    );
    group.finish();
}

criterion_group!(benches, bench_advanced);
criterion_main!(benches);

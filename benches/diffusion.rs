mod common;

use common::{
    bench_gray_case, bench_gray_case_u16, bench_rgb_case, gray_ramp, gray_ramp_u16, mode_gray_1,
    mode_gray_levels2_u16, mode_palette_cga, rgb_gradient, set_gray_throughput, set_rgb_throughput,
    touch_common,
};
use criterion::{criterion_group, criterion_main, Criterion};
use dithr::{
    atkinson_in_place, burkes_in_place, false_floyd_steinberg_in_place, fan_in_place,
    floyd_steinberg_in_place, gradient_based_error_diffusion_in_place,
    jarvis_judice_ninke_in_place, ostromoukhov_in_place, shiau_fan_2_in_place, shiau_fan_in_place,
    sierra_in_place, sierra_lite_in_place, stevenson_arce_in_place, stucki_in_place,
    two_row_sierra_in_place, zhou_fang_in_place,
};

fn bench_diffusion(c: &mut Criterion) {
    touch_common();

    let width = 512;
    let height = 512;
    let fixture = gray_ramp(width, height);

    let mut group = c.benchmark_group("diffusion_gray_512_classic_smallkernels");
    group.sample_size(20);
    set_gray_throughput(&mut group, width, height);

    bench_gray_case(
        &mut group,
        "floyd_steinberg_gray1_512",
        &fixture,
        width,
        height,
        |buffer| floyd_steinberg_in_place(buffer, mode_gray_1()),
    );
    bench_gray_case(
        &mut group,
        "false_floyd_steinberg_gray1_512",
        &fixture,
        width,
        height,
        |buffer| false_floyd_steinberg_in_place(buffer, mode_gray_1()),
    );
    bench_gray_case(
        &mut group,
        "burkes_gray1_512",
        &fixture,
        width,
        height,
        |buffer| burkes_in_place(buffer, mode_gray_1()),
    );
    bench_gray_case(
        &mut group,
        "sierra_lite_gray1_512",
        &fixture,
        width,
        height,
        |buffer| sierra_lite_in_place(buffer, mode_gray_1()),
    );
    bench_gray_case(
        &mut group,
        "atkinson_gray1_512",
        &fixture,
        width,
        height,
        |buffer| atkinson_in_place(buffer, mode_gray_1()),
    );
    group.finish();

    let mut group = c.benchmark_group("diffusion_gray_512_classic_largekernels");
    group.sample_size(20);
    set_gray_throughput(&mut group, width, height);

    bench_gray_case(
        &mut group,
        "jarvis_judice_ninke_gray1_512",
        &fixture,
        width,
        height,
        |buffer| jarvis_judice_ninke_in_place(buffer, mode_gray_1()),
    );
    bench_gray_case(
        &mut group,
        "stucki_gray1_512",
        &fixture,
        width,
        height,
        |buffer| stucki_in_place(buffer, mode_gray_1()),
    );
    bench_gray_case(
        &mut group,
        "sierra_gray1_512",
        &fixture,
        width,
        height,
        |buffer| sierra_in_place(buffer, mode_gray_1()),
    );
    bench_gray_case(
        &mut group,
        "two_row_sierra_gray1_512",
        &fixture,
        width,
        height,
        |buffer| two_row_sierra_in_place(buffer, mode_gray_1()),
    );
    bench_gray_case(
        &mut group,
        "stevenson_arce_gray1_512",
        &fixture,
        width,
        height,
        |buffer| stevenson_arce_in_place(buffer, mode_gray_1()),
    );
    group.finish();

    let mut group = c.benchmark_group("diffusion_gray_512_extended");
    group.sample_size(20);
    set_gray_throughput(&mut group, width, height);

    bench_gray_case(
        &mut group,
        "fan_gray1_512",
        &fixture,
        width,
        height,
        |buffer| fan_in_place(buffer, mode_gray_1()),
    );
    bench_gray_case(
        &mut group,
        "shiau_fan_gray1_512",
        &fixture,
        width,
        height,
        |buffer| shiau_fan_in_place(buffer, mode_gray_1()),
    );
    bench_gray_case(
        &mut group,
        "shiau_fan_2_gray1_512",
        &fixture,
        width,
        height,
        |buffer| shiau_fan_2_in_place(buffer, mode_gray_1()),
    );
    group.finish();

    let mut group = c.benchmark_group("diffusion_gray_512_variable");
    group.sample_size(20);
    set_gray_throughput(&mut group, width, height);

    bench_gray_case(
        &mut group,
        "ostromoukhov_gray1_512",
        &fixture,
        width,
        height,
        |buffer| ostromoukhov_in_place(buffer, mode_gray_1()),
    );
    bench_gray_case(
        &mut group,
        "zhou_fang_gray1_512",
        &fixture,
        width,
        height,
        |buffer| zhou_fang_in_place(buffer, mode_gray_1()),
    );
    bench_gray_case(
        &mut group,
        "gradient_based_error_diffusion_gray1_512",
        &fixture,
        width,
        height,
        |buffer| gradient_based_error_diffusion_in_place(buffer, mode_gray_1()),
    );
    group.finish();

    let width = 256;
    let height = 256;
    let fixture = rgb_gradient(width, height);

    let mut group = c.benchmark_group("diffusion_rgb_256_representative");
    group.sample_size(20);
    set_rgb_throughput(&mut group, width, height);

    bench_rgb_case(
        &mut group,
        "floyd_steinberg_rgb_palette_cga_256",
        &fixture,
        width,
        height,
        |buffer| floyd_steinberg_in_place(buffer, mode_palette_cga()),
    );
    bench_rgb_case(
        &mut group,
        "burkes_rgb_palette_cga_256",
        &fixture,
        width,
        height,
        |buffer| burkes_in_place(buffer, mode_palette_cga()),
    );
    bench_rgb_case(
        &mut group,
        "sierra_lite_rgb_palette_cga_256",
        &fixture,
        width,
        height,
        |buffer| sierra_lite_in_place(buffer, mode_palette_cga()),
    );
    bench_rgb_case(
        &mut group,
        "fan_rgb_palette_cga_256",
        &fixture,
        width,
        height,
        |buffer| fan_in_place(buffer, mode_palette_cga()),
    );
    bench_rgb_case(
        &mut group,
        "shiau_fan_rgb_palette_cga_256",
        &fixture,
        width,
        height,
        |buffer| shiau_fan_in_place(buffer, mode_palette_cga()),
    );
    group.finish();

    let width = 256;
    let height = 256;
    let fixture_u16 = gray_ramp_u16(width, height);
    let mut group = c.benchmark_group("diffusion_gray_u16_256");
    group.sample_size(16);
    set_gray_throughput(&mut group, width, height);
    bench_gray_case_u16(
        &mut group,
        "floyd_steinberg_gray_u16_levels2_256",
        &fixture_u16,
        width,
        height,
        |buffer| floyd_steinberg_in_place(buffer, mode_gray_levels2_u16()),
    );
    bench_gray_case_u16(
        &mut group,
        "burkes_gray_u16_levels2_256",
        &fixture_u16,
        width,
        height,
        |buffer| burkes_in_place(buffer, mode_gray_levels2_u16()),
    );
    group.finish();
}

criterion_group!(benches, bench_diffusion);
criterion_main!(benches);

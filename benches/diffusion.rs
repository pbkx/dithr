mod common;

use common::{
    gray_buffer, gray_ramp, mode_gray_1, mode_palette_cga, rgb_buffer, rgb_gradient, touch_common,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
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

    group.bench_function("floyd_steinberg_gray1_512", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            floyd_steinberg_in_place(&mut buffer, mode_gray_1())
                .expect("floyd-steinberg should succeed");
            black_box(data);
        });
    });

    group.bench_function("false_floyd_steinberg_gray1_512", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            false_floyd_steinberg_in_place(&mut buffer, mode_gray_1())
                .expect("false floyd-steinberg should succeed");
            black_box(data);
        });
    });

    group.bench_function("burkes_gray1_512", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            burkes_in_place(&mut buffer, mode_gray_1()).expect("burkes should succeed");
            black_box(data);
        });
    });

    group.bench_function("sierra_lite_gray1_512", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            sierra_lite_in_place(&mut buffer, mode_gray_1()).expect("sierra lite should succeed");
            black_box(data);
        });
    });

    group.bench_function("atkinson_gray1_512", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            atkinson_in_place(&mut buffer, mode_gray_1()).expect("atkinson should succeed");
            black_box(data);
        });
    });
    group.finish();

    let mut group = c.benchmark_group("diffusion_gray_512_classic_largekernels");
    group.sample_size(20);

    group.bench_function("jarvis_judice_ninke_gray1_512", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            jarvis_judice_ninke_in_place(&mut buffer, mode_gray_1())
                .expect("jarvis-judice-ninke should succeed");
            black_box(data);
        });
    });

    group.bench_function("stucki_gray1_512", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            stucki_in_place(&mut buffer, mode_gray_1()).expect("stucki should succeed");
            black_box(data);
        });
    });

    group.bench_function("sierra_gray1_512", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            sierra_in_place(&mut buffer, mode_gray_1()).expect("sierra should succeed");
            black_box(data);
        });
    });

    group.bench_function("two_row_sierra_gray1_512", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            two_row_sierra_in_place(&mut buffer, mode_gray_1())
                .expect("two-row sierra should succeed");
            black_box(data);
        });
    });

    group.bench_function("stevenson_arce_gray1_512", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            stevenson_arce_in_place(&mut buffer, mode_gray_1())
                .expect("stevenson-arce should succeed");
            black_box(data);
        });
    });
    group.finish();

    let mut group = c.benchmark_group("diffusion_gray_512_extended");
    group.sample_size(20);

    group.bench_function("fan_gray1_512", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            fan_in_place(&mut buffer, mode_gray_1()).expect("fan should succeed");
            black_box(data);
        });
    });

    group.bench_function("shiau_fan_gray1_512", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            shiau_fan_in_place(&mut buffer, mode_gray_1()).expect("shiau-fan should succeed");
            black_box(data);
        });
    });

    group.bench_function("shiau_fan_2_gray1_512", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            shiau_fan_2_in_place(&mut buffer, mode_gray_1()).expect("shiau-fan-2 should succeed");
            black_box(data);
        });
    });
    group.finish();

    let mut group = c.benchmark_group("diffusion_gray_512_variable");
    group.sample_size(20);

    group.bench_function("ostromoukhov_gray1_512", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            ostromoukhov_in_place(&mut buffer, mode_gray_1()).expect("ostromoukhov should succeed");
            black_box(data);
        });
    });

    group.bench_function("zhou_fang_gray1_512", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            zhou_fang_in_place(&mut buffer, mode_gray_1()).expect("zhou-fang should succeed");
            black_box(data);
        });
    });

    group.bench_function("gradient_based_error_diffusion_gray1_512", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            gradient_based_error_diffusion_in_place(&mut buffer, mode_gray_1())
                .expect("gradient-based diffusion should succeed");
            black_box(data);
        });
    });
    group.finish();

    let width = 256;
    let height = 256;
    let fixture = rgb_gradient(width, height);

    let mut group = c.benchmark_group("diffusion_rgb_256_representative");
    group.sample_size(20);

    group.bench_function("floyd_steinberg_rgb_palette_cga_256", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = rgb_buffer(&mut data, width, height);
            floyd_steinberg_in_place(&mut buffer, mode_palette_cga())
                .expect("floyd-steinberg should succeed");
            black_box(data);
        });
    });

    group.bench_function("burkes_rgb_palette_cga_256", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = rgb_buffer(&mut data, width, height);
            burkes_in_place(&mut buffer, mode_palette_cga()).expect("burkes should succeed");
            black_box(data);
        });
    });

    group.bench_function("sierra_lite_rgb_palette_cga_256", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = rgb_buffer(&mut data, width, height);
            sierra_lite_in_place(&mut buffer, mode_palette_cga())
                .expect("sierra lite should succeed");
            black_box(data);
        });
    });

    group.bench_function("fan_rgb_palette_cga_256", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = rgb_buffer(&mut data, width, height);
            fan_in_place(&mut buffer, mode_palette_cga()).expect("fan should succeed");
            black_box(data);
        });
    });

    group.bench_function("shiau_fan_rgb_palette_cga_256", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = rgb_buffer(&mut data, width, height);
            shiau_fan_in_place(&mut buffer, mode_palette_cga()).expect("shiau-fan should succeed");
            black_box(data);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_diffusion);
criterion_main!(benches);

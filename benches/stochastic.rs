mod common;

use common::{
    gray_buffer, gray_ramp, mode_gray_1, mode_palette_bw, mode_palette_cga, rgb_buffer,
    rgb_gradient, touch_common,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dithr::{random_in_place, threshold_in_place};

fn bench_stochastic(c: &mut Criterion) {
    touch_common();

    let mut group = c.benchmark_group("stochastic_gray_1024");
    group.sample_size(20);
    let width = 1024;
    let height = 1024;
    let fixture = gray_ramp(width, height);

    group.bench_function("threshold_in_place_gray1_t127", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            threshold_in_place(&mut buffer, mode_gray_1(), 127);
            black_box(data);
        });
    });

    group.bench_function("random_in_place_gray1_seed1_strength32", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            random_in_place(&mut buffer, mode_gray_1(), 1, 32);
            black_box(data);
        });
    });
    group.finish();

    let mut group = c.benchmark_group("stochastic_gray_1024_palette_bw");
    group.sample_size(20);
    let fixture = gray_ramp(width, height);

    group.bench_function("threshold_in_place_palette_bw_t127", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            threshold_in_place(&mut buffer, mode_palette_bw(), 127);
            black_box(data);
        });
    });

    group.bench_function("random_in_place_palette_bw_seed1_strength32", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            random_in_place(&mut buffer, mode_palette_bw(), 1, 32);
            black_box(data);
        });
    });
    group.finish();

    let mut group = c.benchmark_group("stochastic_rgb_512");
    group.sample_size(20);
    let width = 512;
    let height = 512;
    let fixture = rgb_gradient(width, height);

    group.bench_function("threshold_in_place_rgb_palette_cga_t127", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = rgb_buffer(&mut data, width, height);
            threshold_in_place(&mut buffer, mode_palette_cga(), 127);
            black_box(data);
        });
    });

    group.bench_function("random_in_place_rgb_palette_cga_seed1_strength32", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = rgb_buffer(&mut data, width, height);
            random_in_place(&mut buffer, mode_palette_cga(), 1, 32);
            black_box(data);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_stochastic);
criterion_main!(benches);

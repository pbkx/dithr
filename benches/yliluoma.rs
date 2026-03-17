mod common;

use common::{palette_16_gray, palette_cga, rgb_buffer, rgb_gradient, touch_common};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dithr::{yliluoma_1_in_place, yliluoma_2_in_place, yliluoma_3_in_place};

fn bench_yliluoma(c: &mut Criterion) {
    touch_common();

    let width = 256;
    let height = 256;
    let fixture = rgb_gradient(width, height);

    let mut group = c.benchmark_group("yliluoma_rgb_256_cga");
    group.sample_size(12);
    let palette = palette_cga();

    group.bench_function("yliluoma_1_rgb256_cga", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = rgb_buffer(&mut data, width, height);
            yliluoma_1_in_place(&mut buffer, &palette).expect("yliluoma 1 should succeed");
            black_box(data);
        });
    });

    group.bench_function("yliluoma_2_rgb256_cga", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = rgb_buffer(&mut data, width, height);
            yliluoma_2_in_place(&mut buffer, &palette).expect("yliluoma 2 should succeed");
            black_box(data);
        });
    });

    group.bench_function("yliluoma_3_rgb256_cga", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = rgb_buffer(&mut data, width, height);
            yliluoma_3_in_place(&mut buffer, &palette).expect("yliluoma 3 should succeed");
            black_box(data);
        });
    });
    group.finish();

    let mut group = c.benchmark_group("yliluoma_rgb_256_gray16");
    group.sample_size(12);
    let palette = palette_16_gray();

    group.bench_function("yliluoma_1_rgb256_gray16", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = rgb_buffer(&mut data, width, height);
            yliluoma_1_in_place(&mut buffer, &palette).expect("yliluoma 1 should succeed");
            black_box(data);
        });
    });

    group.bench_function("yliluoma_2_rgb256_gray16", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = rgb_buffer(&mut data, width, height);
            yliluoma_2_in_place(&mut buffer, &palette).expect("yliluoma 2 should succeed");
            black_box(data);
        });
    });

    group.bench_function("yliluoma_3_rgb256_gray16", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = rgb_buffer(&mut data, width, height);
            yliluoma_3_in_place(&mut buffer, &palette).expect("yliluoma 3 should succeed");
            black_box(data);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_yliluoma);
criterion_main!(benches);

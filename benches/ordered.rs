mod common;

use common::{
    gray_buffer, gray_ramp, mode_gray_1, mode_palette_cga, mode_palette_gray4, rgb_buffer,
    rgb_gradient, touch_common, CUSTOM_2X2_MAP,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dithr::{
    bayer_16x16_in_place, bayer_2x2_in_place, bayer_4x4_in_place, bayer_8x8_in_place,
    cluster_dot_4x4_in_place, cluster_dot_8x8_in_place, custom_ordered_in_place,
};

fn bench_ordered(c: &mut Criterion) {
    touch_common();

    let mut group = c.benchmark_group("ordered_gray_1024");
    group.sample_size(20);
    let width = 1024;
    let height = 1024;
    let fixture = gray_ramp(width, height);

    group.bench_function("bayer_2x2_gray1", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            bayer_2x2_in_place(&mut buffer, mode_gray_1()).expect("bayer 2x2 should succeed");
            black_box(data);
        });
    });

    group.bench_function("bayer_4x4_gray1", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            bayer_4x4_in_place(&mut buffer, mode_gray_1()).expect("bayer 4x4 should succeed");
            black_box(data);
        });
    });

    group.bench_function("bayer_8x8_gray1", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            bayer_8x8_in_place(&mut buffer, mode_gray_1()).expect("bayer 8x8 should succeed");
            black_box(data);
        });
    });

    group.bench_function("bayer_16x16_gray1", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            bayer_16x16_in_place(&mut buffer, mode_gray_1()).expect("bayer 16x16 should succeed");
            black_box(data);
        });
    });

    group.bench_function("cluster_dot_4x4_gray1", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            cluster_dot_4x4_in_place(&mut buffer, mode_gray_1())
                .expect("cluster-dot 4x4 should succeed");
            black_box(data);
        });
    });

    group.bench_function("cluster_dot_8x8_gray1", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            cluster_dot_8x8_in_place(&mut buffer, mode_gray_1())
                .expect("cluster-dot 8x8 should succeed");
            black_box(data);
        });
    });

    group.bench_function("custom_ordered_2x2_gray1", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            custom_ordered_in_place(&mut buffer, mode_gray_1(), &CUSTOM_2X2_MAP, 2, 2, 64)
                .expect("custom ordered should succeed");
            black_box(data);
        });
    });
    group.finish();

    let mut group = c.benchmark_group("ordered_gray_palette_1024");
    group.sample_size(20);
    let fixture = gray_ramp(width, height);

    group.bench_function("bayer_8x8_palette_gray4", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            bayer_8x8_in_place(&mut buffer, mode_palette_gray4())
                .expect("bayer 8x8 should succeed");
            black_box(data);
        });
    });

    group.bench_function("cluster_dot_8x8_palette_gray4", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            cluster_dot_8x8_in_place(&mut buffer, mode_palette_gray4())
                .expect("cluster-dot 8x8 should succeed");
            black_box(data);
        });
    });

    group.bench_function("custom_ordered_2x2_palette_gray4", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            custom_ordered_in_place(&mut buffer, mode_palette_gray4(), &CUSTOM_2X2_MAP, 2, 2, 64)
                .expect("custom ordered should succeed");
            black_box(data);
        });
    });
    group.finish();

    let mut group = c.benchmark_group("ordered_rgb_512");
    group.sample_size(20);
    let width = 512;
    let height = 512;
    let fixture = rgb_gradient(width, height);

    group.bench_function("bayer_4x4_rgb_palette_cga", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = rgb_buffer(&mut data, width, height);
            bayer_4x4_in_place(&mut buffer, mode_palette_cga()).expect("bayer 4x4 should succeed");
            black_box(data);
        });
    });

    group.bench_function("bayer_8x8_rgb_palette_cga", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = rgb_buffer(&mut data, width, height);
            bayer_8x8_in_place(&mut buffer, mode_palette_cga()).expect("bayer 8x8 should succeed");
            black_box(data);
        });
    });

    group.bench_function("cluster_dot_8x8_rgb_palette_cga", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = rgb_buffer(&mut data, width, height);
            cluster_dot_8x8_in_place(&mut buffer, mode_palette_cga())
                .expect("cluster-dot 8x8 should succeed");
            black_box(data);
        });
    });

    group.bench_function("custom_ordered_2x2_rgb_palette_cga", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = rgb_buffer(&mut data, width, height);
            custom_ordered_in_place(&mut buffer, mode_palette_cga(), &CUSTOM_2X2_MAP, 2, 2, 64)
                .expect("custom ordered should succeed");
            black_box(data);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_ordered);
criterion_main!(benches);

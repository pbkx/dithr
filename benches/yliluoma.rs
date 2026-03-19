mod common;

use common::{
    bench_rgb_case, palette_16_gray, palette_cga, rgb_gradient, set_rgb_throughput, touch_common,
};
use criterion::{criterion_group, criterion_main, Criterion};
use dithr::{yliluoma_1_in_place, yliluoma_2_in_place, yliluoma_3_in_place};

fn bench_yliluoma(c: &mut Criterion) {
    touch_common();

    let width = 256;
    let height = 256;
    let fixture = rgb_gradient(width, height);

    let mut group = c.benchmark_group("yliluoma_rgb_256_cga");
    group.sample_size(12);
    set_rgb_throughput(&mut group, width, height);
    let palette = palette_cga();

    bench_rgb_case(
        &mut group,
        "yliluoma_1_rgb_256_cga",
        &fixture,
        width,
        height,
        |buffer| yliluoma_1_in_place(buffer, &palette),
    );
    bench_rgb_case(
        &mut group,
        "yliluoma_2_rgb_256_cga",
        &fixture,
        width,
        height,
        |buffer| yliluoma_2_in_place(buffer, &palette),
    );
    bench_rgb_case(
        &mut group,
        "yliluoma_3_rgb_256_cga",
        &fixture,
        width,
        height,
        |buffer| yliluoma_3_in_place(buffer, &palette),
    );
    group.finish();

    let mut group = c.benchmark_group("yliluoma_rgb_256_gray16");
    group.sample_size(12);
    set_rgb_throughput(&mut group, width, height);
    let palette = palette_16_gray();

    bench_rgb_case(
        &mut group,
        "yliluoma_1_rgb_256_gray16",
        &fixture,
        width,
        height,
        |buffer| yliluoma_1_in_place(buffer, &palette),
    );
    bench_rgb_case(
        &mut group,
        "yliluoma_2_rgb_256_gray16",
        &fixture,
        width,
        height,
        |buffer| yliluoma_2_in_place(buffer, &palette),
    );
    bench_rgb_case(
        &mut group,
        "yliluoma_3_rgb_256_gray16",
        &fixture,
        width,
        height,
        |buffer| yliluoma_3_in_place(buffer, &palette),
    );
    group.finish();
}

criterion_group!(benches, bench_yliluoma);
criterion_main!(benches);

mod common;

use common::{
    bench_rgb_case, palette_16_gray, palette_cga, rgb_gradient, set_rgb_throughput, touch_common,
};
use criterion::{criterion_group, criterion_main, Criterion};
use dithr::ordered::{yliluoma_1_in_place, yliluoma_2_in_place, yliluoma_3_in_place};
#[cfg(feature = "rayon")]
use dithr::ordered::{yliluoma_1_in_place_par, yliluoma_2_in_place_par, yliluoma_3_in_place_par};

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

    #[cfg(feature = "rayon")]
    {
        let width = 256;
        let height = 256;
        let fixture = rgb_gradient(width, height);
        let palette = palette_cga();

        let mut seq_1 = fixture.clone();
        let mut par_1 = fixture.clone();
        let mut seq_1_buffer = dithr::rgb_u8(&mut seq_1, width, height, width * 3)
            .expect("valid buffer should construct");
        let mut par_1_buffer = dithr::rgb_u8(&mut par_1, width, height, width * 3)
            .expect("valid buffer should construct");
        yliluoma_1_in_place(&mut seq_1_buffer, &palette).expect("sequential sanity should succeed");
        yliluoma_1_in_place_par(&mut par_1_buffer, &palette)
            .expect("parallel sanity should succeed");
        assert_eq!(seq_1, par_1);

        let mut seq_2 = fixture.clone();
        let mut par_2 = fixture.clone();
        let mut seq_2_buffer = dithr::rgb_u8(&mut seq_2, width, height, width * 3)
            .expect("valid buffer should construct");
        let mut par_2_buffer = dithr::rgb_u8(&mut par_2, width, height, width * 3)
            .expect("valid buffer should construct");
        yliluoma_2_in_place(&mut seq_2_buffer, &palette).expect("sequential sanity should succeed");
        yliluoma_2_in_place_par(&mut par_2_buffer, &palette)
            .expect("parallel sanity should succeed");
        assert_eq!(seq_2, par_2);

        let mut seq_3 = fixture.clone();
        let mut par_3 = fixture.clone();
        let mut seq_3_buffer = dithr::rgb_u8(&mut seq_3, width, height, width * 3)
            .expect("valid buffer should construct");
        let mut par_3_buffer = dithr::rgb_u8(&mut par_3, width, height, width * 3)
            .expect("valid buffer should construct");
        yliluoma_3_in_place(&mut seq_3_buffer, &palette).expect("sequential sanity should succeed");
        yliluoma_3_in_place_par(&mut par_3_buffer, &palette)
            .expect("parallel sanity should succeed");
        assert_eq!(seq_3, par_3);

        let mut group = c.benchmark_group("yliluoma_parallel_rgb_256_cga");
        group.sample_size(12);
        set_rgb_throughput(&mut group, width, height);

        bench_rgb_case(
            &mut group,
            "yliluoma_1_seq_rgb_256_cga",
            &fixture,
            width,
            height,
            |buffer| yliluoma_1_in_place(buffer, &palette),
        );
        bench_rgb_case(
            &mut group,
            "yliluoma_1_par_rgb_256_cga",
            &fixture,
            width,
            height,
            |buffer| yliluoma_1_in_place_par(buffer, &palette),
        );
        bench_rgb_case(
            &mut group,
            "yliluoma_2_seq_rgb_256_cga",
            &fixture,
            width,
            height,
            |buffer| yliluoma_2_in_place(buffer, &palette),
        );
        bench_rgb_case(
            &mut group,
            "yliluoma_2_par_rgb_256_cga",
            &fixture,
            width,
            height,
            |buffer| yliluoma_2_in_place_par(buffer, &palette),
        );
        bench_rgb_case(
            &mut group,
            "yliluoma_3_seq_rgb_256_cga",
            &fixture,
            width,
            height,
            |buffer| yliluoma_3_in_place(buffer, &palette),
        );
        bench_rgb_case(
            &mut group,
            "yliluoma_3_par_rgb_256_cga",
            &fixture,
            width,
            height,
            |buffer| yliluoma_3_in_place_par(buffer, &palette),
        );
        group.finish();
    }
}

criterion_group!(benches, bench_yliluoma);
criterion_main!(benches);

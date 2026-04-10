mod common;

#[cfg(feature = "rayon")]
use common::assert_gray_seq_par_equal;
use common::{
    bench_gray_case, bench_gray_case_f32, bench_gray_case_u16, bench_rgb_case, gray_ramp,
    gray_ramp_f32, gray_ramp_u16, mode_gray_1, mode_gray_levels2_u16, mode_palette_cga,
    mode_palette_gray4, rgb_gradient, set_gray_throughput, set_rgb_throughput, touch_common,
    CUSTOM_2X2_MAP,
};
use criterion::{criterion_group, criterion_main, Criterion};
use dithr::ordered::{
    bayer_16x16_in_place, bayer_2x2_in_place, bayer_4x4_in_place, bayer_8x8_in_place,
    cluster_dot_4x4_in_place, cluster_dot_8x8_in_place, custom_ordered_in_place,
    void_and_cluster_in_place,
};
use dithr::QuantizeMode;
#[cfg(feature = "rayon")]
use dithr::ordered::{
    bayer_16x16_in_place_par, bayer_2x2_in_place_par, bayer_4x4_in_place_par,
    bayer_8x8_in_place_par, cluster_dot_4x4_in_place_par, cluster_dot_8x8_in_place_par,
    custom_ordered_in_place_par,
};

fn bench_ordered(c: &mut Criterion) {
    touch_common();

    let width = 1024;
    let height = 1024;
    let fixture = gray_ramp(width, height);

    let mut group = c.benchmark_group("ordered_gray_1024");
    group.sample_size(24);
    set_gray_throughput(&mut group, width, height);

    bench_gray_case(
        &mut group,
        "bayer_2x2_gray1_1024",
        &fixture,
        width,
        height,
        |buffer| bayer_2x2_in_place(buffer, mode_gray_1()),
    );
    bench_gray_case(
        &mut group,
        "bayer_4x4_gray1_1024",
        &fixture,
        width,
        height,
        |buffer| bayer_4x4_in_place(buffer, mode_gray_1()),
    );
    bench_gray_case(
        &mut group,
        "bayer_8x8_gray1_1024",
        &fixture,
        width,
        height,
        |buffer| bayer_8x8_in_place(buffer, mode_gray_1()),
    );
    bench_gray_case(
        &mut group,
        "bayer_16x16_gray1_1024",
        &fixture,
        width,
        height,
        |buffer| bayer_16x16_in_place(buffer, mode_gray_1()),
    );
    bench_gray_case(
        &mut group,
        "cluster_dot_4x4_gray1_1024",
        &fixture,
        width,
        height,
        |buffer| cluster_dot_4x4_in_place(buffer, mode_gray_1()),
    );
    bench_gray_case(
        &mut group,
        "cluster_dot_8x8_gray1_1024",
        &fixture,
        width,
        height,
        |buffer| cluster_dot_8x8_in_place(buffer, mode_gray_1()),
    );
    bench_gray_case(
        &mut group,
        "void_and_cluster_gray1_1024",
        &fixture,
        width,
        height,
        |buffer| void_and_cluster_in_place(buffer, mode_gray_1()),
    );
    bench_gray_case(
        &mut group,
        "custom_ordered_2x2_gray1_1024",
        &fixture,
        width,
        height,
        |buffer| custom_ordered_in_place(buffer, mode_gray_1(), &CUSTOM_2X2_MAP, 2, 2, 64),
    );
    group.finish();

    let width = 512;
    let height = 512;
    let fixture_u16 = gray_ramp_u16(width, height);

    let mut group = c.benchmark_group("ordered_gray_u16_512");
    group.sample_size(16);
    set_gray_throughput(&mut group, width, height);
    bench_gray_case_u16(
        &mut group,
        "bayer_8x8_gray_u16_levels2_512",
        &fixture_u16,
        width,
        height,
        |buffer| bayer_8x8_in_place(buffer, mode_gray_levels2_u16()),
    );
    group.finish();

    let fixture_f32 = gray_ramp_f32(width, height);
    let mut group = c.benchmark_group("ordered_gray_f32_512");
    group.sample_size(16);
    set_gray_throughput(&mut group, width, height);
    bench_gray_case_f32(
        &mut group,
        "bayer_8x8_gray_f32_levels2_512",
        &fixture_f32,
        width,
        height,
        |buffer| bayer_8x8_in_place(buffer, QuantizeMode::GrayLevels(2)),
    );
    group.finish();

    let mut group = c.benchmark_group("ordered_gray_palette_1024");
    group.sample_size(20);
    set_gray_throughput(&mut group, width, height);

    bench_gray_case(
        &mut group,
        "bayer_8x8_palette_gray4_1024",
        &fixture,
        width,
        height,
        |buffer| bayer_8x8_in_place(buffer, mode_palette_gray4()),
    );
    bench_gray_case(
        &mut group,
        "cluster_dot_8x8_palette_gray4_1024",
        &fixture,
        width,
        height,
        |buffer| cluster_dot_8x8_in_place(buffer, mode_palette_gray4()),
    );
    bench_gray_case(
        &mut group,
        "custom_ordered_2x2_palette_gray4_1024",
        &fixture,
        width,
        height,
        |buffer| custom_ordered_in_place(buffer, mode_palette_gray4(), &CUSTOM_2X2_MAP, 2, 2, 64),
    );
    group.finish();

    let width = 512;
    let height = 512;
    let fixture = rgb_gradient(width, height);

    let mut group = c.benchmark_group("ordered_rgb_512");
    group.sample_size(20);
    set_rgb_throughput(&mut group, width, height);

    bench_rgb_case(
        &mut group,
        "bayer_4x4_rgb_palette_cga_512",
        &fixture,
        width,
        height,
        |buffer| bayer_4x4_in_place(buffer, mode_palette_cga()),
    );
    bench_rgb_case(
        &mut group,
        "bayer_8x8_rgb_palette_cga_512",
        &fixture,
        width,
        height,
        |buffer| bayer_8x8_in_place(buffer, mode_palette_cga()),
    );
    bench_rgb_case(
        &mut group,
        "cluster_dot_8x8_rgb_palette_cga_512",
        &fixture,
        width,
        height,
        |buffer| cluster_dot_8x8_in_place(buffer, mode_palette_cga()),
    );
    bench_rgb_case(
        &mut group,
        "custom_ordered_2x2_rgb_palette_cga_512",
        &fixture,
        width,
        height,
        |buffer| custom_ordered_in_place(buffer, mode_palette_cga(), &CUSTOM_2X2_MAP, 2, 2, 64),
    );
    group.finish();

    #[cfg(feature = "rayon")]
    {
        let width = 512;
        let height = 512;
        let fixture = gray_ramp(width, height);

        assert_gray_seq_par_equal(
            &fixture,
            width,
            height,
            |buffer| bayer_2x2_in_place(buffer, mode_gray_1()),
            |buffer| bayer_2x2_in_place_par(buffer, mode_gray_1()),
        );
        assert_gray_seq_par_equal(
            &fixture,
            width,
            height,
            |buffer| bayer_4x4_in_place(buffer, mode_gray_1()),
            |buffer| bayer_4x4_in_place_par(buffer, mode_gray_1()),
        );
        assert_gray_seq_par_equal(
            &fixture,
            width,
            height,
            |buffer| bayer_8x8_in_place(buffer, mode_gray_1()),
            |buffer| bayer_8x8_in_place_par(buffer, mode_gray_1()),
        );
        assert_gray_seq_par_equal(
            &fixture,
            width,
            height,
            |buffer| bayer_16x16_in_place(buffer, mode_gray_1()),
            |buffer| bayer_16x16_in_place_par(buffer, mode_gray_1()),
        );
        assert_gray_seq_par_equal(
            &fixture,
            width,
            height,
            |buffer| cluster_dot_4x4_in_place(buffer, mode_gray_1()),
            |buffer| cluster_dot_4x4_in_place_par(buffer, mode_gray_1()),
        );
        assert_gray_seq_par_equal(
            &fixture,
            width,
            height,
            |buffer| cluster_dot_8x8_in_place(buffer, mode_gray_1()),
            |buffer| cluster_dot_8x8_in_place_par(buffer, mode_gray_1()),
        );
        assert_gray_seq_par_equal(
            &fixture,
            width,
            height,
            |buffer| custom_ordered_in_place(buffer, mode_gray_1(), &CUSTOM_2X2_MAP, 2, 2, 64),
            |buffer| custom_ordered_in_place_par(buffer, mode_gray_1(), &CUSTOM_2X2_MAP, 2, 2, 64),
        );

        let mut group = c.benchmark_group("ordered_parallel_gray_512");
        group.sample_size(18);
        set_gray_throughput(&mut group, width, height);

        bench_gray_case(
            &mut group,
            "bayer_2x2_seq_gray1_512",
            &fixture,
            width,
            height,
            |buffer| bayer_2x2_in_place(buffer, mode_gray_1()),
        );
        bench_gray_case(
            &mut group,
            "bayer_2x2_par_gray1_512",
            &fixture,
            width,
            height,
            |buffer| bayer_2x2_in_place_par(buffer, mode_gray_1()),
        );
        bench_gray_case(
            &mut group,
            "bayer_4x4_seq_gray1_512",
            &fixture,
            width,
            height,
            |buffer| bayer_4x4_in_place(buffer, mode_gray_1()),
        );
        bench_gray_case(
            &mut group,
            "bayer_4x4_par_gray1_512",
            &fixture,
            width,
            height,
            |buffer| bayer_4x4_in_place_par(buffer, mode_gray_1()),
        );
        bench_gray_case(
            &mut group,
            "bayer_8x8_seq_gray1_512",
            &fixture,
            width,
            height,
            |buffer| bayer_8x8_in_place(buffer, mode_gray_1()),
        );
        bench_gray_case(
            &mut group,
            "bayer_8x8_par_gray1_512",
            &fixture,
            width,
            height,
            |buffer| bayer_8x8_in_place_par(buffer, mode_gray_1()),
        );
        bench_gray_case(
            &mut group,
            "bayer_16x16_seq_gray1_512",
            &fixture,
            width,
            height,
            |buffer| bayer_16x16_in_place(buffer, mode_gray_1()),
        );
        bench_gray_case(
            &mut group,
            "bayer_16x16_par_gray1_512",
            &fixture,
            width,
            height,
            |buffer| bayer_16x16_in_place_par(buffer, mode_gray_1()),
        );
        bench_gray_case(
            &mut group,
            "cluster_dot_4x4_seq_gray1_512",
            &fixture,
            width,
            height,
            |buffer| cluster_dot_4x4_in_place(buffer, mode_gray_1()),
        );
        bench_gray_case(
            &mut group,
            "cluster_dot_4x4_par_gray1_512",
            &fixture,
            width,
            height,
            |buffer| cluster_dot_4x4_in_place_par(buffer, mode_gray_1()),
        );
        bench_gray_case(
            &mut group,
            "cluster_dot_8x8_seq_gray1_512",
            &fixture,
            width,
            height,
            |buffer| cluster_dot_8x8_in_place(buffer, mode_gray_1()),
        );
        bench_gray_case(
            &mut group,
            "cluster_dot_8x8_par_gray1_512",
            &fixture,
            width,
            height,
            |buffer| cluster_dot_8x8_in_place_par(buffer, mode_gray_1()),
        );
        bench_gray_case(
            &mut group,
            "custom_ordered_seq_gray1_512",
            &fixture,
            width,
            height,
            |buffer| custom_ordered_in_place(buffer, mode_gray_1(), &CUSTOM_2X2_MAP, 2, 2, 64),
        );
        bench_gray_case(
            &mut group,
            "custom_ordered_par_gray1_512",
            &fixture,
            width,
            height,
            |buffer| custom_ordered_in_place_par(buffer, mode_gray_1(), &CUSTOM_2X2_MAP, 2, 2, 64),
        );
        group.finish();
    }
}

criterion_group!(benches, bench_ordered);
criterion_main!(benches);

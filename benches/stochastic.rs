mod common;

#[cfg(feature = "rayon")]
use common::assert_gray_seq_par_equal;
use common::{
    bench_gray_case, bench_rgb_case, gray_ramp, mode_gray_1, mode_palette_bw, mode_palette_cga,
    rgb_gradient, set_gray_throughput, set_rgb_throughput, touch_common,
};
use criterion::{criterion_group, criterion_main, Criterion};
use dithr::{
    random_binary_in_place, random_in_place, threshold_binary_in_place, threshold_in_place,
};
#[cfg(feature = "rayon")]
use dithr::{random_binary_in_place_par, threshold_binary_in_place_par};

fn bench_stochastic(c: &mut Criterion) {
    touch_common();

    let width = 1024;
    let height = 1024;
    let fixture = gray_ramp(width, height);

    let mut group = c.benchmark_group("stochastic_binary_gray_1024");
    group.sample_size(24);
    set_gray_throughput(&mut group, width, height);

    bench_gray_case(
        &mut group,
        "threshold_binary_gray1_t127",
        &fixture,
        width,
        height,
        |buffer| threshold_binary_in_place(buffer, mode_gray_1(), 127),
    );
    bench_gray_case(
        &mut group,
        "random_binary_gray1_seed1_strength32",
        &fixture,
        width,
        height,
        |buffer| random_binary_in_place(buffer, mode_gray_1(), 1, 32),
    );
    group.finish();

    let mut group = c.benchmark_group("stochastic_binary_gray_palette_bw_1024");
    group.sample_size(24);
    set_gray_throughput(&mut group, width, height);

    bench_gray_case(
        &mut group,
        "threshold_binary_palette_bw_t127",
        &fixture,
        width,
        height,
        |buffer| threshold_binary_in_place(buffer, mode_palette_bw(), 127),
    );
    bench_gray_case(
        &mut group,
        "random_binary_palette_bw_seed1_strength32",
        &fixture,
        width,
        height,
        |buffer| random_binary_in_place(buffer, mode_palette_bw(), 1, 32),
    );
    group.finish();

    let mut group = c.benchmark_group("stochastic_api_compat_gray_1024");
    group.sample_size(24);
    set_gray_throughput(&mut group, width, height);

    bench_gray_case(
        &mut group,
        "threshold_wrapper_gray1_t127",
        &fixture,
        width,
        height,
        |buffer| threshold_in_place(buffer, mode_gray_1(), 127),
    );
    bench_gray_case(
        &mut group,
        "random_wrapper_gray1_seed1_strength32",
        &fixture,
        width,
        height,
        |buffer| random_in_place(buffer, mode_gray_1(), 1, 32),
    );
    group.finish();

    let width = 512;
    let height = 512;
    let fixture = rgb_gradient(width, height);

    let mut group = c.benchmark_group("stochastic_binary_rgb_512");
    group.sample_size(20);
    set_rgb_throughput(&mut group, width, height);

    bench_rgb_case(
        &mut group,
        "threshold_binary_rgb_palette_cga_t127",
        &fixture,
        width,
        height,
        |buffer| threshold_binary_in_place(buffer, mode_palette_cga(), 127),
    );
    bench_rgb_case(
        &mut group,
        "random_binary_rgb_palette_cga_seed1_strength32",
        &fixture,
        width,
        height,
        |buffer| random_binary_in_place(buffer, mode_palette_cga(), 1, 32),
    );
    group.finish();

    #[cfg(feature = "rayon")]
    {
        let width = 256;
        let height = 256;
        let fixture = gray_ramp(width, height);

        assert_gray_seq_par_equal(
            &fixture,
            width,
            height,
            |buffer| threshold_binary_in_place(buffer, mode_gray_1(), 127),
            |buffer| threshold_binary_in_place_par(buffer, mode_gray_1(), 127),
        );
        assert_gray_seq_par_equal(
            &fixture,
            width,
            height,
            |buffer| random_binary_in_place(buffer, mode_gray_1(), 7, 64),
            |buffer| random_binary_in_place_par(buffer, mode_gray_1(), 7, 64),
        );

        let mut group = c.benchmark_group("stochastic_parallel_gray_256");
        group.sample_size(18);
        set_gray_throughput(&mut group, width, height);

        bench_gray_case(
            &mut group,
            "threshold_binary_seq_gray1_256",
            &fixture,
            width,
            height,
            |buffer| threshold_binary_in_place(buffer, mode_gray_1(), 127),
        );
        bench_gray_case(
            &mut group,
            "threshold_binary_par_gray1_256",
            &fixture,
            width,
            height,
            |buffer| threshold_binary_in_place_par(buffer, mode_gray_1(), 127),
        );
        bench_gray_case(
            &mut group,
            "random_binary_seq_gray1_256",
            &fixture,
            width,
            height,
            |buffer| random_binary_in_place(buffer, mode_gray_1(), 7, 64),
        );
        bench_gray_case(
            &mut group,
            "random_binary_par_gray1_256",
            &fixture,
            width,
            height,
            |buffer| random_binary_in_place_par(buffer, mode_gray_1(), 7, 64),
        );
        group.finish();
    }
}

criterion_group!(benches, bench_stochastic);
criterion_main!(benches);

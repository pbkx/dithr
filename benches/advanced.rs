mod common;

use common::{
    gray_buffer, gray_ramp, mode_gray_1, mode_palette_cga, rgb_buffer, rgb_gradient, touch_common,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dithr::{
    direct_binary_search_in_place, electrostatic_halftoning_in_place, knuth_dot_diffusion_in_place,
    lattice_boltzmann_in_place, riemersma_in_place,
};

fn bench_advanced(c: &mut Criterion) {
    touch_common();

    let mut group = c.benchmark_group("advanced_gray_riemersma_dotdiff_256");
    group.sample_size(10);
    let width = 256;
    let height = 256;
    let fixture = gray_ramp(width, height);

    group.bench_function("riemersma_gray1_256", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            riemersma_in_place(&mut buffer, mode_gray_1()).expect("riemersma should succeed");
            black_box(data);
        });
    });

    group.bench_function("knuth_dot_diffusion_gray1_256", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            knuth_dot_diffusion_in_place(&mut buffer, mode_gray_1())
                .expect("knuth dot diffusion should succeed");
            black_box(data);
        });
    });
    group.finish();

    let mut group = c.benchmark_group("advanced_rgb_riemersma_dotdiff_256");
    group.sample_size(10);
    let width = 256;
    let height = 256;
    let fixture = rgb_gradient(width, height);

    group.bench_function("riemersma_rgb_palette_cga_256", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = rgb_buffer(&mut data, width, height);
            riemersma_in_place(&mut buffer, mode_palette_cga()).expect("riemersma should succeed");
            black_box(data);
        });
    });

    group.bench_function("knuth_dot_diffusion_rgb_palette_cga_256", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = rgb_buffer(&mut data, width, height);
            knuth_dot_diffusion_in_place(&mut buffer, mode_palette_cga())
                .expect("knuth dot diffusion should succeed");
            black_box(data);
        });
    });
    group.finish();

    let mut group = c.benchmark_group("advanced_gray_dbs_128");
    group.sample_size(10);
    let width = 128;
    let height = 128;
    let fixture = gray_ramp(width, height);

    group.bench_function("direct_binary_search_gray_128_iters8", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            direct_binary_search_in_place(&mut buffer, 8)
                .expect("direct binary search should succeed");
            black_box(data);
        });
    });

    group.bench_function("lattice_boltzmann_gray_128_steps8", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            lattice_boltzmann_in_place(&mut buffer, 8).expect("lattice-boltzmann should succeed");
            black_box(data);
        });
    });

    group.bench_function("electrostatic_halftoning_gray_128_steps8", |b| {
        b.iter(|| {
            let mut data = fixture.clone();
            let mut buffer = gray_buffer(&mut data, width, height);
            electrostatic_halftoning_in_place(&mut buffer, 8)
                .expect("electrostatic halftoning should succeed");
            black_box(data);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_advanced);
criterion_main!(benches);

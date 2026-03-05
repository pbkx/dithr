#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelTap {
    pub dx: i8,
    pub dy: i8,
    pub weight_num: i16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ErrorKernel {
    pub taps: &'static [KernelTap],
    pub weight_den: i16,
}

pub const FLOYD_STEINBERG: ErrorKernel = ErrorKernel {
    taps: &[
        KernelTap {
            dx: 1,
            dy: 0,
            weight_num: 7,
        },
        KernelTap {
            dx: -1,
            dy: 1,
            weight_num: 3,
        },
        KernelTap {
            dx: 0,
            dy: 1,
            weight_num: 5,
        },
        KernelTap {
            dx: 1,
            dy: 1,
            weight_num: 1,
        },
    ],
    weight_den: 16,
};

pub const FALSE_FLOYD_STEINBERG: ErrorKernel = ErrorKernel {
    taps: &[
        KernelTap {
            dx: 1,
            dy: 0,
            weight_num: 3,
        },
        KernelTap {
            dx: 0,
            dy: 1,
            weight_num: 3,
        },
        KernelTap {
            dx: 1,
            dy: 1,
            weight_num: 2,
        },
    ],
    weight_den: 8,
};

pub const JARVIS_JUDICE_NINKE: ErrorKernel = ErrorKernel {
    taps: &[
        KernelTap {
            dx: 1,
            dy: 0,
            weight_num: 7,
        },
        KernelTap {
            dx: 2,
            dy: 0,
            weight_num: 5,
        },
        KernelTap {
            dx: -2,
            dy: 1,
            weight_num: 3,
        },
        KernelTap {
            dx: -1,
            dy: 1,
            weight_num: 5,
        },
        KernelTap {
            dx: 0,
            dy: 1,
            weight_num: 7,
        },
        KernelTap {
            dx: 1,
            dy: 1,
            weight_num: 5,
        },
        KernelTap {
            dx: 2,
            dy: 1,
            weight_num: 3,
        },
        KernelTap {
            dx: -2,
            dy: 2,
            weight_num: 1,
        },
        KernelTap {
            dx: -1,
            dy: 2,
            weight_num: 3,
        },
        KernelTap {
            dx: 0,
            dy: 2,
            weight_num: 5,
        },
        KernelTap {
            dx: 1,
            dy: 2,
            weight_num: 3,
        },
        KernelTap {
            dx: 2,
            dy: 2,
            weight_num: 1,
        },
    ],
    weight_den: 48,
};

pub const STUCKI: ErrorKernel = ErrorKernel {
    taps: &[
        KernelTap {
            dx: 1,
            dy: 0,
            weight_num: 8,
        },
        KernelTap {
            dx: 2,
            dy: 0,
            weight_num: 4,
        },
        KernelTap {
            dx: -2,
            dy: 1,
            weight_num: 2,
        },
        KernelTap {
            dx: -1,
            dy: 1,
            weight_num: 4,
        },
        KernelTap {
            dx: 0,
            dy: 1,
            weight_num: 8,
        },
        KernelTap {
            dx: 1,
            dy: 1,
            weight_num: 4,
        },
        KernelTap {
            dx: 2,
            dy: 1,
            weight_num: 2,
        },
        KernelTap {
            dx: -2,
            dy: 2,
            weight_num: 1,
        },
        KernelTap {
            dx: -1,
            dy: 2,
            weight_num: 2,
        },
        KernelTap {
            dx: 0,
            dy: 2,
            weight_num: 4,
        },
        KernelTap {
            dx: 1,
            dy: 2,
            weight_num: 2,
        },
        KernelTap {
            dx: 2,
            dy: 2,
            weight_num: 1,
        },
    ],
    weight_den: 42,
};

pub const BURKES: ErrorKernel = ErrorKernel {
    taps: &[
        KernelTap {
            dx: 1,
            dy: 0,
            weight_num: 8,
        },
        KernelTap {
            dx: 2,
            dy: 0,
            weight_num: 4,
        },
        KernelTap {
            dx: -2,
            dy: 1,
            weight_num: 2,
        },
        KernelTap {
            dx: -1,
            dy: 1,
            weight_num: 4,
        },
        KernelTap {
            dx: 0,
            dy: 1,
            weight_num: 8,
        },
        KernelTap {
            dx: 1,
            dy: 1,
            weight_num: 4,
        },
        KernelTap {
            dx: 2,
            dy: 1,
            weight_num: 2,
        },
    ],
    weight_den: 32,
};

pub const SIERRA: ErrorKernel = ErrorKernel {
    taps: &[
        KernelTap {
            dx: 1,
            dy: 0,
            weight_num: 5,
        },
        KernelTap {
            dx: 2,
            dy: 0,
            weight_num: 3,
        },
        KernelTap {
            dx: -2,
            dy: 1,
            weight_num: 2,
        },
        KernelTap {
            dx: -1,
            dy: 1,
            weight_num: 4,
        },
        KernelTap {
            dx: 0,
            dy: 1,
            weight_num: 5,
        },
        KernelTap {
            dx: 1,
            dy: 1,
            weight_num: 4,
        },
        KernelTap {
            dx: 2,
            dy: 1,
            weight_num: 2,
        },
        KernelTap {
            dx: -1,
            dy: 2,
            weight_num: 2,
        },
        KernelTap {
            dx: 0,
            dy: 2,
            weight_num: 3,
        },
        KernelTap {
            dx: 1,
            dy: 2,
            weight_num: 2,
        },
    ],
    weight_den: 32,
};

pub const TWO_ROW_SIERRA: ErrorKernel = ErrorKernel {
    taps: &[
        KernelTap {
            dx: 1,
            dy: 0,
            weight_num: 4,
        },
        KernelTap {
            dx: 2,
            dy: 0,
            weight_num: 3,
        },
        KernelTap {
            dx: -2,
            dy: 1,
            weight_num: 1,
        },
        KernelTap {
            dx: -1,
            dy: 1,
            weight_num: 2,
        },
        KernelTap {
            dx: 0,
            dy: 1,
            weight_num: 3,
        },
        KernelTap {
            dx: 1,
            dy: 1,
            weight_num: 2,
        },
        KernelTap {
            dx: 2,
            dy: 1,
            weight_num: 1,
        },
    ],
    weight_den: 16,
};

pub const SIERRA_LITE: ErrorKernel = ErrorKernel {
    taps: &[
        KernelTap {
            dx: 1,
            dy: 0,
            weight_num: 2,
        },
        KernelTap {
            dx: -1,
            dy: 1,
            weight_num: 1,
        },
        KernelTap {
            dx: 0,
            dy: 1,
            weight_num: 1,
        },
    ],
    weight_den: 4,
};

pub const SIERRA_2_4A: ErrorKernel = ErrorKernel {
    taps: &[
        KernelTap {
            dx: 1,
            dy: 0,
            weight_num: 2,
        },
        KernelTap {
            dx: -1,
            dy: 1,
            weight_num: 1,
        },
        KernelTap {
            dx: 0,
            dy: 1,
            weight_num: 1,
        },
    ],
    weight_den: 4,
};

pub const STEVENSON_ARCE: ErrorKernel = ErrorKernel {
    taps: &[
        KernelTap {
            dx: 2,
            dy: 0,
            weight_num: 32,
        },
        KernelTap {
            dx: -3,
            dy: 1,
            weight_num: 12,
        },
        KernelTap {
            dx: -1,
            dy: 1,
            weight_num: 26,
        },
        KernelTap {
            dx: 1,
            dy: 1,
            weight_num: 30,
        },
        KernelTap {
            dx: 3,
            dy: 1,
            weight_num: 16,
        },
        KernelTap {
            dx: -2,
            dy: 2,
            weight_num: 12,
        },
        KernelTap {
            dx: 0,
            dy: 2,
            weight_num: 26,
        },
        KernelTap {
            dx: 2,
            dy: 2,
            weight_num: 12,
        },
        KernelTap {
            dx: -3,
            dy: 3,
            weight_num: 5,
        },
        KernelTap {
            dx: -1,
            dy: 3,
            weight_num: 12,
        },
        KernelTap {
            dx: 1,
            dy: 3,
            weight_num: 12,
        },
        KernelTap {
            dx: 3,
            dy: 3,
            weight_num: 5,
        },
    ],
    weight_den: 200,
};

pub const ATKINSON: ErrorKernel = ErrorKernel {
    taps: &[
        KernelTap {
            dx: 1,
            dy: 0,
            weight_num: 1,
        },
        KernelTap {
            dx: 2,
            dy: 0,
            weight_num: 1,
        },
        KernelTap {
            dx: -1,
            dy: 1,
            weight_num: 1,
        },
        KernelTap {
            dx: 0,
            dy: 1,
            weight_num: 1,
        },
        KernelTap {
            dx: 1,
            dy: 1,
            weight_num: 1,
        },
        KernelTap {
            dx: 0,
            dy: 2,
            weight_num: 1,
        },
    ],
    weight_den: 8,
};

pub const FAN: ErrorKernel = ErrorKernel {
    taps: &[
        KernelTap {
            dx: 1,
            dy: 0,
            weight_num: 7,
        },
        KernelTap {
            dx: -2,
            dy: 1,
            weight_num: 1,
        },
        KernelTap {
            dx: -1,
            dy: 1,
            weight_num: 3,
        },
        KernelTap {
            dx: 0,
            dy: 1,
            weight_num: 5,
        },
    ],
    weight_den: 16,
};

pub const SHIAU_FAN: ErrorKernel = ErrorKernel {
    taps: &[
        KernelTap {
            dx: 1,
            dy: 0,
            weight_num: 4,
        },
        KernelTap {
            dx: -2,
            dy: 1,
            weight_num: 1,
        },
        KernelTap {
            dx: -1,
            dy: 1,
            weight_num: 1,
        },
        KernelTap {
            dx: 0,
            dy: 1,
            weight_num: 2,
        },
    ],
    weight_den: 8,
};

pub const SHIAU_FAN_2: ErrorKernel = ErrorKernel {
    taps: &[
        KernelTap {
            dx: 1,
            dy: 0,
            weight_num: 8,
        },
        KernelTap {
            dx: -3,
            dy: 1,
            weight_num: 1,
        },
        KernelTap {
            dx: -2,
            dy: 1,
            weight_num: 1,
        },
        KernelTap {
            dx: -1,
            dy: 1,
            weight_num: 2,
        },
        KernelTap {
            dx: 0,
            dy: 1,
            weight_num: 4,
        },
    ],
    weight_den: 16,
};

#[cfg(test)]
mod tests {
    use super::{
        ErrorKernel, ATKINSON, BURKES, FALSE_FLOYD_STEINBERG, FAN, FLOYD_STEINBERG,
        JARVIS_JUDICE_NINKE, SHIAU_FAN, SHIAU_FAN_2, SIERRA, SIERRA_2_4A, SIERRA_LITE,
        STEVENSON_ARCE, STUCKI, TWO_ROW_SIERRA,
    };

    #[test]
    fn floyd_steinberg_denominator_positive() {
        let kernel = all_kernels()[0];
        assert!(kernel.weight_den > 0);
    }

    #[test]
    fn all_kernel_taps_point_forward_or_same_row_forward() {
        for kernel in all_kernels() {
            for tap in kernel.taps {
                assert!(tap.dy > 0 || (tap.dy == 0 && tap.dx > 0));
            }
        }
    }

    #[test]
    fn all_kernel_weights_nonnegative() {
        for kernel in all_kernels() {
            assert!(kernel.weight_den > 0);
            for tap in kernel.taps {
                assert!(tap.weight_num >= 0);
            }
        }
    }

    fn all_kernels() -> [&'static ErrorKernel; 14] {
        [
            &FLOYD_STEINBERG,
            &FALSE_FLOYD_STEINBERG,
            &JARVIS_JUDICE_NINKE,
            &STUCKI,
            &BURKES,
            &SIERRA,
            &TWO_ROW_SIERRA,
            &SIERRA_LITE,
            &SIERRA_2_4A,
            &STEVENSON_ARCE,
            &ATKINSON,
            &FAN,
            &SHIAU_FAN,
            &SHIAU_FAN_2,
        ]
    }
}

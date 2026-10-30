use eframe::egui::IconData;

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// Evaluates the StudioEQ Master Pro horizontal waveform palette:
/// `0.0` -> Electric Cyan (`#38BDF8`)
/// `0.5` -> Vivid Purple  (`#A855F7`)
/// `1.0` -> Warm Amber    (`#F59E0B`)
#[inline]
fn gradient_color(t: f32) -> [f32; 3] {
    let tc = t.clamp(0.0, 1.0);
    let cyan = [56.0, 189.0, 248.0];
    let purple = [168.0, 85.0, 247.0];
    let amber = [245.0, 158.0, 11.0];

    if tc < 0.5 {
        let u = (tc * 2.0).clamp(0.0, 1.0);
        let s = u * u * (3.0 - 2.0 * u);
        lerp3(cyan, purple, 0.25 * s + 0.75 * u)
    } else {
        let u = ((tc - 0.5) * 2.0).clamp(0.0, 1.0);
        let s = u * u * (3.0 - 2.0 * u);
        lerp3(purple, amber, 0.25 * s + 0.75 * u)
    }
}

/// Evaluates a single supersample `(u, v)` in `[0, 1]^2` of the StudioEQ Master Pro icon,
/// returning `(rgb_0_255, alpha_0_1)`.
#[inline]
fn sample_icon_subpixel(u: f32, v: f32, inv_size: f32) -> ([f32; 3], f32) {
    let px = u - 0.5;
    let py = v - 0.5;
    let aa = 0.85 * inv_size;

    // 1. Squircle Signed Distance Field (macOS / Windows 11 Pro Audio proportions)
    let half = 0.462_f32;
    let rad = 0.208_f32;
    let qx = px.abs() - (half - rad);
    let qy = py.abs() - (half - rad);
    let d_sq = qx.max(0.0).hypot(qy.max(0.0)) + qx.max(qy).min(0.0) - rad;

    let alpha_sq = 1.0 - smoothstep(-aa, aa, d_sq);
    if alpha_sq <= 0.0 {
        return ([0.0, 0.0, 0.0], 0.0);
    }

    // 2. Deep Obsidian-Blue Squircle Background (#0F172A -> #1E293B)
    let bg_dark = [15.0, 23.0, 42.0];
    let bg_light = [30.0, 41.0, 59.0];
    let diag_t = (1.0 - (0.50 * u + 0.70 * v)).clamp(0.0, 1.0);
    let radial_dome = (1.0 - 1.45 * (px + 0.06).hypot(py + 0.14)).clamp(0.0, 1.0);
    let bg_mix = (0.52 * diag_t + 0.58 * radial_dome).clamp(0.0, 1.0);
    let mut rgb = lerp3(bg_dark, bg_light, bg_mix);

    // 3. Subtle Studio EQ Grid & Parametric Bell Curve Backdrop
    let in_inner = 1.0 - smoothstep(-0.075, -0.035, d_sq);
    let grid_half_w = (0.85 * inv_size).max(0.0035);
    let mut grid_v = 0.0_f32;
    for gx in [-0.20_f32, 0.0, 0.20] {
        let line = 1.0 - smoothstep(0.0, grid_half_w, (px - gx).abs());
        if line > grid_v {
            grid_v = line;
        }
    }

    let line_0db_w = (0.95 * inv_size).max(0.004);
    let line_0db = 1.0 - smoothstep(0.0, line_0db_w, py.abs());
    let grid_gain = (0.045 * grid_v + 0.11 * line_0db) * in_inner;
    rgb[0] += 148.0 * grid_gain;
    rgb[1] += 163.0 * grid_gain;
    rgb[2] += 184.0 * grid_gain;

    let wave_col = gradient_color((u - 0.16) / 0.68);

    let g1 = (-((px + 0.16) / 0.13).powi(2)).exp();
    let g2 = (-((px + 0.01) / 0.10).powi(2)).exp();
    let g3 = (-((px - 0.16) / 0.13).powi(2)).exp();
    let eq_y = -0.145 * g1 + 0.105 * g2 - 0.215 * g3;

    let y_lo = eq_y.min(0.0);
    let y_hi = eq_y.max(0.0);
    let d_eq_fill = (y_lo - py).max(py - y_hi);
    let eq_x_fade = 1.0 - smoothstep(0.30, 0.38, px.abs());
    let eq_fill_alpha = (1.0 - smoothstep(-aa, aa, d_eq_fill)) * eq_x_fade * in_inner * 0.15;
    rgb = lerp3(rgb, wave_col, eq_fill_alpha);

    let eq_stroke_w = (1.10 * inv_size).max(0.005);
    let eq_stroke = (1.0 - smoothstep(0.0, eq_stroke_w, (py - eq_y).abs()))
        * eq_x_fade
        * in_inner
        * 0.24;
    rgb = lerp3(rgb, wave_col, eq_stroke);

    // 4. Stylized 7-Bar Sculpted Studio Audio Waveform (#38BDF8 -> #A855F7 -> #F59E0B)
    const BARS: [(f32, f32); 7] = [
        (-0.30, 0.070),
        (-0.20, 0.172),
        (-0.10, 0.282),
        (0.00, 0.178),
        (0.10, 0.322),
        (0.20, 0.212),
        (0.30, 0.095),
    ];
    let r_bar = 0.0275_f32;
    let mut d_wave = 10.0_f32;
    let abs_py = py.abs();
    for &(bx, bh) in &BARS {
        let dx = px - bx;
        let dy = (abs_py - bh).max(0.0);
        let d_cap = dx.hypot(dy) - r_bar;
        if d_cap < d_wave {
            d_wave = d_cap;
        }
    }

    let d_pos = d_wave.max(0.0);
    let glow = (((-d_pos * 15.0).exp() * 0.25 + (-d_pos * 44.0).exp() * 0.23) * in_inner)
        .min(0.52);
    rgb[0] += wave_col[0] * glow;
    rgb[1] += wave_col[1] * glow;
    rgb[2] += wave_col[2] * glow;

    let wave_alpha = (1.0 - smoothstep(-aa, aa, d_wave)) * in_inner;
    let core_spec = (1.0 - smoothstep(-0.021, -0.004, d_wave)) * 0.18;
    let top_sheen = (0.5 - py * 1.1).clamp(0.0, 1.0) * 0.08;
    let spec = (core_spec + top_sheen).clamp(0.0, 0.26);
    let wave_lit = lerp3(wave_col, [255.0, 255.0, 255.0], spec);
    rgb = lerp3(rgb, wave_lit, wave_alpha);

    // 5. Subtle Glass Dome Sheen & Thin Luminous Squircle Border
    let glass_arc = (1.0 - smoothstep(-0.02, 0.05, py + 0.10 - 0.22 * px * px))
        * (1.0 - smoothstep(-0.04, 0.0, d_sq));
    rgb[0] += 210.0 * glass_arc * 0.032;
    rgb[1] += 230.0 * glass_arc * 0.032;
    rgb[2] += 255.0 * glass_arc * 0.032;

    let border_w = (1.12 * inv_size).max(0.0135);
    let d_border = (d_sq + border_w * 0.5).abs() - border_w * 0.5;
    let border_mask = 1.0 - smoothstep(-aa, aa, d_border);

    let rim_grad = gradient_color(0.60 * u + 0.40 * v);
    let top_left = (1.0 - (0.85 * u + 0.90 * v)).clamp(0.0, 1.0);
    let spec_gain = 0.42 * top_left + 0.14;
    let rim_color = [
        rim_grad[0] * 0.58 + 220.0 * spec_gain,
        rim_grad[1] * 0.58 + 238.0 * spec_gain,
        rim_grad[2] * 0.58 + 255.0 * spec_gain,
    ];
    let border_alpha = (border_mask * (0.48 + 0.40 * top_left)).clamp(0.0, 1.0);
    rgb = lerp3(rgb, rim_color, border_alpha);

    (
        [
            rgb[0].clamp(0.0, 255.0),
            rgb[1].clamp(0.0, 255.0),
            rgb[2].clamp(0.0, 255.0),
        ],
        alpha_sq.clamp(0.0, 1.0),
    )
}

/// Programmatically builds the anti-aliased `64x64` RGBA window/taskbar icon
/// for `StudioEQ Master Pro` (`ViewportBuilder::with_icon(build_app_icon())`).
pub fn build_app_icon() -> IconData {
    const SIZE: u32 = 64;
    const SS: u32 = 4;
    const SUB_SAMPLES: f32 = (SS * SS) as f32;

    let inv_size = 1.0 / (SIZE as f32);
    let inv_grid = 1.0 / ((SIZE * SS) as f32);
    let mut rgba = Vec::with_capacity((SIZE * SIZE * 4) as usize);

    for iy in 0..SIZE {
        for ix in 0..SIZE {
            let mut premul_r = 0.0_f32;
            let mut premul_g = 0.0_f32;
            let mut premul_b = 0.0_f32;
            let mut sum_a = 0.0_f32;

            for sy in 0..SS {
                let v = (((iy * SS + sy) as f32) + 0.5) * inv_grid;
                for sx in 0..SS {
                    let u = (((ix * SS + sx) as f32) + 0.5) * inv_grid;
                    let (rgb, a) = sample_icon_subpixel(u, v, inv_size);
                    premul_r += rgb[0] * a;
                    premul_g += rgb[1] * a;
                    premul_b += rgb[2] * a;
                    sum_a += a;
                }
            }

            if sum_a > 1.0e-4 {
                let inv_a = 1.0 / sum_a;
                let r = (premul_r * inv_a).round().clamp(0.0, 255.0) as u8;
                let g = (premul_g * inv_a).round().clamp(0.0, 255.0) as u8;
                let b = (premul_b * inv_a).round().clamp(0.0, 255.0) as u8;
                let a = ((sum_a / SUB_SAMPLES) * 255.0).round().clamp(0.0, 255.0) as u8;
                rgba.extend_from_slice(&[r, g, b, a]);
            } else {
                rgba.extend_from_slice(&[0, 0, 0, 0]);
            }
        }
    }

    IconData {
        rgba,
        width: SIZE,
        height: SIZE,
    }
}

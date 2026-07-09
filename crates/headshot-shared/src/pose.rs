//! Pose encoding → camera matrices, unprojection (doc/01 §0, §5).
//!
//! Conventions: extrinsics are camera-from-world (OpenCV axes, `p_cam =
//! R·p_world + t`); frame 0 is the reference frame (world = its camera
//! space); quaternions are scalar-last xyzw and may be non-unit; principal
//! point at the image center; depth is z-depth.

/// Decoded camera: world→camera extrinsics + pinhole intrinsics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    /// Rotation, row-major (world→camera).
    pub r: [[f32; 3]; 3],
    pub t: [f32; 3],
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
}

/// Scalar-last quaternion (x, y, z, w) → rotation matrix, robust to
/// non-unit q via `s = 2/|q|²` (doc/01 §5.1).
pub fn quat_to_mat(q: [f32; 4]) -> [[f32; 3]; 3] {
    let [x, y, z, w] = q;
    let n = x * x + y * y + z * z + w * w;
    let s = if n > 0.0 { 2.0 / n } else { 0.0 };
    [
        [
            1.0 - s * (y * y + z * z),
            s * (x * y - z * w),
            s * (x * z + y * w),
        ],
        [
            s * (x * y + z * w),
            1.0 - s * (x * x + z * z),
            s * (y * z - x * w),
        ],
        [
            s * (x * z - y * w),
            s * (y * z + x * w),
            1.0 - s * (x * x + y * y),
        ],
    ]
}

impl Camera {
    /// Decode one 9-component pose encoding `[t(3), quat xyzw(4), fov_h,
    /// fov_w]` for an image of `width`×`height` pixels (doc/01 §0).
    pub fn from_pose_enc(enc: &[f32], width: u32, height: u32) -> Self {
        assert_eq!(enc.len(), 9);
        let r = quat_to_mat([enc[3], enc[4], enc[5], enc[6]]);
        let (w, h) = (width as f32, height as f32);
        Camera {
            r,
            t: [enc[0], enc[1], enc[2]],
            fy: (h / 2.0) / (enc[7] / 2.0).tan(),
            fx: (w / 2.0) / (enc[8] / 2.0).tan(),
            cx: w / 2.0,
            cy: h / 2.0,
        }
    }

    /// Integer pixel + z-depth → world point (frame-0 camera space):
    /// `p_world = Rᵀ·(p_cam − t)` (doc/01 §5.2; integer coords, no
    /// half-pixel offset — matches the reference).
    pub fn unproject(&self, x: u32, y: u32, depth: f32) -> [f32; 3] {
        let pc = [
            (x as f32 - self.cx) / self.fx * depth,
            (y as f32 - self.cy) / self.fy * depth,
            depth,
        ];
        let d = [pc[0] - self.t[0], pc[1] - self.t[1], pc[2] - self.t[2]];
        [
            self.r[0][0] * d[0] + self.r[1][0] * d[1] + self.r[2][0] * d[2],
            self.r[0][1] * d[0] + self.r[1][1] * d[1] + self.r[2][1] * d[2],
            self.r[0][2] * d[0] + self.r[1][2] * d[1] + self.r[2][2] * d[2],
        ]
    }

    /// World point → (pixel x, pixel y, z-depth).
    pub fn project(&self, p: [f32; 3]) -> (f32, f32, f32) {
        let pc = [
            self.r[0][0] * p[0] + self.r[0][1] * p[1] + self.r[0][2] * p[2] + self.t[0],
            self.r[1][0] * p[0] + self.r[1][1] * p[1] + self.r[1][2] * p[2] + self.t[1],
            self.r[2][0] * p[0] + self.r[2][1] * p[1] + self.r[2][2] * p[2] + self.t[2],
        ];
        (pc[0] / pc[2] * self.fx + self.cx, pc[1] / pc[2] * self.fy + self.cy, pc[2])
    }

    /// Camera center in world coordinates: `c = −Rᵀ·t` (doc/06 §2).
    pub fn center(&self) -> [f32; 3] {
        let t = self.t;
        [
            -(self.r[0][0] * t[0] + self.r[1][0] * t[1] + self.r[2][0] * t[2]),
            -(self.r[0][1] * t[0] + self.r[1][1] * t[1] + self.r[2][1] * t[2]),
            -(self.r[0][2] * t[0] + self.r[1][2] * t[1] + self.r[2][2] * t[2]),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32, tol: f32, what: &str) {
        assert!((a - b).abs() < tol, "{what}: {a} vs {b}");
    }

    #[test]
    fn quat_identity_and_axis() {
        let r = quat_to_mat([0.0, 0.0, 0.0, 1.0]);
        for (i, row) in r.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                assert_close(*v, (i == j) as u8 as f32, 1e-6, "identity");
            }
        }
        // negated quaternion = same rotation
        let r2 = quat_to_mat([0.0, 0.0, 0.0, -1.0]);
        assert_eq!(r, r2);

        // 90° about z: q = (0, 0, sin45, cos45); rotates x̂ → ŷ
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let r = quat_to_mat([0.0, 0.0, s, s]);
        assert_close(r[0][0], 0.0, 1e-6, "r00");
        assert_close(r[1][0], 1.0, 1e-6, "r10");
        assert_close(r[0][1], -1.0, 1e-6, "r01");
    }

    #[test]
    fn quat_non_unit_matches_normalized() {
        let q = [0.3f32, -0.5, 0.1, 0.9];
        let norm = q.iter().map(|v| v * v).sum::<f32>().sqrt();
        let qn: Vec<f32> = q.iter().map(|v| v / norm).collect();
        let a = quat_to_mat(q);
        let b = quat_to_mat([qn[0], qn[1], qn[2], qn[3]]);
        // scaling by 3 too
        let c = quat_to_mat([3.0 * q[0], 3.0 * q[1], 3.0 * q[2], 3.0 * q[3]]);
        for i in 0..3 {
            for j in 0..3 {
                assert_close(a[i][j], b[i][j], 1e-5, "non-unit vs normalized");
                assert_close(a[i][j], c[i][j], 1e-5, "scaled");
            }
        }
        // orthonormality
        for row in &a {
            let dot: f32 = row.iter().map(|v| v * v).sum();
            assert_close(dot, 1.0, 1e-5, "row norm");
        }
    }

    #[test]
    fn pose_enc_round_trip_and_reprojection() {
        // an arbitrary non-unit pose encoding
        let enc = [0.4f32, -0.2, 1.5, 0.11, -0.32, 0.05, 0.92, 0.8, 1.1];
        let cam = Camera::from_pose_enc(&enc, 640, 480);

        // FoV round-trip
        assert_close(2.0 * (240.0 / cam.fy).atan(), enc[7], 1e-5, "fov_h");
        assert_close(2.0 * (320.0 / cam.fx).atan(), enc[8], 1e-5, "fov_w");

        // project(unproject(x, y, d)) = (x, y, d)
        for (x, y, d) in [(0u32, 0u32, 2.0f32), (320, 240, 1.0), (639, 479, 7.5), (17, 401, 0.3)] {
            let p = cam.unproject(x, y, d);
            let (px, py, pd) = cam.project(p);
            assert_close(px, x as f32, 1e-2, "px");
            assert_close(py, y as f32, 1e-2, "py");
            assert_close(pd, d, 1e-4, "pd");
        }

        // the camera center projects to depth 0 and unproject(c)=identityish:
        // center must satisfy R·c + t = 0
        let c = cam.center();
        let pc = [
            cam.r[0][0] * c[0] + cam.r[0][1] * c[1] + cam.r[0][2] * c[2] + cam.t[0],
            cam.r[1][0] * c[0] + cam.r[1][1] * c[1] + cam.r[1][2] * c[2] + cam.t[1],
            cam.r[2][0] * c[0] + cam.r[2][1] * c[1] + cam.r[2][2] * c[2] + cam.t[2],
        ];
        for v in pc {
            assert_close(v, 0.0, 1e-5, "center");
        }
    }
}

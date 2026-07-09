// Cooperative-matrix (WMMA) GEMM: out (M,N) = x (M,K) @ w^T + bias, with
// x, w, out f16 and f32 accumulation — the trunk's workhorse (doc/03 §1).
//
// Uses the 16x16x16 f16→f32 configuration (present on RDNA3/RADV). One
// workgroup (one wave; redundant-but-correct if the driver packs two
// subgroups) computes a 16(M)×64(N) output stripe: the A tile is loaded
// once per K-step and reused across 4 B tiles. Requires K % 16 == 0 and
// N % 64 == 0 (all trunk GEMMs satisfy both); M is arbitrary — tensor
// allocations are padded to 16-row multiples (zero-filled), and the store
// goes through workgroup scratch with scalar bounds-checked writes, which
// also converts the f32 accumulator to f16.

enable f16;
enable wgpu_cooperative_matrix;

alias MatA = coop_mat16x16<f16, A>;
alias MatB = coop_mat16x16<f16, B>;
alias MatC = coop_mat16x16<f32, C>;

struct Params {
    m: u32,
    k: u32,
    n: u32,
    has_bias: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f16>;
@group(0) @binding(2) var<storage, read> w: array<f16>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f16>;

const NT: u32 = 4u; // B/C tiles per workgroup along N

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let m0 = wg.y * 16u;
    let n0 = wg.x * (16u * NT);

    var c: array<MatC, NT>;
    let k_tiles = p.k / 16u;
    for (var t = 0u; t < k_tiles; t++) {
        // A tile: x rows m0..m0+16, k-cols t*16.. — row-major, stride K.
        // Rows past M read zero-padded allocation slack.
        let a = coopLoadT<MatA>(&x[m0 * p.k + t * 16u], p.k);
        for (var j = 0u; j < NT; j++) {
            // B tile: w is (N,K) row-major, so B[kk][nn] = w[(n0+nn)*K + kk0+kk]
            // is column-major with stride K.
            let b = coopLoad<MatB>(&w[(n0 + j * 16u) * p.k + t * 16u], p.k);
            c[j] = coopMultiplyAdd(a, b, c[j]);
        }
    }

    for (var j = 0u; j < NT; j++) {
        // f32 tile → workgroup scratch (row-major 16x16), then scalar
        // bias-add + f16 convert + bounds-checked global write.
        coopStoreT(c[j], &scratch[0], 16u);
        workgroupBarrier();
        for (var e = li; e < 256u; e += 64u) {
            let mm = e / 16u;
            let nn = e % 16u;
            let row = m0 + mm;
            let col = n0 + j * 16u + nn;
            if row < p.m && col < p.n {
                var v = scratch[e];
                if p.has_bias != 0u {
                    v += bias[col];
                }
                out[row * p.n + col] = f16(v);
            }
        }
        workgroupBarrier();
    }
}

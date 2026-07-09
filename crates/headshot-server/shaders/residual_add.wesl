// Elementwise residual add (doc/03 §1, "reusable" list). First kernel through
// the WESL pipeline; exercised by tests/gpu_smoke.rs. LayerScale folds into
// this kernel later (doc/03 §1 item 4).

@group(0) @binding(0) var<storage, read_write> acc: array<f32>;
@group(0) @binding(1) var<storage, read> addend: array<f32>;

@compute @workgroup_size(256)
fn residual_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < arrayLength(&acc) {
        acc[i] = acc[i] + addend[i];
    }
}

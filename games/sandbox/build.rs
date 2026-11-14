//! Exports `ae_wasm::InstanceSetBuilder` needs: the heap owner's allocator and
//! constructors, and every instance's stack pointer and TLS setup.
fn main() {
    for export in ["malloc", "free", "__wasm_call_ctors", "__stack_pointer", "__tls_size", "__tls_align", "__wasm_init_tls"] {
        println!("cargo:rustc-link-arg=--export={export}");
    }
}

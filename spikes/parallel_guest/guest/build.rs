//! Exports the host needs to give every instance its own stack and TLS block
//! over one shared linear memory.
fn main() {
    for export in ["malloc", "free", "__wasm_call_ctors", "__stack_pointer", "__tls_base", "__tls_size", "__tls_align", "__wasm_init_tls"] {
        println!("cargo:rustc-link-arg=--export={export}");
    }
}

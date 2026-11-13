//! Link this module's static data into its own region of the shared memory
//! (MODULE2_BASE, agreed with the host) and export what the host needs to
//! give each instance its own stack and TLS block.
fn main() {
    println!("cargo:rustc-link-arg=--global-base=67108864"); // 64 MiB
    for export in ["__wasm_call_ctors", "__stack_pointer", "__tls_base", "__tls_size", "__tls_align", "__wasm_init_tls", "__data_end", "__heap_base"] {
        println!("cargo:rustc-link-arg=--export={export}");
    }
}

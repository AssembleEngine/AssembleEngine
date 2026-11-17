//! Embeds the application icon in the Windows executable. embed-resource is a
//! no-op elsewhere; on other platforms the icon lives in `ae_render`'s window.

fn main() {
    let icon = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/icon.ico");
    println!("cargo:rerun-if-changed={}", icon.display());

    let out = std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR"));
    let resource = out.join("icon.rc");
    let icon = icon.to_string_lossy().replace('\\', "/");
    std::fs::write(&resource, format!("IDI_ICON1 ICON \"{icon}\"\n")).expect("write icon.rc");

    let _ = embed_resource::compile(&resource, embed_resource::NONE);
}

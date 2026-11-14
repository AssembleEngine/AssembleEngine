//! Print a game's components and their fields' view classes.
fn main() -> anyhow::Result<()> {
    let name = std::env::args().nth(1).unwrap_or_else(|| "crossing".into());
    let wasm = ae_sim::build::game(&name)?;
    let schema = ae_sim::schema::Schema::from_module(&wasm)?;
    for c in &schema.components {
        let fields: Vec<String> = c.fields.iter().map(|f| format!("{}:{}", f.name, f.view())).collect();
        println!("{:#018x} {} [{}]", c.id, c.name, fields.join(", "));
    }
    Ok(())
}

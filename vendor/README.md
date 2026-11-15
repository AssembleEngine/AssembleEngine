# vendored Flecs

| Path | What |
| --- | --- |
| `flecs-upstream/` | Pristine upstream `distr/flecs.c` + `flecs.h` (see its README for the commit). Never edited. |
| `patches/` | Our changes to those two files, as unified diffs against `flecs-upstream/`. |
| `flecs_ecs_sys/` | Build crate from Indra-db/Flecs-Rust `bf956db` (bindings + `build.rs`). `src/flecs.c`/`flecs.h` = upstream + patches. |
| `upstream-glue/` | The crate's original `flecs_rust.c/h` (Rust-binding glue, written for older Flecs internals). Replaced in `flecs_ecs_sys/src` by a stub: this project uses none of it. |

Other local changes to the crate: standalone `Cargo.toml` (workspace inheritance removed), `src/extensions.rs`
(removed `ecs_entity_desc_t` fields that no longer exist), `src/bindings*.rs` regenerated against the new header.

## Patches
- `0001-column-allocator.patch`: `ecs_column_allocator_set()` + `ecs_column_malloc/realloc/free`. Table column
  arrays use them; everything else keeps using `ecs_os_*`. The engine points the column allocator at guest memory,
  so only component data is guest-writable (design §2, §7). Entity-id arrays stay on the host heap; the engine
  copies them into a read scratch buffer per chunk.
- `0002-c23-const-string-search.patch`: keep `strchr`/`strrchr` results const-correct when compiled as C23
  (the GCC default), where those functions preserve the constness of their argument. The casts are only where
  Flecs intentionally edits mutable parser token buffers.

## Updating Flecs
1. Copy `distr/flecs.c`, `distr/flecs.h` from upstream into `flecs-upstream/` and `flecs_ecs_sys/src/`.
2. `cd flecs_ecs_sys/src && for p in ../../patches/*.patch; do patch -p1 < "$p"; done` (resolve any drift,
   regenerate the affected patch).
3. Regenerate bindings once: add feature `regenerate_binding` to the dependency in `Cargo.toml`, `cargo build`, remove it.
4. `cargo test`; debug and release digests must match; expect digests to change when entity-id allocation changes.

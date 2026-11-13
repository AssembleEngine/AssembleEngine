Pristine upstream amalgamation, unmodified. Do not edit.

- Source: https://github.com/SanderMertens/flecs  `distr/flecs.c`, `distr/flecs.h`
- Commit: 68631db79f079995c90f783e86801e2b16279702 (master, 2026-10-16; reports Flecs 4.1.6)
- The crate in `../flecs_ecs_sys` (from Indra-db/Flecs-Rust bf956db) compiles
  `src/flecs.c` / `src/flecs.h`, which are this pair plus `../patches/*.patch`.
- Refresh: copy the two files here, copy over to `../flecs_ecs_sys/src/`, re-apply patches
  (`git apply`/`patch -p1`), rebuild with `--features regenerate_binding`.

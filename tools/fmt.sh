#!/bin/sh
# Formats the whole tree: every workspace package, plus the guest crates that
# are not workspace members (each has its own manifest and its own `[workspace]`
# table: `games/*`, `render/*`, `spikes/parallel_guest/guest*`,
# `crates/ae_sim/tests/guest`). `rustfmt.toml` and the pinned toolchain
# (`rust-toolchain.toml`) decide the result.
#
#   tools/fmt.sh            format in place
#   tools/fmt.sh --check    exit 1 and print the files that differ, changing
#                           nothing (what the pre-commit hook runs)
#
# `vendor/` is never formatted: it is the patched flecs copy, and `cargo fmt
# --all` would follow the `flecs_ecs_sys` path dependency into it. So this
# script names the workspace packages (`cargo fmt -p ...`, which does not follow
# path dependencies) instead of using `--all`.
set -eu

# shellcheck disable=SC2086 # $members and $guests are word lists built below
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

case "${1:-}" in
    "") check=no ;;
    --check) check=yes ;;
    -h | --help)
        sed -n '2,9s/^# \{0,1\}//p' "$0"
        exit 0
        ;;
    *)
        echo "usage: tools/fmt.sh [--check]" >&2
        exit 2
        ;;
esac

# Workspace members are the manifests without their own `[workspace]` table
# (the root has one, and so does every guest). A member's package name is the
# first `name = "..."` line in its manifest.
members=""
guests=""
for manifest in $(find . -name Cargo.toml -not -path './vendor/*' -not -path './target/*' -print | sort); do
    if grep -q '^\[workspace\]' "$manifest"; then
        [ "$manifest" = ./Cargo.toml ] || guests="$guests $manifest"
    else
        members="$members -p $(sed -n 's/^name = "\(.*\)"$/\1/p' "$manifest" | head -n 1)"
    fi
done

status=0
if [ "$check" = yes ]; then
    cargo fmt $members -- --check || status=1
else
    cargo fmt $members || status=1
fi

for manifest in $guests; do
    if [ "$check" = yes ]; then
        cargo fmt --manifest-path "$manifest" -- --check || status=1
    else
        cargo fmt --manifest-path "$manifest" || status=1
    fi
done

exit "$status"

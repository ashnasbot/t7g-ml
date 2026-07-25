# ScummVM `CellGame` — the original 7th Guest Stauf AI

`cell.cpp` and `cell.h` are vendored from
[ScummVM](https://github.com/scummvm/scummvm) `engines/groovie/logic/`. They
implement `Groovie::CellGame`, the AI Stauf plays in the microscope puzzle.

Used via `src/stauf_wasm.cpp`, as a selectable opponent in the browser SPA.

## Provenance

**Base version: `b91b8a0d08e07b07991e9fa16cba16301c415d1a`** (2024-01-08,
*"GROOVIE: ScummVM was upgraded to GPLv3. Sync the secondary license with it"*).

Differences from upstream:

| File | Difference |
|---|---|
| `cell.cpp` | none (content identical) |
| `cell.h` | **modified** - adds `CellGame::setMoveCount()`, 7 lines |

Both files also have CRLF line endings and no trailing newline, from the
original copy.

### The `cell.h` modification

`setMoveCount()` exposes the private `_moveCount` field so it can be seeded from
outside the class. `calcMove()` picks its real search depth with
`depths[3*(depth-2) + _moveCount%3]`, and a fresh `CellGame` always starts at 0
— so without this, every call would use the same depth slot. Seeding it with a
per-side cumulative move index is what reproduces the original game's
move-to-move depth variation.

## Licence — GPLv3-or-later

Full text in `LICENSE`. Note this is stricter than ScummVM overall (GPLv2+);
these two files carry a v3+ header.

### Effect on the rest of the repo

The project is MIT. with this vendored GPL code here only.

Any **artefact that links this code in** such as `make stauf-wasm` is also
GPL, and so the `public/` bundle from `make pages`, is conveyed under GPLv3.
That target emits `LICENSE.GPLv3` and `README-licensing.md` (the source offer);
nothing commited, by design.

## Keeping the GPL surface to two files

`cell.cpp` includes three ScummVM headers. Rather than vendor those (each
GPLv3), `shim/` holds independently-written MIT replacements covering only what
`cell.{cpp,h}` actually reference:

```
shim/common/scummsys.h        byte/uint16/int8 typedefs, printf macros, <cassert>
shim/common/textconsole.h     warning() / error() declarations
shim/common/config-manager.h  empty (ConfMan is never referenced)
shim/groovie/logic/cell.h     include redirect to ../../../cell.h
```

So the GPL surface is exactly `cell.cpp` + `cell.h`. `shim/`,
`src/stauf_wasm.cpp` and the build rules are ours and MIT — though the binary
they produce is GPLv3, because it contains `CellGame`.

## Building

```sh
source 3rd_party/emsdk/emsdk_env.sh
make stauf-wasm          # -> build/wasm/stauf.{mjs,wasm}
```
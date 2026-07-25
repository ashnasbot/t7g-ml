# ScummVM `CellGame` — the original 7th Guest Stauf AI

`cell.cpp` and `cell.h` are vendored unmodified from
[ScummVM](https://github.com/scummvm/scummvm) `engines/groovie/logic/`. They
implement `Groovie::CellGame`, the AI Stauf plays in the microscope puzzle.

Used here as the fixed anchor of the rating ladder (`lib/eval_db.py`) and, via
`src/stauf_wasm.cpp`, as a selectable opponent in the browser SPA.

> **Provenance:** the upstream commit was not recorded when these were first
> copied into the untracked `3rd_party/cell/`. The files are unmodified. Diff
> against upstream and pin the hash here before relying on this for compliance.

## Licence — GPLv3-or-later

Full text in `LICENSE`. Note this is stricter than ScummVM overall (GPLv2+);
these two files carry a v3+ header.

They also carry a notice that **MojoTouch** was *exclusively* licensed this code
for closed-source products on 2021-11-10. That grant confers nothing on us, and
is why asking for a permissive relicence would likely fail: granting e.g. MIT
would permit closed-source use by anyone, cutting across that exclusivity. Treat
these files as GPLv3-only here.

### Effect on the rest of the repo

The project is MIT. Vendoring GPL code here relicenses none of it — MIT files
stay MIT and are reusable as such, including out of the published SPA.

What is affected is any **artefact that links this code in**: `make stauf-wasm`,
and so the `public/` bundle from `make pages`, is conveyed under GPLv3. That
target emits `LICENSE.GPLv3` and `README-licensing.md` (the source offer);
`public/` is gitignored, so those must be generated, not committed.

Corresponding Source is defined by function, not licence — MIT source satisfies
it, being GPL-compatible and more permissive. No relicensing needed.

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

`tests/test_stauf_wasm.py` asserts this build plays identically to the native
`cell_dll.so` that anchors the ladder.

The native build still comes from the untracked `3rd_party/cell/` working copy
(`cell_dll.cpp` + CMake) and is not distributed.

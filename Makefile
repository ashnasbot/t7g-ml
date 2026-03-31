CC     = gcc

# -march=native is fastest for local dev but produces a DLL that crashes on
# CPUs lacking the same instruction extensions (STATUS_ILLEGAL_INSTRUCTION).
# Use x86-64-v2 (SSE4.2 baseline, supported since ~2010) for a portable build,
# or native for a local-only build.
MARCH  ?= x86-64-v2
CFLAGS = -O3 -march=$(MARCH) -ffast-math -static-libgcc

.PHONY: dll dll-native clean clean-native clean-portable

# ── Portable build (x86-64-v2, outputs to lib/, bundled in wheel) ────────────
dll: lib/micro3.dll lib/micro4.dll lib/micro_mcts.dll lib/micro_mcts_heuristic.dll lib/cell_dll.dll

lib/micro3.dll: micro_3.c t7g_core.h
	$(CC) $(CFLAGS) micro_3.c -o $@ --shared

lib/micro4.dll: micro_4.c t7g_core.h
	$(CC) $(CFLAGS) micro_4.c -o $@ --shared

lib/micro_mcts.dll: micro_mcts.c
	$(CC) $(CFLAGS) micro_mcts.c -o $@ --shared

lib/micro_mcts_heuristic.dll: micro_mcts_heuristic.c
	$(CC) $(CFLAGS) micro_mcts_heuristic.c -o $@ --shared -lm

# cell_dll is built with CMake's default arch (x86-64) and is already portable.
lib/cell_dll.dll: cell/build
	cd cell/build && cmake --build . --config Release

# ── Native build (march=native, faster locally, outputs to ./, NOT in wheel) ──
# Loaded in preference to lib/ builds by _find_dll in lib/t7g.py and lib/mcgs.py.
dll-native: micro3.dll micro4.dll micro_mcts.dll micro_mcts_heuristic.dll

micro3.dll: micro_3.c t7g_core.h
	$(CC) -O3 -march=native -ffast-math -static-libgcc micro_3.c -o $@ --shared

micro4.dll: micro_4.c t7g_core.h
	$(CC) -O3 -march=native -ffast-math -static-libgcc micro_4.c -o $@ --shared

micro_mcts.dll: micro_mcts.c
	$(CC) -O3 -march=native -ffast-math -static-libgcc micro_mcts.c -o $@ --shared

micro_mcts_heuristic.dll: micro_mcts_heuristic.c
	$(CC) -O3 -march=native -ffast-math -static-libgcc micro_mcts_heuristic.c -o $@ --shared -lm

# ── Clean ─────────────────────────────────────────────────────────────────────
clean-native:
	-del /f /q micro3.dll micro4.dll micro_mcts.dll micro_mcts_heuristic.dll 2>nul

clean-portable:
	-del /f /q lib\micro3.dll lib\micro4.dll lib\micro_mcts.dll lib\micro_mcts_heuristic.dll lib\cell_dll.dll 2>nul
	-del /f /q lib\*.pyd lib\*.so 2>nul
	-rd /s /q build 2>nul

clean: clean-native clean-portable

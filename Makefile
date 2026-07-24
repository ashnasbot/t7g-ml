CC     = gcc
CXX    = g++

# -march=native is fastest for local dev but produces a library that crashes on
# CPUs lacking the same instruction extensions (STATUS_ILLEGAL_INSTRUCTION).
# Use x86-64-v2 (SSE4.2 baseline, supported since ~2010) for a portable build,
# or native for a local-only build.
MARCH  ?= x86-64-v2
CFLAGS = -O3 -march=$(MARCH) -fPIC -ffast-math -static-libgcc

# Windows cross-compilation from Linux (apt install mingw-w64).
# No -fPIC (not a concept on Windows); -static-libgcc avoids a libgcc DLL dep.
CROSS_CC    = x86_64-w64-mingw32-gcc
CROSS_FLAGS = -O3 -march=$(MARCH) -ffast-math -static-libgcc

# Optional ONNX Runtime policy prior.
# Build with: make WITH_ONNX=1 [ORT_ROOT=/path/to/onnxruntime]
# Requires onnxruntime C headers and shared library at ORT_ROOT.
WITH_ONNX   ?= 0
ORT_ROOT    ?= /usr/local
ORT_INCLUDE ?= $(ORT_ROOT)/include
ORT_LIB_DIR ?= $(ORT_ROOT)/lib
ifeq ($(WITH_ONNX),1)
  CFLAGS_ONNX  = -DBB_USE_ONNX -I$(ORT_INCLUDE)
  LFLAGS_ONNX  = -L$(ORT_LIB_DIR) -lonnxruntime -lm -Wl,-rpath,$(ORT_LIB_DIR)
  CFLAGS      += $(CFLAGS_ONNX)
else
  CFLAGS_ONNX  =
  LFLAGS_ONNX  =
endif

# C sources live in src/; native (-march=native) builds land in build/.
SRC := src

# Detect platform - $(OS) is set to "Windows_NT" by cmd.exe/mingw32-make.
ifeq ($(OS),Windows_NT)
    EXT     := dll
    MKDIR   := mkdir
    RM_F    := del /f /q
    RM_DIR  := rd /s /q
else
    EXT     := so
    MKDIR   := mkdir -p
    RM_F    := rm -f
    RM_DIR  := rm -rf
endif

.PHONY: dll dll-native dll-windows clean clean-native clean-portable clean-windows test-mcts

build:
	$(MKDIR) build

test-mcts: $(SRC)/micro_mcts.c tests/test_micro_mcts.c
	gcc -O0 -g -Wall $(SRC)/micro_mcts.c tests/test_micro_mcts.c -o test_micro_mcts -lm
	./test_micro_mcts

#  Portable build (x86-64-v2, outputs to lib/, bundled in wheel)
dll: lib/micro3.$(EXT) lib/micro4.$(EXT) lib/micro_mcts.$(EXT) lib/micro_mcts_heuristic.$(EXT) lib/beehive4.$(EXT)

lib/micro3.$(EXT): $(SRC)/micro_3.c $(SRC)/bb_core.h
	$(CC) $(CFLAGS) $(SRC)/micro_3.c -o $@ --shared $(LFLAGS_ONNX)

lib/micro4.$(EXT): $(SRC)/micro_4.c $(SRC)/bb_core.h
	$(CC) $(CFLAGS) $(SRC)/micro_4.c -o $@ --shared $(LFLAGS_ONNX)

lib/beehive4.$(EXT): $(SRC)/beehive_4.c
	$(CC) $(CFLAGS) $(SRC)/beehive_4.c -o $@ --shared -lm

lib/micro_mcts.$(EXT): $(SRC)/micro_mcts.c
	$(CC) $(CFLAGS) $(SRC)/micro_mcts.c -o $@ --shared -lm

lib/micro_mcts_heuristic.$(EXT): $(SRC)/micro_mcts_heuristic.c
	$(CC) $(CFLAGS) $(SRC)/micro_mcts_heuristic.c -o $@ --shared -lm

#  Native build (march=native, faster locally, outputs to build/, NOT in wheel)
# Loaded in preference to lib/ builds by _find_dll in lib/t7g.py and lib/mcgs.py.
dll-native: build/micro3.$(EXT) build/micro4.$(EXT) build/micro_mcts.$(EXT) build/micro_mcts_heuristic.$(EXT) build/beehive4.$(EXT)

build/micro3.$(EXT): $(SRC)/micro_3.c $(SRC)/bb_core.h | build
	$(CC) -O3 -march=native -ffast-math -fPIC $(CFLAGS_ONNX) $(SRC)/micro_3.c -o $@ --shared $(LFLAGS_ONNX)

build/micro4.$(EXT): $(SRC)/micro_4.c $(SRC)/bb_core.h | build
	$(CC) -O3 -march=native -ffast-math -fPIC $(CFLAGS_ONNX) $(SRC)/micro_4.c -o $@ --shared $(LFLAGS_ONNX)

build/micro_mcts.$(EXT): $(SRC)/micro_mcts.c | build
	$(CC) -O3 -march=native -ffast-math -fPIC $(SRC)/micro_mcts.c -o $@ --shared -lm

build/micro_mcts_heuristic.$(EXT): $(SRC)/micro_mcts_heuristic.c | build
	$(CC) -O3 -march=native -ffast-math -fPIC $(SRC)/micro_mcts_heuristic.c -o $@ --shared -lm

build/beehive4.$(EXT): $(SRC)/beehive_4.c | build
	$(CC) -O3 -march=native -ffast-math -fPIC $(SRC)/beehive_4.c -o $@ --shared -lm

#  WebAssembly build (Emscripten) — engine core for the self-contained browser
# webapp.  Only micro_mcts.c is ported: net eval stays JS-side (ONNX Runtime
# Web), so there is no C<->JS callback.  The JS driver pulls pending leaves via
# mcgs_get_pending_obs, runs the net, and feeds them back with mcgs_commit_batch
# (batch width is bounded by the Gumbel top-k, so batches stay small).
#
# Requires emsdk on PATH:  source 3rd_party/emsdk/emsdk_env.sh
# -ffast-math matches the native/portable builds so search selection stays in
# parity (verified against the .so by tests, not just assumed).
EMCC      ?= emcc
WASM_OUT  := build/wasm
# Underscore-prefixed exports = every mcgs_* entry point + the allocator the JS
# driver needs to stage boards/policies into wasm memory.
WASM_EXPORTS := _malloc,_free,_mcgs_clear,_mcgs_commit_batch,_mcgs_commit_batch_many,_mcgs_commit_expansion,_mcgs_create,_mcgs_create_ex,_mcgs_destroy,_mcgs_edge_used,_mcgs_get_best_action,_mcgs_get_leaf_board,_mcgs_get_leaf_turn,_mcgs_get_pending_boards,_mcgs_get_pending_obs,_mcgs_get_pending_obs_many,_mcgs_get_result,_mcgs_get_root_value,_mcgs_init,_mcgs_is_done,_mcgs_pending_count,_mcgs_pending_counts,_mcgs_search_destroy,_mcgs_set_clock_obs,_mcgs_set_completion_n0,_mcgs_set_num_simulations,_mcgs_set_rng_seed,_mcgs_set_sigma_scale,_mcgs_start_search,_mcgs_step,_mcgs_step_many,_mcgs_tt_size
EMFLAGS   := -O3 -ffast-math --no-entry \
             -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=web,worker,node \
             -sALLOW_MEMORY_GROWTH=1 -sWASM_BIGINT \
             -sEXPORTED_RUNTIME_METHODS=ccall,cwrap,getValue,setValue,HEAPF32,HEAPU8,HEAPU32,HEAP32

.PHONY: wasm clean-wasm

wasm: $(WASM_OUT)/micro_mcts.mjs

$(WASM_OUT)/micro_mcts.mjs: $(SRC)/micro_mcts.c $(SRC)/bb_core.h | $(WASM_OUT)
	$(EMCC) $(EMFLAGS) -sEXPORTED_FUNCTIONS=$(WASM_EXPORTS) $(SRC)/micro_mcts.c -o $@

$(WASM_OUT):
	$(MKDIR) $(WASM_OUT)

clean-wasm:
	$(RM_DIR) $(WASM_OUT)

#  Static single-page app (GitHub-Pages ready).
# Source lives in webapp/spa/ (index.html, app.mjs, engine.mjs, exported model).
# onnxruntime-web is loaded from the jsdelivr CDN at runtime (see app.mjs), so
# its 21 MB wasm blob is NOT committed or published here.  `make pages` folds in
# the wasm engine blobs and assembles public/, the directory GitHub Pages
# publishes.  `make model` re-exports net2.onnx from a checkpoint (CKPT=...).
SPA       := webapp/spa
PUBLIC    := public
CKPT      ?= export/models/run_net2b/promoted_iter0085.pt
ORT_VER   ?= 1.20.1

.PHONY: pages model dev-vendor clean-pages

model: $(SPA)/models/net2.onnx
$(SPA)/models/net2.onnx: $(CKPT) scripts/export_onnx_web.py lib/net2.py
	python scripts/export_onnx_web.py $(CKPT) $@

pages: wasm $(SPA)/models/net2.onnx
	-$(RM_DIR) $(PUBLIC)
	$(MKDIR) $(PUBLIC) $(PUBLIC)/models
	cp $(SPA)/index.html $(SPA)/app.mjs $(SPA)/engine.mjs $(PUBLIC)/
	cp $(SPA)/models/net2.onnx $(PUBLIC)/models/
	cp $(WASM_OUT)/micro_mcts.mjs $(WASM_OUT)/micro_mcts.wasm $(PUBLIC)/
	@echo "public/ assembled — serve it, or push to a GitHub Pages branch/dir."

# Fetch onnxruntime-web into a gitignored local dir for the node parity harness
# (tests/scratchpad import it directly; node can't import from an https URL).
dev-vendor:
	$(MKDIR) $(SPA)/vendor/ort
	cd $(SPA)/vendor/ort && for f in ort.wasm.min.mjs ort-wasm-simd-threaded.wasm ort-wasm-simd-threaded.mjs; do \
	  curl -sSL "https://cdn.jsdelivr.net/npm/onnxruntime-web@$(ORT_VER)/dist/$$f" -o "$$f"; done
	@echo "ORT vendored to $(SPA)/vendor/ort (gitignored, dev only)."

clean-pages:
	$(RM_DIR) $(PUBLIC)

#  Windows cross-compile from Linux (apt install mingw-w64)
# Outputs lib/*.dll so they sit alongside the .so files and are picked up
# on Windows by the same _find_dll() helpers in lib/*.py.
dll-windows: lib/micro3.dll lib/micro4.dll lib/micro_mcts.dll lib/micro_mcts_heuristic.dll lib/beehive4.dll

lib/micro3.dll: $(SRC)/micro_3.c $(SRC)/bb_core.h
	$(CROSS_CC) $(CROSS_FLAGS) $(SRC)/micro_3.c -o $@ --shared

lib/micro4.dll: $(SRC)/micro_4.c $(SRC)/bb_core.h
	$(CROSS_CC) $(CROSS_FLAGS) $(SRC)/micro_4.c -o $@ --shared

lib/beehive4.dll: $(SRC)/beehive_4.c
	$(CROSS_CC) $(CROSS_FLAGS) $(SRC)/beehive_4.c -o $@ --shared

lib/micro_mcts.dll: $(SRC)/micro_mcts.c
	$(CROSS_CC) $(CROSS_FLAGS) $(SRC)/micro_mcts.c -o $@ --shared -lm

lib/micro_mcts_heuristic.dll: $(SRC)/micro_mcts_heuristic.c
	$(CROSS_CC) $(CROSS_FLAGS) $(SRC)/micro_mcts_heuristic.c -o $@ --shared -lm

clean-windows:
	rm -f lib/micro3.dll lib/micro4.dll lib/micro_mcts.dll lib/micro_mcts_heuristic.dll lib/beehive4.dll

#  Clean
# clean-native drops the whole build/ dir; clean-portable drops the lib/ builds
# (leaving any hand-kept libs such as the local GPL test engines in place).
ifeq ($(OS),Windows_NT)
clean-native:
	-rd /s /q build 2>nul

clean-portable:
	-del /f /q lib\micro3.dll lib\micro4.dll lib\micro_mcts.dll lib\micro_mcts_heuristic.dll lib\beehive4.dll 2>nul
	-del /f /q lib\micro3.so lib\micro4.so lib\micro_mcts.so lib\micro_mcts_heuristic.so lib\beehive4.so 2>nul
	-del /f /q lib\*.pyd 2>nul
else
clean-native:
	rm -rf build

clean-portable:
	rm -f lib/micro3.so lib/micro4.so lib/micro_mcts.so lib/micro_mcts_heuristic.so lib/beehive4.so
	rm -f lib/micro3.dll lib/micro4.dll lib/micro_mcts.dll lib/micro_mcts_heuristic.dll lib/beehive4.dll
	rm -f lib/*.pyd
endif

clean: clean-native clean-portable clean-wasm clean-pages

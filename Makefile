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

clean: clean-native clean-portable

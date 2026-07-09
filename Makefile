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

# Detect platform - $(OS) is set to "Windows_NT" by cmd.exe/mingw32-make.
ifeq ($(OS),Windows_NT)
    EXT     := dll
    RM_F    := del /f /q
    RM_DIR  := rd /s /q
    DEVNULL := 2>nul
    SEP     := \\
else
    EXT     := so
    RM_F    := rm -f
    RM_DIR  := rm -rf
    DEVNULL :=
    SEP     := /
endif

.PHONY: dll dll-native dll-windows clean clean-native clean-portable clean-windows test-mcts

test-mcts: micro_mcts.c tests/test_micro_mcts.c
	gcc -O0 -g -Wall micro_mcts.c tests/test_micro_mcts.c -o test_micro_mcts -lm
	./test_micro_mcts

#  Portable build (x86-64-v2, outputs to lib/, bundled in wheel) 
dll: lib/micro3.$(EXT) lib/micro4.$(EXT) lib/micro_mcts.$(EXT) lib/micro_mcts_heuristic.$(EXT) lib/beehive4.$(EXT)

lib/micro3.$(EXT): micro_3.c bb_core.h
	$(CC) $(CFLAGS) micro_3.c -o $@ --shared $(LFLAGS_ONNX)

lib/micro4.$(EXT): micro_4.c bb_core.h
	$(CC) $(CFLAGS) micro_4.c -o $@ --shared $(LFLAGS_ONNX)

lib/beehive4.$(EXT): beehive_4.c
	$(CC) $(CFLAGS) beehive_4.c -o $@ --shared -lm

lib/micro_mcts.$(EXT): micro_mcts.c
	$(CC) $(CFLAGS) micro_mcts.c -o $@ --shared -lm

lib/micro_mcts_heuristic.$(EXT): micro_mcts_heuristic.c
	$(CC) $(CFLAGS) micro_mcts_heuristic.c -o $@ --shared -lm

#  Native build (march=native, faster locally, outputs to ./, NOT in wheel) 
# Loaded in preference to lib/ builds by _find_dll in lib/t7g.py and lib/mcgs.py.
dll-native: micro3.$(EXT) micro4.$(EXT) micro_mcts.$(EXT) micro_mcts_heuristic.$(EXT) beehive4.$(EXT)

micro3.$(EXT): micro_3.c bb_core.h
	$(CC) -O3 -march=native -ffast-math -fPIC $(CFLAGS_ONNX) micro_3.c -o $@ --shared $(LFLAGS_ONNX)

micro4.$(EXT): micro_4.c bb_core.h
	$(CC) -O3 -march=native -ffast-math -fPIC $(CFLAGS_ONNX) micro_4.c -o $@ --shared $(LFLAGS_ONNX)

micro_mcts.$(EXT): micro_mcts.c
	$(CC) -O3 -march=native -ffast-math -fPIC micro_mcts.c -o $@ --shared -lm

micro_mcts_heuristic.$(EXT): micro_mcts_heuristic.c
	$(CC) -O3 -march=native -ffast-math -fPIC micro_mcts_heuristic.c -o $@ --shared -lm

beehive4.$(EXT): beehive_4.c
	$(CC) -O3 -march=native -ffast-math -fPIC beehive_4.c -o $@ --shared -lm

#  Windows cross-compile from Linux (apt install mingw-w64)
# Outputs lib/*.dll so they sit alongside the .so files and are picked up
# on Windows by the same _find_dll() helpers in lib/*.py.
dll-windows: lib/micro3.dll lib/micro4.dll lib/micro_mcts.dll lib/micro_mcts_heuristic.dll lib/beehive4.dll

lib/micro3.dll: micro_3.c bb_core.h
	$(CROSS_CC) $(CROSS_FLAGS) micro_3.c -o $@ --shared

lib/micro4.dll: micro_4.c bb_core.h
	$(CROSS_CC) $(CROSS_FLAGS) micro_4.c -o $@ --shared

lib/beehive4.dll: beehive_4.c
	$(CROSS_CC) $(CROSS_FLAGS) beehive_4.c -o $@ --shared

lib/micro_mcts.dll: micro_mcts.c
	$(CROSS_CC) $(CROSS_FLAGS) micro_mcts.c -o $@ --shared -lm

lib/micro_mcts_heuristic.dll: micro_mcts_heuristic.c
	$(CROSS_CC) $(CROSS_FLAGS) micro_mcts_heuristic.c -o $@ --shared -lm

clean-windows:
	rm -f lib/micro3.dll lib/micro4.dll lib/micro_mcts.dll lib/micro_mcts_heuristic.dll lib/beehive4.dll

#  Clean
ifeq ($(OS),Windows_NT)
clean-native:
	-del /f /q micro3.dll micro4.dll micro_mcts.dll micro_mcts_heuristic.dll beehive4.dll 2>nul

clean-portable:
	-del /f /q lib\micro3.dll lib\micro4.dll lib\micro_mcts.dll lib\micro_mcts_heuristic.dll lib\beehive4.dll 2>nul
	-del /f /q lib\*.pyd lib\*.so 2>nul
	-rd /s /q build 2>nul
else
clean-native:
	rm -f micro3.so micro4.so micro_mcts.so micro_mcts_heuristic.so beehive4.so

clean-portable:
	rm -f lib/micro3.so lib/micro4.so lib/micro_mcts.so lib/micro_mcts_heuristic.so lib/beehive4.so
	rm -f lib/*.dll lib/*.pyd
	rm -rf build
endif

clean: clean-native clean-portable

// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Ashnas
//
// Minimal replacement for ScummVM's common/textconsole.h.  cell.cpp includes it
// (via cell.h) purely for the `warning()` declaration used by its four
// out-of-range getters; `error()` is declared for completeness and is never
// reached in normal play.  Both are *defined* in stauf_wasm.cpp.
//
// ScummVM's own header is GPLv3; this is an independent re-declaration written
// from the two call signatures cell.cpp actually uses, which keeps the vendored
// GPL surface at exactly cell.cpp + cell.h.
#pragma once

#include "common/scummsys.h"

void NORETURN_PRE error(MSVC_PRINTF const char *s, ...) GCC_PRINTF(1, 2) NORETURN_POST;
void warning(MSVC_PRINTF const char *s, ...) GCC_PRINTF(1, 2);

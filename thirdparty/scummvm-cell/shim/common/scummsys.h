// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Ashnas
//
// Minimal replacement for ScummVM's common/scummsys.h, supplying only the
// integer typedefs and printf-attribute macros that cell.{cpp,h} reference.
// Written from the usage sites, not copied from ScummVM, so that the vendored
// GPL surface stays limited to cell.cpp + cell.h.
#pragma once

#include <cstdint>
#include <cstddef>
#include <cassert>   // cell.cpp asserts on its board-stack bounds

typedef uint8_t  byte;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef int8_t   int8;

#define NORETURN_PRE
#define NORETURN_POST
#define MSVC_PRINTF
#define GCC_PRINTF(a, b)

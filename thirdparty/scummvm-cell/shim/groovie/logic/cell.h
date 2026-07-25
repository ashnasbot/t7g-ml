// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Ashnas
//
// Include redirect: cell.cpp asks for "groovie/logic/cell.h" (its path inside
// the ScummVM tree).  With shim/ on the include path that resolves here, and we
// forward to the vendored header sitting next to cell.cpp.
#pragma once
#include "../../../cell.h"

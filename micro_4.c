/*
 * micro_4.c - Minimax solver, pure material + clone-mobility eval.
 *
 * Leaf evaluator: (piece_count_diff × 10) + clone-mobility × 0.3.
 * Centrality is intentionally omitted: iterative deepening makes it
 * unnecessary and it can destabilise aspiration windows.
 *
 * See bb_core.h for the shared bitboard engine (board types, TT, move gen,
 * NegaScout skeleton).  Only leaf_eval() differs between solver variants.
 */
#include "bb_core.h"

static float leaf_eval(uint64_t mover_bb, uint64_t opp_bb) {
    float material = (float)(__builtin_popcountll(mover_bb)
                           - __builtin_popcountll(opp_bb)) * 10.0f;

    uint64_t occupied = mover_bb | opp_bb;
    uint64_t mr = 0, or_ = 0;
    uint64_t bb = mover_bb;
    while (bb) { int sq = __builtin_ctzll(bb); mr  |= neighbor1_mask[sq]; bb &= bb-1; }
    bb = opp_bb;
    while (bb) { int sq = __builtin_ctzll(bb); or_ |= neighbor1_mask[sq]; bb &= bb-1; }
    mr  &= ~occupied;
    or_ &= ~occupied;

    return material + (float)(__builtin_popcountll(mr) - __builtin_popcountll(or_)) * 0.3f;
}

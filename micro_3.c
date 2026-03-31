/*
 * micro_3.c — Minimax solver, material + centrality leaf evaluation.
 *
 * Leaf evaluator: (piece_count_diff × 10) + centrality bonus (Manhattan
 * distance from centre, ×0.1 per cell).  Fast, cache-friendly, no BFS.
 *
 * See t7g_core.h for the shared game kernel (board types, TT, move gen,
 * alpha-beta skeleton).  Only get_score() differs between solver variants.
 */
#include "t7g_core.h"

/* Heuristic score from player's perspective: material (×10) plus centrality
 * bonus (6 − Manhattan-distance-from-centre, ×0.1 per cell). */
static float get_score(uint8_t board[7][7], uint8_t player) {
    uint8_t opponent = (player == BLUE) ? GREEN : BLUE;
    float score = (count_cells(board, player) - count_cells(board, opponent)) * 10.0f;
    for (int y = 0; y < BOARD_SIZE; y++)
        for (int x = 0; x < BOARD_SIZE; x++) {
            float bonus = (6.0f - (float)(abs(x - 3) + abs(y - 3))) * 0.1f;
            if      (board[y][x] == player)   score += bonus;
            else if (board[y][x] == opponent) score -= bonus;
        }
    /* Mobility: reward constraining the opponent's clone options. */
    score += (count_clone_squares(board, player) - count_clone_squares(board, opponent)) * 0.3f;
    return score;
}

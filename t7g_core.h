/*
 * t7g_core.h — shared game kernel for Microscope (Ataxx 7x7) minimax solvers.
 *
 * Included by micro_3.c (material evaluation) and micro_4.c (BFS territory).
 * All symbols are file-scoped (static) so each translation unit that includes
 * this header gets its own independent copy of the TT and Zobrist state.
 *
 * Each .c file must define:
 *   static float get_score(uint8_t board[7][7], uint8_t player);
 * before or after including this header.  minimax_cached() (defined here)
 * calls get_score() via the forward declaration at the top of this file.
 */
#pragma once
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#define BOARD_SIZE 7

#define EMPTY 0
#define GREEN 1
#define BLUE  2

#define TT_SIZE (1 << 20)   // 1M entries (~32 MB)
#define TT_MASK (TT_SIZE - 1)
#define SCORE_INF     1000.0f
#define WIN_THRESHOLD  800.0f   /* scores above this are proven wins, not material */

/* Strip ply info before TT storage so win/loss scores are ply-neutral. */
static inline float tt_normalize(float score, int ply) {
    if (score >  WIN_THRESHOLD) return score + (float)ply;
    if (score < -WIN_THRESHOLD) return score - (float)ply;
    return score;
}

/* Re-apply current ply on retrieval to recover root-relative distance. */
static inline float tt_denormalize(float score, int ply) {
    if (score >  WIN_THRESHOLD) return score - (float)ply;
    if (score < -WIN_THRESHOLD) return score + (float)ply;
    return score;
}
#define BLUE_COLOUR_KEY 0x9e3779b97f4a7c15ULL

#define TT_EXACT       0
#define TT_LOWER_BOUND 1
#define TT_UPPER_BOUND 2

typedef struct {
    uint64_t hash;
    float    score;
    int8_t   depth;
    int8_t   bound;
    int16_t  best_move;
} TTEntry;

typedef struct { int16_t move; int16_t score; } ScoredMove;

// ---------------------------------------------------------------------------
// Static-per-TU globals
// ---------------------------------------------------------------------------

static TTEntry  *tt_table           = NULL;
static uint64_t  zobrist_table[49][3];
static bool      zobrist_initialized = false;

// ---------------------------------------------------------------------------
// Forward declaration — implemented differently by each .c file
// ---------------------------------------------------------------------------

static float get_score(uint8_t board[7][7], uint8_t player);

// ---------------------------------------------------------------------------
// Zobrist / TT lifecycle
// ---------------------------------------------------------------------------

static uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *state = x;
    return x;
}

static void init_zobrist(void) {
    if (zobrist_initialized) return;
    uint64_t seed = 0x123456789ABCDEF0ULL;
    for (int i = 0; i < 49; i++)
        for (int j = 0; j < 3; j++)
            zobrist_table[i][j] = xorshift64(&seed);
    zobrist_initialized = true;
}

void init_tt(void) {
    if (tt_table == NULL) {
        tt_table = (TTEntry*)calloc(TT_SIZE, sizeof(TTEntry));
        init_zobrist();
    }
}

void free_tt(void) {
    if (tt_table != NULL) { free(tt_table); tt_table = NULL; }
}

void clear_tt(void) {
    if (tt_table != NULL)
        memset(tt_table, 0, TT_SIZE * sizeof(TTEntry));
}

static uint64_t compute_zobrist_hash(uint8_t board[7][7]) {
    uint64_t hash = 0;
    for (int pos = 0; pos < 49; pos++) {
        uint8_t cell = board[pos / 7][pos % 7];
        if (cell != EMPTY) hash ^= zobrist_table[pos][cell];
    }
    return hash;
}

static bool tt_probe(uint64_t hash, int depth, float alpha, float beta,
                     float *score, int16_t *best_move, int ply) {
    TTEntry *entry = &tt_table[hash & TT_MASK];
    if (entry->hash != hash) return false;
    if (entry->depth < depth) return false;
    *best_move = entry->best_move;
    float s = tt_denormalize(entry->score, ply);
    if      (entry->bound == TT_EXACT)       { *score = s; return true; }
    else if (entry->bound == TT_LOWER_BOUND) { if (s >= beta)  { *score = s; return true; } }
    else if (entry->bound == TT_UPPER_BOUND) { if (s <= alpha) { *score = s; return true; } }
    return false;
}

static void tt_store(uint64_t hash, int depth, float score,
                     int8_t bound, int16_t best_move, int ply) {
    TTEntry *entry = &tt_table[hash & TT_MASK];
    if (entry->hash != hash || entry->depth <= depth) {
        entry->hash      = hash;
        entry->score     = tt_normalize(score, ply);
        entry->depth     = depth;
        entry->bound     = bound;
        entry->best_move = best_move;
    }
}

// ---------------------------------------------------------------------------
// Board helpers
// ---------------------------------------------------------------------------

static int count_cells(uint8_t board[7][7], uint8_t player) {
    int count = 0;
    for (int y = 0; y < BOARD_SIZE; y++)
        for (int x = 0; x < BOARD_SIZE; x++)
            if (board[y][x] == player) count++;
    return count;
}

/* Count empty cells adjacent (distance-1) to any piece of `player`.
 * Proxy for clone mobility: fewer empty neighbours = more constrained. */
static int count_clone_squares(uint8_t board[7][7], uint8_t player) {
    bool seen[7][7] = {0};
    int count = 0;
    for (int y = 0; y < BOARD_SIZE; y++)
        for (int x = 0; x < BOARD_SIZE; x++) {
            if (board[y][x] != player) continue;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) {
                    if (dy == 0 && dx == 0) continue;
                    int ny = y + dy, nx = x + dx;
                    if (ny < 0 || ny >= BOARD_SIZE || nx < 0 || nx >= BOARD_SIZE) continue;
                    if (board[ny][nx] == EMPTY && !seen[ny][nx]) {
                        seen[ny][nx] = true;
                        count++;
                    }
                }
        }
    return count;
}

static inline void board_from_python(bool game_board[7][7][2],
                                     uint8_t board[7][7]) {
    for (int y = 0; y < 7; y++)
        for (int x = 0; x < 7; x++)
            board[y][x] = game_board[y][x][1] ? BLUE
                        : game_board[y][x][0]  ? GREEN
                        :                        EMPTY;
}

// ---------------------------------------------------------------------------
// Move generation
// ---------------------------------------------------------------------------

static bool any_moves(bool valid_moves[7][7][5][5]) {
    return memchr(valid_moves, 1, sizeof(bool[7][7][5][5])) != NULL;
}

static void get_valid_moves(uint8_t board[7][7], uint8_t player,
                            bool valid_moves[7][7][5][5]) {
    memset(valid_moves, 0, sizeof(bool[7][7][5][5]));
    bool *moves = (bool *)valid_moves;
    for (int y = 0; y < BOARD_SIZE; y++) {
        for (int x = 0; x < BOARD_SIZE; x++) {
            if (board[y][x] != player) continue;
            int u_lo = (2 - x > 0) ? 2 - x : 0;
            int u_hi = (9 - x < 5) ? 9 - x : 5;
            int v_lo = (2 - y > 0) ? 2 - y : 0;
            int v_hi = (9 - y < 5) ? 9 - y : 5;
            for (int u = u_lo; u < u_hi; u++)
                for (int v = v_lo; v < v_hi; v++)
                    if (board[y + v - 2][x + u - 2] == EMPTY)
                        moves[25 * (7 * y + x) + 5 * v + u] = true;
        }
    }
}

static void make_move(uint8_t board[7][7], int action, uint8_t player) {
    int piece = action / 25, mv = action % 25;
    int from_x = piece % 7, from_y = piece / 7;
    int mv_x = (mv % 5) - 2, mv_y = (mv / 5) - 2;
    int to_x = from_x + mv_x, to_y = from_y + mv_y;

    if (abs(mv_x) == 2 || abs(mv_y) == 2) board[from_y][from_x] = EMPTY;
    board[to_y][to_x] = player;
    for (int cy = (to_y > 0 ? to_y-1 : 0); cy <= (to_y < 6 ? to_y+1 : 6); cy++)
        for (int cx = (to_x > 0 ? to_x-1 : 0); cx <= (to_x < 6 ? to_x+1 : 6); cx++)
            if (board[cy][cx] != player && board[cy][cx] != EMPTY)
                board[cy][cx] = player;
}

// ---------------------------------------------------------------------------
// Move ordering
// ---------------------------------------------------------------------------

static int count_captures(uint8_t board[7][7], int action, uint8_t player) {
    int piece = action / 25, mv = action % 25;
    int to_x  = (piece % 7) + (mv % 5) - 2;
    int to_y  = (piece / 7) + (mv / 5) - 2;
    if (to_x < 0 || to_x >= 7 || to_y < 0 || to_y >= 7) return 0;
    int captures = 0;
    for (int cy = (to_y > 0 ? to_y-1 : 0); cy <= (to_y < 6 ? to_y+1 : 6); cy++)
        for (int cx = (to_x > 0 ? to_x-1 : 0); cx <= (to_x < 6 ? to_x+1 : 6); cx++)
            if (board[cy][cx] != player && board[cy][cx] != EMPTY)
                captures++;
    return captures;
}

static int compare_scored_moves(const void *a, const void *b) {
    return ((ScoredMove*)b)->score - ((ScoredMove*)a)->score;
}

static int order_moves(uint8_t board[7][7], bool *moves, uint8_t player,
                       int16_t tt_best_move, ScoredMove *scored_moves) {
    int move_count = 0;
    for (int i = 0; i < 1225; i++) {
        if (!moves[i]) continue;
        scored_moves[move_count].move  = i;
        scored_moves[move_count].score = (i == tt_best_move)
            ? 10000
            : (int16_t)(count_captures(board, i, player) * 100);
        move_count++;
    }
    qsort(scored_moves, move_count, sizeof(ScoredMove), compare_scored_moves);
    return move_count;
}

// ---------------------------------------------------------------------------
// Alpha-beta minimax — calls get_score() for leaf evaluation
// ---------------------------------------------------------------------------

static float minimax_cached(uint8_t board[7][7], int depth,
                             float alpha, float beta,
                             bool max_player, uint8_t score_player,
                             uint64_t hash, int ply) {
    uint64_t keyed_hash = hash ^ (score_player == BLUE ? BLUE_COLOUR_KEY : 0ULL);

    float tt_score; int16_t tt_best_move = -1;
    if (tt_probe(keyed_hash, depth, alpha, beta, &tt_score, &tt_best_move, ply))
        return tt_score;

    uint8_t colour   = max_player ? score_player
                                  : (score_player == BLUE ? GREEN : BLUE);
    uint8_t opponent = (score_player == BLUE) ? GREEN : BLUE;
    float   bias     = max_player ? 0.5f : -0.5f;

    int sc_cells  = count_cells(board, score_player);
    int opp_cells = count_cells(board, opponent);
    if (sc_cells  == 0) { float s = -(SCORE_INF - ply); tt_store(keyed_hash, depth, s, TT_EXACT, -1, ply); return s; }
    if (opp_cells == 0) { float s =  (SCORE_INF - ply); tt_store(keyed_hash, depth, s, TT_EXACT, -1, ply); return s; }

    if (depth == 0) {
        float score = get_score(board, score_player) + bias;
        tt_store(keyed_hash, depth, score, TT_EXACT, -1, ply);
        return score;
    }

    bool valid_moves[7][7][5][5];
    get_valid_moves(board, colour, valid_moves);
    if (!any_moves(valid_moves)) {
        bool opp_moves[7][7][5][5];
        get_valid_moves(board, colour == BLUE ? GREEN : BLUE, opp_moves);
        if (!any_moves(opp_moves)) {
            float score = get_score(board, score_player) + bias;
            tt_store(keyed_hash, depth, score, TT_EXACT, -1, ply);
            return score;
        }
        return minimax_cached(board, depth - 1, alpha, beta,
                              !max_player, score_player, hash ^ 1ULL, ply + 1);
    }

    ScoredMove scored_moves[1225];
    int move_count = order_moves(board, (bool *)valid_moves, colour,
                                 tt_best_move, scored_moves);

    float   value     = max_player ? -SCORE_INF : SCORE_INF;
    int16_t best_move = -1;
    int8_t  bound     = TT_UPPER_BOUND;

    for (int idx = 0; idx < move_count; idx++) {
        int i = scored_moves[idx].move;
        uint8_t child[7][7];
        memcpy(child, board, sizeof(child));
        make_move(child, i, colour);

        uint64_t child_hash = compute_zobrist_hash(child);
        float eval = minimax_cached(child, depth - 1, alpha, beta,
                                    !max_player, score_player, child_hash, ply + 1);
        if (max_player) {
            if (eval > value) { value = eval; best_move = i; }
            alpha = fmaxf(alpha, eval);
        } else {
            if (eval < value) { value = eval; best_move = i; }
            beta = fminf(beta, eval);
        }
        if (beta <= alpha) { bound = TT_LOWER_BOUND; break; }
    }
    if (bound == TT_UPPER_BOUND) bound = TT_EXACT;

    tt_store(keyed_hash, depth, value, bound, best_move, ply);
    return value;
}

// ---------------------------------------------------------------------------
// Public API — identical signature in every solver, thin wrapper over kernel
// ---------------------------------------------------------------------------

int find_best_move(bool game_board[7][7][2], int depth, bool as_blue) {
    init_tt();
    uint8_t board[7][7];
    board_from_python(game_board, board);
    uint8_t player = as_blue ? BLUE : GREEN;

    bool valid_moves[7][7][5][5];
    get_valid_moves(board, player, valid_moves);
    ScoredMove scored_moves[1225];
    int move_count = order_moves(board, (bool *)valid_moves, player, -1, scored_moves);

    float best_score = -(SCORE_INF + 1.0f);
    int   residx     = -1;

    for (int idx = 0; idx < move_count; idx++) {
        int i = scored_moves[idx].move;
        uint8_t child[7][7];
        memcpy(child, board, sizeof(child));
        make_move(child, i, player);

        uint64_t hash  = compute_zobrist_hash(child);
        float    score = minimax_cached(child, depth - 1,
                                        -SCORE_INF, SCORE_INF,
                                        false, player, hash, 1);
        if (score > best_score) { best_score = score; residx = i; }
    }

    // In a proven loss, prefer the best-capturing clone over a jump.
    if (best_score < -WIN_THRESHOLD && residx >= 0) {
        for (int idx = 0; idx < move_count; idx++) {
            int i = scored_moves[idx].move;
            int mv = i % 25;
            if (abs((mv % 5) - 2) <= 1 && abs((mv / 5) - 2) <= 1) { residx = i; break; }
        }
    }
    return residx;
}

float minimax_score(bool game_board[7][7][2], int depth, bool as_blue) {
    init_tt();
    uint8_t board[7][7];
    board_from_python(game_board, board);
    uint8_t player = as_blue ? BLUE : GREEN;
    uint64_t hash  = compute_zobrist_hash(board);
    float score = minimax_cached(board, depth, -SCORE_INF, SCORE_INF,
                                 true, player, hash, 0);
    if (score >  WIN_THRESHOLD) return  1.0f;
    if (score < -WIN_THRESHOLD) return -1.0f;
    return tanhf(score / 100.0f);
}

void score_root_moves(bool game_board[7][7][2], int depth, bool as_blue,
                      float out_scores[1225]) {
    init_tt();
    uint8_t board[7][7];
    board_from_python(game_board, board);
    uint8_t player = as_blue ? BLUE : GREEN;

    for (int i = 0; i < 1225; i++) out_scores[i] = -2.0f;

    bool valid_moves[7][7][5][5];
    get_valid_moves(board, player, valid_moves);
    ScoredMove scored_moves[1225];
    int move_count = order_moves(board, (bool *)valid_moves, player, -1, scored_moves);

    for (int idx = 0; idx < move_count; idx++) {
        int i = scored_moves[idx].move;
        uint8_t child[7][7];
        memcpy(child, board, sizeof(child));
        make_move(child, i, player);

        uint64_t hash  = compute_zobrist_hash(child);
        float    score = minimax_cached(child, depth - 1,
                                        -SCORE_INF, SCORE_INF,
                                        false, player, hash, 1);
        if (score >  WIN_THRESHOLD) { out_scores[i] =  1.0f; continue; }
        if (score < -WIN_THRESHOLD) { out_scores[i] = -1.0f; continue; }
        out_scores[i] = tanhf(score / 100.0f);
    }
}

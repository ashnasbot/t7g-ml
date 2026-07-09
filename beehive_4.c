/*
 * beehive_4.c – Minimax solver for The Beehive Puzzle (hex Ataxx, 61-cell board).
 *
 * Board: 61 hexagonal cells, radius-4 axial grid.
 * Actions: cell_idx * 18 + dir_idx  (0-1097)
 *   dir_idx 0-5:  clone (source stays, destination gains piece)
 *   dir_idx 6-17: jump  (source clears, destination gains piece)
 * Capture: after landing, all 6 clone-adjacent opponent pieces flip to mover.
 *
 * Evaluation: material * 10  +  clone-mobility * 0.3
 * Search:     NegaScout (PVS), transposition table (1M entries),
 *             history heuristic, iterative deepening with aspiration windows.
 *
 * Public API (ctypes-compatible):
 *   void  init_tt(void)
 *   void  free_tt(void)
 *   void  clear_tt(void)
 *   int   find_best_move      (bool game_board[61][2], int depth, bool as_yellow)
 *   int   find_best_move_timed(bool game_board[61][2], int max_ms, bool as_yellow)
 *   float minimax_score       (bool game_board[61][2], int depth, bool as_yellow)
 *
 * game_board[ci][0] = Red channel, [ci][1] = Yellow channel (matches lib/beehive.py:
 *   RED=0, YELLOW=1).
 */
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* ── Constants ─────────────────────────────────────────────────────────── */

#define RADIUS     4
#define N_CELLS    61
#define N_DIRS     18           /* 6 clone + 12 jump */
#define N_ACTIONS  (N_CELLS * N_DIRS)   /* 1098 */

#define YELLOW_PLAYER  1
#define RED_PLAYER     2

#define TT_SIZE        (1 << 20)
#define TT_MASK        (TT_SIZE - 1)
#define SCORE_INF      1000.0f
#define WIN_THRESHOLD   800.0f
#define MAX_DEPTH        20

#define TT_EXACT        0
#define TT_LOWER_BOUND  1
#define TT_UPPER_BOUND  2

/* ── Types ─────────────────────────────────────────────────────────────── */

typedef struct {
    uint64_t hash;
    float    score;
    int8_t   depth;
    int8_t   bound;
    int16_t  best_move;
} TTEntry;

typedef struct { int16_t move; int16_t score; } ScoredMove;

typedef struct {
    uint64_t player_bb;   /* mover's pieces after the move */
    uint64_t opp_bb;      /* opponent's pieces after the move (with captures removed) */
    uint64_t hash;
} MoveResult;

/* ── File-scoped state ─────────────────────────────────────────────────── */

int last_depth_reached = 0;  /* updated by find_best_move_timed after each completed depth */

static TTEntry  *tt_table  = NULL;
static uint64_t  zobrist_piece[N_CELLS][3];
static uint64_t  zobrist_turn;
static int32_t   history[N_CELLS][N_CELLS];
static bool      tables_initialized = false;

/* Geometry tables built once in init_tables() */
static int8_t   cells_q[N_CELLS];
static int8_t   cells_r[N_CELLS];
static int8_t   dest_tbl[N_CELLS][N_DIRS];  /* destination cell index, or -1 */
static uint64_t neighbor1_mask[N_CELLS];     /* bitmask: 6 clone-neighbours (for capture) */
static uint64_t neighbor2_mask[N_CELLS];     /* bitmask: all valid destinations (for move gen) */
static int8_t   cidx[9][9];                  /* [q+4][r+4] → cell index, -1 if invalid */

/* Hex direction vectors – must match lib/beehive.py exactly */
static const int CLONE_DQ[6]  = {  1, -1,  0,  0,  1, -1 };
static const int CLONE_DR[6]  = {  0,  0,  1, -1, -1,  1 };
static const int JUMP_DQ[12]  = {  2, -2,  0,  0,  2, -2,  1, -1,  1, -1,  2, -2 };
static const int JUMP_DR[12]  = {  0,  0,  2, -2, -2,  2,  1, -1, -2,  2, -1,  1 };

/* ── PRNG ─────────────────────────────────────────────────────────────── */

static uint64_t xorshift64(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return *s = x;
}

/* ── One-time initialisation ───────────────────────────────────────────── */

static void init_tables(void) {
    if (tables_initialized) return;

    /* Build CELLS list in same order as Python (r outer, q inner). */
    memset(cidx, -1, sizeof(cidx));
    int n = 0;
    for (int r = -RADIUS; r <= RADIUS; r++)
        for (int q = -RADIUS; q <= RADIUS; q++)
            if (abs(q + r) <= RADIUS) {
                cells_q[n] = (int8_t)q;
                cells_r[n] = (int8_t)r;
                cidx[q + RADIUS][r + RADIUS] = (int8_t)n;
                n++;
            }
    /* n == 61 */

    /* Destination table and neighbour bitmasks. */
    for (int ci = 0; ci < N_CELLS; ci++) {
        int q = cells_q[ci], r = cells_r[ci];
        neighbor1_mask[ci] = 0;
        neighbor2_mask[ci] = 0;

        for (int di = 0; di < 6; di++) {
            int nq = q + CLONE_DQ[di], nr = r + CLONE_DR[di];
            int8_t dest = -1;
            if (abs(nq) <= RADIUS && abs(nr) <= RADIUS && abs(nq + nr) <= RADIUS)
                dest = cidx[nq + RADIUS][nr + RADIUS];
            dest_tbl[ci][di] = dest;
            if (dest >= 0) {
                neighbor1_mask[ci] |= 1ULL << dest;
                neighbor2_mask[ci] |= 1ULL << dest;
            }
        }
        for (int di = 0; di < 12; di++) {
            int nq = q + JUMP_DQ[di], nr = r + JUMP_DR[di];
            int8_t dest = -1;
            if (abs(nq) <= RADIUS && abs(nr) <= RADIUS && abs(nq + nr) <= RADIUS)
                dest = cidx[nq + RADIUS][nr + RADIUS];
            dest_tbl[ci][6 + di] = dest;
            if (dest >= 0)
                neighbor2_mask[ci] |= 1ULL << dest;
        }
    }

    /* Zobrist tables. */
    uint64_t seed = 0x1A2B3C4D5E6F7A8BULL;
    for (int i = 0; i < N_CELLS; i++) {
        zobrist_piece[i][0]             = 0;
        zobrist_piece[i][YELLOW_PLAYER] = xorshift64(&seed);
        zobrist_piece[i][RED_PLAYER]    = xorshift64(&seed);
    }
    zobrist_turn = xorshift64(&seed);

    tables_initialized = true;
}

/* ── TT lifecycle ──────────────────────────────────────────────────────── */

void init_tt(void) {
    init_tables();
    if (tt_table == NULL)
        tt_table = (TTEntry *)calloc(TT_SIZE, sizeof(TTEntry));
    memset(history, 0, sizeof(history));
}

void free_tt(void) {
    if (tt_table != NULL) { free(tt_table); tt_table = NULL; }
}

void clear_tt(void) {
    if (tt_table != NULL)
        memset(tt_table, 0, TT_SIZE * sizeof(TTEntry));
}

/* ── TT helpers ────────────────────────────────────────────────────────── */

static inline float tt_normalize(float score, int ply) {
    if (score >  WIN_THRESHOLD) return score + (float)ply;
    if (score < -WIN_THRESHOLD) return score - (float)ply;
    return score;
}

static inline float tt_denormalize(float score, int ply) {
    if (score >  WIN_THRESHOLD) return score - (float)ply;
    if (score < -WIN_THRESHOLD) return score + (float)ply;
    return score;
}

static bool tt_probe(uint64_t hash, int depth, float alpha, float beta,
                     float *score_out, int16_t *best_move_out, int ply) {
    TTEntry *e = &tt_table[hash & TT_MASK];
    if (e->hash != hash) { *best_move_out = -1; return false; }
    *best_move_out = e->best_move;
    if (e->depth < depth) return false;
    float s = tt_denormalize(e->score, ply);
    if      (e->bound == TT_EXACT                    ) { *score_out = s; return true; }
    else if (e->bound == TT_LOWER_BOUND && s >= beta ) { *score_out = s; return true; }
    else if (e->bound == TT_UPPER_BOUND && s <= alpha) { *score_out = s; return true; }
    return false;
}

static void tt_store(uint64_t hash, int depth, float score, int8_t bound,
                     int16_t best_move, int ply) {
    TTEntry *e = &tt_table[hash & TT_MASK];
    if (e->hash != hash || e->depth <= depth) {
        e->hash      = hash;
        e->score     = tt_normalize(score, ply);
        e->depth     = (int8_t)depth;
        e->bound     = bound;
        e->best_move = best_move;
    }
}

/* ── Board helpers ─────────────────────────────────────────────────────── */

static uint64_t compute_hash(uint64_t yellow_bb, uint64_t red_bb, bool yellow_to_move) {
    uint64_t hash = 0;
    uint64_t bb = yellow_bb;
    while (bb) { int sq = __builtin_ctzll(bb); hash ^= zobrist_piece[sq][YELLOW_PLAYER]; bb &= bb - 1; }
    bb = red_bb;
    while (bb) { int sq = __builtin_ctzll(bb); hash ^= zobrist_piece[sq][RED_PLAYER];    bb &= bb - 1; }
    if (yellow_to_move) hash ^= zobrist_turn;
    return hash;
}

static void board_from_python(bool game_board[N_CELLS][2],
                              uint64_t *yellow_bb, uint64_t *red_bb) {
    *yellow_bb = 0; *red_bb = 0;
    for (int ci = 0; ci < N_CELLS; ci++) {
        if (game_board[ci][1]) *yellow_bb |= 1ULL << ci;
        if (game_board[ci][0]) *red_bb    |= 1ULL << ci;
    }
}

/* ── Move generation ───────────────────────────────────────────────────── */

static bool has_moves_bb(uint64_t mover_bb, uint64_t opp_bb) {
    uint64_t occupied = mover_bb | opp_bb;
    uint64_t empty    = ~occupied;
    uint64_t pieces   = mover_bb;
    while (pieces) {
        int ci = __builtin_ctzll(pieces); pieces &= pieces - 1;
        /* Clone moves: any empty 1-step neighbour */
        if (neighbor1_mask[ci] & empty) return true;
        /* Jump moves: empty 2-step dest with opponent adj and no own adj */
        uint64_t jumps = (neighbor2_mask[ci] & ~neighbor1_mask[ci]) & empty;
        while (jumps) {
            int dest = __builtin_ctzll(jumps); jumps &= jumps - 1;
            if ((neighbor1_mask[dest] & opp_bb) && !(neighbor1_mask[dest] & mover_bb))
                return true;
        }
    }
    return false;
}

static int gen_moves(uint64_t mover_bb, uint64_t opp_bb, int16_t out_moves[N_ACTIONS]) {
    int      count    = 0;
    uint64_t occupied = mover_bb | opp_bb;
    uint64_t pieces   = mover_bb;
    while (pieces) {
        int ci = __builtin_ctzll(pieces); pieces &= pieces - 1;
        for (int di = 0; di < N_DIRS; di++) {
            int dest = (int)(int8_t)dest_tbl[ci][di];
            if (dest < 0) continue;
            if ((occupied >> dest) & 1) continue;
            if (di >= 6) {
                /* Jump: only valid when dest has opponent neighbours (must
                 * capture) and no own neighbours (clone would dominate). */
                uint64_t adj = neighbor1_mask[dest];
                if (!(adj & opp_bb)) continue;
                if (  adj & mover_bb) continue;
            }
            out_moves[count++] = (int16_t)(ci * N_DIRS + di);
        }
    }
    return count;
}

/* ── Make move ─────────────────────────────────────────────────────────── */

static MoveResult make_move_bb(uint64_t mover_bb, uint64_t opp_bb,
                                uint64_t hash, int action, int mover_color) {
    int opp_color = (mover_color == YELLOW_PLAYER) ? RED_PLAYER : YELLOW_PLAYER;
    int ci   = action / N_DIRS;
    int di   = action % N_DIRS;
    int dest = (int)(int8_t)dest_tbl[ci][di];

    if (di >= 6) {   /* jump: vacate source cell */
        mover_bb &= ~(1ULL << ci);
        hash     ^= zobrist_piece[ci][mover_color];
    }

    mover_bb |= 1ULL << dest;
    hash     ^= zobrist_piece[dest][mover_color];

    /* Capture: flip all 6 clone-adjacent opponent pieces. */
    uint64_t captured = neighbor1_mask[dest] & opp_bb;
    opp_bb   ^= captured;
    mover_bb |= captured;
    uint64_t cap = captured;
    while (cap) {
        int sq = __builtin_ctzll(cap); cap &= cap - 1;
        hash ^= zobrist_piece[sq][opp_color];
        hash ^= zobrist_piece[sq][mover_color];
    }

    hash ^= zobrist_turn;   /* flip the turn bit */

    MoveResult r = { mover_bb, opp_bb, hash };
    return r;
}

/* ── Move ordering ─────────────────────────────────────────────────────── */

static int compare_scored_moves(const void *a, const void *b) {
    return (int)((ScoredMove *)b)->score - (int)((ScoredMove *)a)->score;
}

static void order_moves(int16_t *moves, int count, uint64_t opp_bb,
                        int16_t tt_best, ScoredMove *out) {
    for (int i = 0; i < count; i++) {
        int action   = moves[i];
        int ci       = action / N_DIRS;
        int di       = action % N_DIRS;
        int dest     = (int)(int8_t)dest_tbl[ci][di];
        int is_clone = (di < 6);

        int16_t sc;
        if (action == (int)tt_best) {
            sc = 30000;
        } else {
            int caps = (dest >= 0) ? __builtin_popcountll(neighbor1_mask[dest] & opp_bb) : 0;
            int hist = (dest >= 0) ? (history[ci][dest] >> 8) : 0;
            int raw  = caps * 200 + is_clone * 80 + hist;
            sc = (int16_t)(raw < 29000 ? raw : 29000);
        }
        out[i] = (ScoredMove){ (int16_t)action, sc };
    }
    qsort(out, count, sizeof(ScoredMove), compare_scored_moves);
}

/* ── Leaf evaluation ───────────────────────────────────────────────────── */

static float leaf_eval(uint64_t mover_bb, uint64_t opp_bb) {
    float material = (float)(__builtin_popcountll(mover_bb)
                           - __builtin_popcountll(opp_bb)) * 10.0f;

    uint64_t occupied = mover_bb | opp_bb;
    uint64_t mr = 0, or_ = 0;
    uint64_t bb = mover_bb;
    while (bb) { int ci = __builtin_ctzll(bb); mr |= neighbor1_mask[ci]; bb &= bb - 1; }
    bb = opp_bb;
    while (bb) { int ci = __builtin_ctzll(bb); or_ |= neighbor1_mask[ci]; bb &= bb - 1; }

    mr  &= ~occupied;
    or_ &= ~occupied;

    return material + (float)(__builtin_popcountll(mr) - __builtin_popcountll(or_)) * 0.3f;
}

/* ── NegaScout (PVS) ───────────────────────────────────────────────────── */

static float negamax(uint64_t mover_bb, uint64_t opp_bb, int mover_color,
                     int depth, float alpha, float beta, uint64_t hash, int ply) {
    float   tt_score;
    int16_t tt_best;
    if (tt_probe(hash, depth, alpha, beta, &tt_score, &tt_best, ply))
        return tt_score;

    if (mover_bb == 0) {
        float s = -(SCORE_INF - (float)ply);
        tt_store(hash, depth, s, TT_EXACT, -1, ply);
        return s;
    }
    if (opp_bb == 0) {
        float s = SCORE_INF - (float)ply;
        tt_store(hash, depth, s, TT_EXACT, -1, ply);
        return s;
    }

    if (depth == 0) {
        float s = leaf_eval(mover_bb, opp_bb) + 0.5f;
        tt_store(hash, depth, s, TT_EXACT, -1, ply);
        return s;
    }

    int16_t moves[N_ACTIONS];
    int move_count = gen_moves(mover_bb, opp_bb, moves);

    if (move_count == 0) {
        if (!has_moves_bb(opp_bb, mover_bb)) {
            float s = leaf_eval(mover_bb, opp_bb) + 0.5f;
            tt_store(hash, depth, s, TT_EXACT, -1, ply);
            return s;
        }
        /* Current player passes; XOR turn key but don't decrement depth. */
        int opp_color = (mover_color == YELLOW_PLAYER) ? RED_PLAYER : YELLOW_PLAYER;
        return -negamax(opp_bb, mover_bb, opp_color,
                        depth - 1, -beta, -alpha, hash ^ zobrist_turn, ply + 1);
    }

    ScoredMove scored[N_ACTIONS];
    order_moves(moves, move_count, opp_bb, tt_best, scored);

    float   value     = -SCORE_INF;
    int16_t best_move = -1;
    int8_t  bound     = TT_UPPER_BOUND;
    int     opp_color = (mover_color == YELLOW_PLAYER) ? RED_PLAYER : YELLOW_PLAYER;

    for (int i = 0; i < move_count; i++) {
        int action = scored[i].move;
        MoveResult child = make_move_bb(mover_bb, opp_bb, hash, action, mover_color);

        float eval;
        if (i == 0) {
            eval = -negamax(child.opp_bb, child.player_bb, opp_color,
                            depth - 1, -beta, -alpha, child.hash, ply + 1);
        } else {
            eval = -negamax(child.opp_bb, child.player_bb, opp_color,
                            depth - 1, -alpha - 1.0f, -alpha, child.hash, ply + 1);
            if (eval > alpha && eval < beta)
                eval = -negamax(child.opp_bb, child.player_bb, opp_color,
                                depth - 1, -beta, -alpha, child.hash, ply + 1);
        }

        if (eval > value) { value = eval; best_move = (int16_t)action; }
        if (eval > alpha) { alpha = eval; bound = TT_EXACT; }
        if (alpha >= beta) {
            int ci   = action / N_DIRS;
            int dest = (int)(int8_t)dest_tbl[ci][action % N_DIRS];
            if (dest >= 0) {
                int hval = 1 << (depth < 19 ? depth : 19);
                history[ci][dest] += hval;
            }
            bound = TT_LOWER_BOUND;
            break;
        }
    }

    tt_store(hash, depth, value, bound, best_move, ply);
    return value;
}

/* ── Root search ───────────────────────────────────────────────────────── */

static float root_negascout(uint64_t mover_bb, uint64_t opp_bb, int mover_color,
                             int depth, float alpha, float beta, uint64_t hash,
                             clock_t start, double deadline_ms,
                             int16_t *best_move_out) {
    int16_t moves[N_ACTIONS];
    int move_count = gen_moves(mover_bb, opp_bb, moves);
    if (move_count == 0) { *best_move_out = -1; return -(SCORE_INF - 1.0f); }

    float   dummy;
    int16_t tt_best = -1;
    tt_probe(hash, depth, -SCORE_INF, SCORE_INF, &dummy, &tt_best, 0);

    ScoredMove scored[N_ACTIONS];
    order_moves(moves, move_count, opp_bb, tt_best, scored);

    float   value     = -SCORE_INF;
    int16_t best_move = scored[0].move;
    int     opp_color = (mover_color == YELLOW_PLAYER) ? RED_PLAYER : YELLOW_PLAYER;

    for (int i = 0; i < move_count; i++) {
        int action = scored[i].move;
        MoveResult child = make_move_bb(mover_bb, opp_bb, hash, action, mover_color);

        float eval;
        if (i == 0) {
            eval = -negamax(child.opp_bb, child.player_bb, opp_color,
                            depth - 1, -beta, -alpha, child.hash, 1);
        } else {
            eval = -negamax(child.opp_bb, child.player_bb, opp_color,
                            depth - 1, -alpha - 1.0f, -alpha, child.hash, 1);
            if (eval > alpha && eval < beta)
                eval = -negamax(child.opp_bb, child.player_bb, opp_color,
                                depth - 1, -beta, -alpha, child.hash, 1);
        }

        if (eval > value) { value = eval; best_move = (int16_t)action; }
        if (eval > alpha) alpha = eval;
        if (alpha >= beta) break;

        if (deadline_ms > 0.0) {
            double el = (double)(clock() - start) * 1000.0 / CLOCKS_PER_SEC;
            if (el >= deadline_ms) break;
        }
    }

    /* In a proven loss, prefer cloning over jumping to stay connected. */
    if (value < -WIN_THRESHOLD) {
        for (int i = 0; i < move_count; i++) {
            if ((scored[i].move % N_DIRS) < 6) { best_move = scored[i].move; break; }
        }
    }

    *best_move_out = best_move;
    return value;
}

/* ── Public API ─────────────────────────────────────────────────────────── */

int find_best_move(bool game_board[N_CELLS][2], int depth, bool as_yellow) {
    init_tt();
    uint64_t yellow_bb, red_bb;
    board_from_python(game_board, &yellow_bb, &red_bb);
    int      mover_color = as_yellow ? YELLOW_PLAYER : RED_PLAYER;
    uint64_t mover_bb    = as_yellow ? yellow_bb : red_bb;
    uint64_t opp_bb      = as_yellow ? red_bb    : yellow_bb;
    uint64_t hash        = compute_hash(yellow_bb, red_bb, as_yellow);

    int16_t best_move = -1;
    root_negascout(mover_bb, opp_bb, mover_color, depth,
                   -SCORE_INF, SCORE_INF, hash, 0, -1.0, &best_move);
    return (int)best_move;
}

int find_best_move_timed(bool game_board[N_CELLS][2], int max_ms, bool as_yellow) {
    init_tt();
    uint64_t yellow_bb, red_bb;
    board_from_python(game_board, &yellow_bb, &red_bb);
    int      mover_color = as_yellow ? YELLOW_PLAYER : RED_PLAYER;
    uint64_t mover_bb    = as_yellow ? yellow_bb : red_bb;
    uint64_t opp_bb      = as_yellow ? red_bb    : yellow_bb;
    uint64_t hash        = compute_hash(yellow_bb, red_bb, as_yellow);

    clock_t start      = clock();
    int16_t best_move  = -1;
    float   best_score = 0.0f;
    float   asp_delta  = 20.0f;
    /* Track scores by parity: even/odd depths oscillate ~60 pts; same-parity is stable. */
    float   parity_score[2] = {0.0f, 0.0f};
    int     parity_seen[2]  = {0, 0};

    for (int d = 1; d <= MAX_DEPTH; d++) {
        for (int i = 0; i < N_CELLS; i++)
            for (int j = 0; j < N_CELLS; j++)
                history[i][j] >>= 1;

        int par = d & 1;
        float center = parity_seen[par] ? parity_score[par] : best_score;
        float lo = (d <= 2) ? -SCORE_INF : center - asp_delta;
        float hi = (d <= 2) ? +SCORE_INF : center + asp_delta;
        float cur_delta = asp_delta;

        int16_t move_this_depth = best_move;

        while (1) {
            float score = root_negascout(mover_bb, opp_bb, mover_color, d,
                                         lo, hi, hash, start, (double)max_ms,
                                         &move_this_depth);
            if (score <= lo && lo > -SCORE_INF) {
                lo -= cur_delta * 2.0f;
                if (lo < -SCORE_INF) lo = -SCORE_INF;
                cur_delta *= 2.0f;
            } else if (score >= hi && hi < SCORE_INF) {
                hi += cur_delta * 2.0f;
                if (hi > SCORE_INF) hi = SCORE_INF;
                cur_delta *= 2.0f;
            } else {
                best_score = score;
                parity_score[par] = score;
                parity_seen[par] = 1;
                break;
            }
            double el = (double)(clock() - start) * 1000.0 / CLOCKS_PER_SEC;
            if (el >= (double)max_ms) return (int)best_move;
        }

        best_move = move_this_depth;
        asp_delta = 20.0f;
        last_depth_reached = d;

        double elapsed = (double)(clock() - start) * 1000.0 / CLOCKS_PER_SEC;
        if (elapsed >= (double)max_ms) break;
        if (best_score > WIN_THRESHOLD || best_score < -WIN_THRESHOLD) break;
    }

    return (int)best_move;
}

float minimax_score(bool game_board[N_CELLS][2], int depth, bool as_yellow) {
    init_tt();
    uint64_t yellow_bb, red_bb;
    board_from_python(game_board, &yellow_bb, &red_bb);
    int      mover_color = as_yellow ? YELLOW_PLAYER : RED_PLAYER;
    uint64_t mover_bb    = as_yellow ? yellow_bb : red_bb;
    uint64_t opp_bb      = as_yellow ? red_bb    : yellow_bb;
    uint64_t hash        = compute_hash(yellow_bb, red_bb, as_yellow);

    float score = negamax(mover_bb, opp_bb, mover_color, depth,
                          -SCORE_INF, SCORE_INF, hash, 0);
    if (score >  WIN_THRESHOLD) return  1.0f;
    if (score < -WIN_THRESHOLD) return -1.0f;
    return tanhf(score / 100.0f);
}

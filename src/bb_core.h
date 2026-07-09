/*
 * bb_core.h - shared bitboard engine for Microscope (Ataxx 7x7) minimax solvers.
 *
 * Included by micro_3.c (material + centrality eval) and micro_4.c (material only).
 * All symbols are file-scoped (static) so each translation unit that includes
 * this header gets its own independent copy of the TT and Zobrist state.
 *
 * Board representation: two uint64_t bitmaps (green_bb, blue_bb), low 49 bits,
 *   row-major: bit i = cell (i/7, i%7).
 *
 * Search: pure negamax + PVS (NegaScout), iterative deepening, aspiration
 *   windows, transposition table with incremental Zobrist, history heuristic.
 *
 * Each .c file must define:
 *   static float leaf_eval(uint64_t mover_bb, uint64_t opp_bb);
 * before or after including this header.  negamax() (defined here) calls it
 * via the forward declaration below.
 *
 * centrality_table[49] is initialised here and available to any leaf_eval
 * that wants it.  Value: (6 - Manhattan distance from centre cell (3,3)).
 *
 * Public API (ctypes-compatible):
 *   int   find_best_move      (bool game_board[7][7][2], int depth, bool as_blue)
 *   float minimax_score       (bool game_board[7][7][2], int depth, bool as_blue)
 *   void  score_root_moves    (bool game_board[7][7][2], int depth, bool as_blue,
 *                              float out_scores[1225])
 *   int   find_best_move_timed(bool game_board[7][7][2], int max_ms, bool as_blue)
 */
#pragma once
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/*  Constants  */

#define GREEN_PLAYER  1
#define BLUE_PLAYER   2

#define TT_SIZE        (1 << 20)
#define TT_MASK        (TT_SIZE - 1)
#define SCORE_INF      1000.0f
#define WIN_THRESHOLD   800.0f
#define MAX_DEPTH        20

#define TT_EXACT        0
#define TT_LOWER_BOUND  1
#define TT_UPPER_BOUND  2

/*  Types  */

typedef struct {
    uint64_t hash;
    float    score;
    int8_t   depth;
    int8_t   bound;
    int16_t  best_move;
} TTEntry;

typedef struct { int16_t move; int16_t score; } ScoredMove;

typedef struct { uint64_t green_bb; uint64_t blue_bb; } BBState;

typedef struct {
    uint64_t player_bb;
    uint64_t opp_bb;
    uint64_t hash;
} MoveResult;

/*  File-scoped globals  */

static TTEntry  *tt_table             = NULL;
static uint64_t  zobrist_piece[49][3];
static uint64_t  zobrist_turn;
static uint64_t  neighbor1_mask[49];
static uint64_t  neighbor2_mask[49];
static float     centrality_table[49]; /* (6 - Manhattan dist from centre)  */
static int32_t   history[49][49];
static bool      tables_initialized = false;

/* ── Optional neural policy prior (ONNX) ──────────────────────────────── */
#ifdef BB_USE_ONNX
#  include "onnxruntime_c_api.h"
static const OrtApi *g_ort       = NULL;
static OrtEnv       *ort_env     = NULL;
static OrtSession   *ort_session = NULL;
#endif
static float  policy_prior_buf[1225];
static bool   policy_prior_valid   = false;
static bool   policy_prior_enabled = false;  /* toggled by set_active; run respects this */

/*  Forward declaration - implemented differently by each .c file  */

static float leaf_eval(uint64_t mover_bb, uint64_t opp_bb);

/*  TT score normalization  */

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

/*  One-time initialization  */

static uint64_t xorshift64(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return *s = x;
}

static void init_tables(void) {
    if (tables_initialized) return;

    uint64_t seed = 0xA7B3C1D9E5F20846ULL;
    for (int i = 0; i < 49; i++) {
        zobrist_piece[i][0] = 0;
        zobrist_piece[i][GREEN_PLAYER] = xorshift64(&seed);
        zobrist_piece[i][BLUE_PLAYER]  = xorshift64(&seed);
    }
    zobrist_turn = xorshift64(&seed);

    for (int sq = 0; sq < 49; sq++) {
        int y = sq / 7, x = sq % 7;
        neighbor1_mask[sq] = 0;
        neighbor2_mask[sq] = 0;
        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                if (dy == 0 && dx == 0) continue;
                int ny = y + dy, nx = x + dx;
                if (ny < 0 || ny >= 7 || nx < 0 || nx >= 7) continue;
                int nsq = ny * 7 + nx;
                neighbor2_mask[sq] |= (1ULL << nsq);
                if (abs(dy) <= 1 && abs(dx) <= 1)
                    neighbor1_mask[sq] |= (1ULL << nsq);
            }
        }
        centrality_table[sq] = 6.0f - (float)(abs(x - 3) + abs(y - 3));
    }

    tables_initialized = true;
}

/*  TT lifecycle  */

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

static bool tt_probe(uint64_t hash, int depth, float alpha, float beta,
                     float *score_out, int16_t *best_move_out, int ply) {
    TTEntry *e = &tt_table[hash & TT_MASK];
    if (e->hash != hash) { *best_move_out = -1; return false; }
    *best_move_out = e->best_move;
    if (e->depth < depth) return false;
    float s = tt_denormalize(e->score, ply);
    if      (e->bound == TT_EXACT                        ) { *score_out = s; return true; }
    else if (e->bound == TT_LOWER_BOUND && s >= beta     ) { *score_out = s; return true; }
    else if (e->bound == TT_UPPER_BOUND && s <= alpha    ) { *score_out = s; return true; }
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

/*  Board conversion  */

static BBState board_from_python(bool game_board[7][7][2]) {
    BBState s = {0, 0};
    for (int pos = 0; pos < 49; pos++) {
        int y = pos / 7, x = pos % 7;
        if (game_board[y][x][0]) s.green_bb |= (1ULL << pos);
        if (game_board[y][x][1]) s.blue_bb  |= (1ULL << pos);
    }
    return s;
}

/*  Zobrist hash  */

static uint64_t compute_hash(uint64_t green_bb, uint64_t blue_bb, bool blue_to_move) {
    uint64_t hash = 0;
    uint64_t bb = green_bb;
    while (bb) { int sq = __builtin_ctzll(bb); hash ^= zobrist_piece[sq][GREEN_PLAYER]; bb &= bb-1; }
    bb = blue_bb;
    while (bb) { int sq = __builtin_ctzll(bb); hash ^= zobrist_piece[sq][BLUE_PLAYER];  bb &= bb-1; }
    if (blue_to_move) hash ^= zobrist_turn;
    return hash;
}

/*  Move generation  */

static bool has_moves_bb(uint64_t mover_bb, uint64_t opp_bb) {
    uint64_t occupied = mover_bb | opp_bb;
    uint64_t pieces   = mover_bb;
    while (pieces) {
        int sq = __builtin_ctzll(pieces); pieces &= pieces - 1;
        if (neighbor2_mask[sq] & ~occupied) return true;
    }
    return false;
}

static int gen_moves(uint64_t mover_bb, uint64_t opp_bb, int16_t out_moves[1225]) {
    int count    = 0;
    uint64_t occupied = mover_bb | opp_bb;
    uint64_t pieces   = mover_bb;
    while (pieces) {
        int from_sq = __builtin_ctzll(pieces); pieces &= pieces - 1;
        int from_y  = from_sq / 7, from_x = from_sq % 7;
        uint64_t dests = neighbor2_mask[from_sq] & ~occupied;
        while (dests) {
            int to_sq = __builtin_ctzll(dests); dests &= dests - 1;
            int dy = (to_sq / 7) - from_y;
            int dx = (to_sq % 7) - from_x;
            out_moves[count++] = (int16_t)(from_sq * 25 + (dy + 2) * 5 + (dx + 2));
        }
    }
    return count;
}

static MoveResult make_move_bb(uint64_t mover_bb, uint64_t opp_bb,
                                uint64_t hash, int action, int mover_color) {
    int opp_color = 3 - mover_color;
    int from_sq   = action / 25;
    int mv        = action % 25;
    int mv_x      = (mv % 5) - 2;
    int mv_y      = (mv / 5) - 2;
    int to_sq     = (from_sq / 7 + mv_y) * 7 + (from_sq % 7 + mv_x);

    if (abs(mv_x) == 2 || abs(mv_y) == 2) {
        mover_bb &= ~(1ULL << from_sq);
        hash     ^= zobrist_piece[from_sq][mover_color];
    }

    mover_bb |= (1ULL << to_sq);
    hash     ^= zobrist_piece[to_sq][mover_color];

    uint64_t captured = neighbor1_mask[to_sq] & opp_bb;
    opp_bb   ^= captured;
    mover_bb |= captured;
    uint64_t cap = captured;
    while (cap) {
        int sq = __builtin_ctzll(cap); cap &= cap - 1;
        hash ^= zobrist_piece[sq][opp_color];
        hash ^= zobrist_piece[sq][mover_color];
    }

    hash ^= zobrist_turn;

    MoveResult r = { mover_bb, opp_bb, hash };
    return r;
}

/*  Move ordering  */

static int compare_scored_moves(const void *a, const void *b) {
    return (int)((ScoredMove *)b)->score - (int)((ScoredMove *)a)->score;
}

static void order_moves(int16_t *moves, int count, uint64_t opp_bb,
                        int16_t tt_best, ScoredMove *out) {
    for (int i = 0; i < count; i++) {
        int action  = moves[i];
        int from_sq = action / 25;
        int mv      = action % 25;
        int to_sq   = (from_sq / 7 + (mv / 5) - 2) * 7 + (from_sq % 7 + (mv % 5) - 2);

        int16_t sc;
        if (action == (int)tt_best) {
            sc = 30000;
        } else {
            int caps   = __builtin_popcountll(neighbor1_mask[to_sq] & opp_bb);
            int hist   = history[from_sq][to_sq] >> 8;
            int sc_raw = caps * 200 + hist;
            sc = (int16_t)(sc_raw < 29000 ? sc_raw : 29000);
        }
        out[i] = (ScoredMove){ (int16_t)action, sc };
    }
    qsort(out, count, sizeof(ScoredMove), compare_scored_moves);
}

/*  Core negamax with NegaScout (PVS)  */

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
        float s =  (SCORE_INF - (float)ply);
        tt_store(hash, depth, s, TT_EXACT, -1, ply);
        return s;
    }

    if (depth == 0) {
        float s = leaf_eval(mover_bb, opp_bb) + 0.5f;
        tt_store(hash, depth, s, TT_EXACT, -1, ply);
        return s;
    }

    int16_t moves[1225];
    int move_count = gen_moves(mover_bb, opp_bb, moves);

    if (move_count == 0) {
        if (!has_moves_bb(opp_bb, mover_bb)) {
            float s = leaf_eval(mover_bb, opp_bb) + 0.5f;
            tt_store(hash, depth, s, TT_EXACT, -1, ply);
            return s;
        }
        return -negamax(opp_bb, mover_bb, 3 - mover_color,
                        depth - 1, -beta, -alpha,
                        hash ^ zobrist_turn, ply + 1);
    }

    ScoredMove scored[1225];
    order_moves(moves, move_count, opp_bb, tt_best, scored);

    float   value     = -SCORE_INF;
    int16_t best_move = -1;
    int8_t  bound     = TT_UPPER_BOUND;
    int     opp_color = 3 - mover_color;

    for (int i = 0; i < move_count; i++) {
        int action  = scored[i].move;
        int from_sq = action / 25;
        int mv      = action % 25;
        int to_sq   = (from_sq / 7 + (mv / 5) - 2) * 7 + (from_sq % 7 + (mv % 5) - 2);

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
            int hval = 1 << (depth < 19 ? depth : 19);
            history[from_sq][to_sq] += hval;
            bound = TT_LOWER_BOUND;
            break;
        }
    }

    tt_store(hash, depth, value, bound, best_move, ply);
    return value;
}

/*  Root search  */

static float root_negascout(uint64_t mover_bb, uint64_t opp_bb, int mover_color,
                             int depth, float alpha, float beta, uint64_t hash,
                             clock_t start, double deadline_ms,
                             int16_t *best_move_out) {
    int16_t moves[1225];
    int move_count = gen_moves(mover_bb, opp_bb, moves);
    if (move_count == 0) { *best_move_out = -1; return -(SCORE_INF - 1.0f); }

    float   dummy;
    int16_t tt_best = -1;
    tt_probe(hash, depth, -SCORE_INF, SCORE_INF, &dummy, &tt_best, 0);

    ScoredMove scored[1225];
    if (policy_prior_valid) {
        for (int i = 0; i < move_count; i++) {
            int   action = moves[i];
            float p      = policy_prior_buf[action];
            int16_t sc   = (action == (int)tt_best) ? 30000
                         : (int16_t)(p * 28000.0f < 29000.0f ? p * 28000.0f : 29000.0f);
            scored[i] = (ScoredMove){ (int16_t)action, sc };
        }
        qsort(scored, move_count, sizeof(ScoredMove), compare_scored_moves);
    } else {
        order_moves(moves, move_count, opp_bb, tt_best, scored);
    }

    float   value     = -SCORE_INF;
    int16_t best_move = scored[0].move;
    int     opp_color = 3 - mover_color;

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

    if (value < -WIN_THRESHOLD) {
        for (int i = 0; i < move_count; i++) {
            int mv = scored[i].move % 25;
            if (abs((mv % 5) - 2) <= 1 && abs((mv / 5) - 2) <= 1) {
                best_move = scored[i].move;
                break;
            }
        }
    }

    *best_move_out = best_move;
    return value;
}

/* ── Policy prior helpers ─────────────────────────────────────────────── */

static void build_obs_nhwc(uint64_t mover_bb, uint64_t opp_bb, float obs[196]) {
    memset(obs, 0, 196 * sizeof(float));
    for (int sq = 0; sq < 49; sq++) {
        int base    = sq * 4;
        obs[base+0] = (float)((opp_bb   >> sq) & 1);   /* ch0: opponent */
        obs[base+1] = (float)((mover_bb >> sq) & 1);   /* ch1: mover    */
        obs[base+2] = 1.0f;                             /* ch2: constant */
        /* obs[base+3] = 0.0f  (unused, zero from memset) */
    }
}

#ifdef BB_USE_ONNX
static void policy_prior_run(uint64_t mover_bb, uint64_t opp_bb) {
    policy_prior_valid = false;
    if (!ort_session || !policy_prior_enabled) return;

    float obs[196];
    build_obs_nhwc(mover_bb, opp_bb, obs);

    int64_t       shape[]  = {1, 7, 7, 4};
    OrtMemoryInfo *mem_info = NULL;
    OrtStatus     *s        = g_ort->CreateCpuMemoryInfo(
        OrtArenaAllocator, OrtMemTypeDefault, &mem_info);
    if (s) { g_ort->ReleaseStatus(s); return; }

    OrtValue *input_val = NULL;
    s = g_ort->CreateTensorWithDataAsOrtValue(
        mem_info, obs, sizeof(obs), shape, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_val);
    g_ort->ReleaseMemoryInfo(mem_info);
    if (s) { g_ort->ReleaseStatus(s); return; }

    const char *in_names[]  = {"obs"};
    const char *out_names[] = {"logits"};
    OrtValue   *out_val     = NULL;
    s = g_ort->Run(ort_session, NULL,
                   in_names,  (const OrtValue *const *)&input_val, 1,
                   out_names, 1, &out_val);
    g_ort->ReleaseValue(input_val);
    if (s) { g_ort->ReleaseStatus(s); return; }

    float *logits = NULL;
    g_ort->GetTensorMutableData(out_val, (void **)&logits);

    /* stable softmax into policy_prior_buf */
    float maxl = logits[0];
    for (int i = 1; i < 1225; i++) if (logits[i] > maxl) maxl = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < 1225; i++) { policy_prior_buf[i] = expf(logits[i] - maxl); sum += policy_prior_buf[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < 1225; i++) policy_prior_buf[i] *= inv;

    g_ort->ReleaseValue(out_val);
    policy_prior_valid = true;
}

int policy_prior_load(const char *onnx_path) {
    if (!g_ort) g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    if (!ort_env) {
        OrtStatus *s = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "bb_policy", &ort_env);
        if (s) { g_ort->ReleaseStatus(s); ort_env = NULL; return 0; }
    }

    if (ort_session) { g_ort->ReleaseSession(ort_session); ort_session = NULL; }

    OrtSessionOptions *opts = NULL;
    g_ort->CreateSessionOptions(&opts);
    g_ort->SetIntraOpNumThreads(opts, 1);
    g_ort->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL);

    OrtStatus *s;
#ifdef _WIN32
    wchar_t wpath[1024];
    mbstowcs(wpath, onnx_path, 1023);
    wpath[1023] = L'\0';
    s = g_ort->CreateSession(ort_env, wpath, opts, &ort_session);
#else
    s = g_ort->CreateSession(ort_env, onnx_path, opts, &ort_session);
#endif
    g_ort->ReleaseSessionOptions(opts);
    if (s) { g_ort->ReleaseStatus(s); ort_session = NULL; return 0; }

    /* warm-up: trigger JIT kernel compilation now so first game move is fast */
    policy_prior_enabled = true;
    policy_prior_run(0ULL, 0ULL);
    policy_prior_valid = false;

    return 1;
}

void policy_prior_unload(void) {
    policy_prior_enabled = false;
    policy_prior_valid   = false;
    if (ort_session) { g_ort->ReleaseSession(ort_session); ort_session = NULL; }
    if (ort_env)     { g_ort->ReleaseEnv(ort_env);         ort_env     = NULL; }
}

/* Cheap toggle — keeps session alive, just enables/disables ordering.
   Useful for benchmarking: load once, flip per player. */
void policy_prior_set_active(int active) {
    policy_prior_enabled = active && (ort_session != NULL);
    if (!policy_prior_enabled) policy_prior_valid = false;
}

#else  /* BB_USE_ONNX not defined — stub everything out */

static void policy_prior_run(uint64_t m, uint64_t o) { (void)m; (void)o; policy_prior_valid = false; }
int  policy_prior_load(const char *p)    { (void)p; return 0; }
void policy_prior_unload(void)           {}
void policy_prior_set_active(int active) { (void)active; }

#endif /* BB_USE_ONNX */

/*  Public API  */

int find_best_move(bool game_board[7][7][2], int depth, bool as_blue) {
    init_tt();
    BBState  s           = board_from_python(game_board);
    int      mover_color = as_blue ? BLUE_PLAYER  : GREEN_PLAYER;
    uint64_t mover_bb    = as_blue ? s.blue_bb  : s.green_bb;
    uint64_t opp_bb      = as_blue ? s.green_bb : s.blue_bb;
    uint64_t hash        = compute_hash(s.green_bb, s.blue_bb, as_blue);

    policy_prior_run(mover_bb, opp_bb);
    int16_t best_move = -1;
    root_negascout(mover_bb, opp_bb, mover_color, depth,
                   -SCORE_INF, SCORE_INF, hash, 0, -1.0, &best_move);
    return (int)best_move;
}

int find_best_move_timed(bool game_board[7][7][2], int max_ms, bool as_blue) {
    init_tt();
    BBState  s           = board_from_python(game_board);
    int      mover_color = as_blue ? BLUE_PLAYER  : GREEN_PLAYER;
    uint64_t mover_bb    = as_blue ? s.blue_bb  : s.green_bb;
    uint64_t opp_bb      = as_blue ? s.green_bb : s.blue_bb;
    uint64_t hash        = compute_hash(s.green_bb, s.blue_bb, as_blue);

    clock_t start      = clock();
    int16_t best_move  = -1;
    float   best_score = 0.0f;
    float   asp_delta  = 30.0f;

    policy_prior_run(mover_bb, opp_bb);   /* computed once; reused across all ID depths */

    for (int d = 1; d <= MAX_DEPTH; d++) {
        for (int i = 0; i < 49; i++)
            for (int j = 0; j < 49; j++)
                history[i][j] >>= 1;

        float lo = (d == 1) ? -SCORE_INF : best_score - asp_delta;
        float hi = (d == 1) ? +SCORE_INF : best_score + asp_delta;
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
                break;
            }
            double el = (double)(clock() - start) * 1000.0 / CLOCKS_PER_SEC;
            if (el >= (double)max_ms) return (int)best_move;
        }

        best_move = move_this_depth;
        asp_delta = 30.0f;

        double elapsed = (double)(clock() - start) * 1000.0 / CLOCKS_PER_SEC;
        if (elapsed >= (double)max_ms) break;

        if (best_score > WIN_THRESHOLD || best_score < -WIN_THRESHOLD) break;
    }

    return (int)best_move;
}

float minimax_score(bool game_board[7][7][2], int depth, bool as_blue) {
    init_tt();
    BBState  s           = board_from_python(game_board);
    int      mover_color = as_blue ? BLUE_PLAYER  : GREEN_PLAYER;
    uint64_t mover_bb    = as_blue ? s.blue_bb  : s.green_bb;
    uint64_t opp_bb      = as_blue ? s.green_bb : s.blue_bb;
    uint64_t hash        = compute_hash(s.green_bb, s.blue_bb, as_blue);

    float score = negamax(mover_bb, opp_bb, mover_color, depth,
                          -SCORE_INF, SCORE_INF, hash, 0);
    if (score >  WIN_THRESHOLD) return  1.0f;
    if (score < -WIN_THRESHOLD) return -1.0f;
    return tanhf(score / 100.0f);
}

void score_root_moves(bool game_board[7][7][2], int depth, bool as_blue,
                      float out_scores[1225]) {
    init_tt();
    BBState  s           = board_from_python(game_board);
    int      mover_color = as_blue ? BLUE_PLAYER  : GREEN_PLAYER;
    uint64_t mover_bb    = as_blue ? s.blue_bb  : s.green_bb;
    uint64_t opp_bb      = as_blue ? s.green_bb : s.blue_bb;
    uint64_t hash        = compute_hash(s.green_bb, s.blue_bb, as_blue);
    int      opp_color   = 3 - mover_color;

    for (int i = 0; i < 1225; i++) out_scores[i] = -2.0f;

    int16_t moves[1225];
    int move_count = gen_moves(mover_bb, opp_bb, moves);

    for (int idx = 0; idx < move_count; idx++) {
        int        action = moves[idx];
        MoveResult child  = make_move_bb(mover_bb, opp_bb, hash, action, mover_color);

        float score = -negamax(child.opp_bb, child.player_bb, opp_color,
                               depth - 1, -SCORE_INF, SCORE_INF, child.hash, 1);
        if (score >  WIN_THRESHOLD) { out_scores[action] =  1.0f; continue; }
        if (score < -WIN_THRESHOLD) { out_scores[action] = -1.0f; continue; }
        out_scores[action] = tanhf(score / 100.0f);
    }
}

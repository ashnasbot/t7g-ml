/*
 * Minimax solver for the T7G 'Microscope' puzzle (Ataxx variant, 7x7).
 *
 * Internal board: uint8_t[7][7], cells are EMPTY=0, GREEN=1, BLUE=2.
 * Public API (find_best_move) accepts bool[7][7][2] from Python and converts
 * at the boundary — 49 bytes, negligible overhead.
 *
 * Features:
 *   - Alpha-beta minimax with transposition table (Zobrist hashing)
 *   - Move ordering: TT best-move first, then by capture count
 *   - Separate TT namespaces per colour perspective (via BLUE_COLOUR_KEY XOR)
 *   - ±SCORE_INF sentinels for proven wins/losses; finite so -ffast-math is safe
 */
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define BOARD_SIZE 7

#define EMPTY 0
#define GREEN 1
#define BLUE  2

// Transposition table
#define TT_SIZE (1 << 20)   // 1M entries (~32 MB)
#define TT_MASK (TT_SIZE - 1)
#define SCORE_INF 1000.0f   // Sentinel: scores are bounded to ±49 (piece counts)
#define BLUE_COLOUR_KEY 0x9e3779b97f4a7c15ULL  // XOR'd into hash to separate colour perspectives

#define TT_EXACT       0
#define TT_LOWER_BOUND 1
#define TT_UPPER_BOUND 2

typedef struct {
    uint64_t hash;
    float    score;
    int8_t   depth;
    int8_t   bound;      // TT_EXACT, TT_LOWER_BOUND, or TT_UPPER_BOUND
    int16_t  best_move;
} TTEntry;

static TTEntry *tt_table = NULL;

// Zobrist tables — indexed [position][cell_value]; cell_value matches EMPTY/GREEN/BLUE
static uint64_t zobrist_table[49][3];
static bool     zobrist_initialized = false;

// ---------------------------------------------------------------------------
// Zobrist / TT lifecycle
// ---------------------------------------------------------------------------

/* Simple xorshift PRNG used once at startup to fill the Zobrist table. */
static uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *state = x;
    return x;
}

/* Fill Zobrist tables with random 64-bit values. Called once at first init. */
static void init_zobrist(void) {
    if (zobrist_initialized) return;
    uint64_t seed = 0x123456789ABCDEF0ULL;
    for (int i = 0; i < 49; i++)
        for (int j = 0; j < 3; j++)
            zobrist_table[i][j] = xorshift64(&seed);
    zobrist_initialized = true;
}

/* Allocate the TT and Zobrist tables on first use. No-op on subsequent calls. */
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

/* XOR-hash of occupied cells; EMPTY cells contribute 0 (skipped). */
uint64_t compute_zobrist_hash(uint8_t board[7][7]) {
    uint64_t hash = 0;
    for (int pos = 0; pos < 49; pos++) {
        uint8_t cell = board[pos / 7][pos % 7];
        if (cell != EMPTY) hash ^= zobrist_table[pos][cell];
    }
    return hash;
}

/* Look up hash in TT. Returns true and sets *score if the entry is usable at
 * this depth and within the current alpha-beta window; also copies best_move. */
static bool tt_probe(uint64_t hash, int depth, float alpha, float beta,
                     float *score, int16_t *best_move) {
    TTEntry *entry = &tt_table[hash & TT_MASK];
    if (entry->hash != hash) return false;
    if (entry->depth < depth) return false;

    *best_move = entry->best_move;

    if      (entry->bound == TT_EXACT)       { *score = entry->score; return true; }
    else if (entry->bound == TT_LOWER_BOUND) { if (entry->score >= beta)  { *score = entry->score; return true; } }
    else if (entry->bound == TT_UPPER_BOUND) { if (entry->score <= alpha) { *score = entry->score; return true; } }
    return false;
}

/* Write or replace a TT entry. Replaces only if same hash (update) or the new
 * entry has greater or equal depth (deeper results are more valuable). */
static void tt_store(uint64_t hash, int depth, float score,
                     int8_t bound, int16_t best_move) {
    TTEntry *entry = &tt_table[hash & TT_MASK];
    if (entry->hash != hash || entry->depth <= depth) {
        entry->hash = hash; entry->score = score;
        entry->depth = depth; entry->bound = bound;
        entry->best_move = best_move;
    }
}

// ---------------------------------------------------------------------------
// Board evaluation
// ---------------------------------------------------------------------------

/* Count cells belonging to player. */
static int count_cells(uint8_t board[7][7], uint8_t player) {
    int count = 0;
    for (int y = 0; y < BOARD_SIZE; y++)
        for (int x = 0; x < BOARD_SIZE; x++)
            if (board[y][x] == player) count++;
    return count;
}

/* Heuristic score from player's perspective: material (×10) plus centrality
 * bonus (Manhattan distance from centre, scaled ×0.1). */
static float get_score(uint8_t board[7][7], uint8_t player) {
    uint8_t opponent = (player == BLUE) ? GREEN : BLUE;
    float score = (count_cells(board, player) - count_cells(board, opponent)) * 10.0f;
    for (int y = 0; y < BOARD_SIZE; y++)
        for (int x = 0; x < BOARD_SIZE; x++) {
            float bonus = (6.0f - (float)(abs(x - 3) + abs(y - 3))) * 0.1f;
            if      (board[y][x] == player)   score += bonus;
            else if (board[y][x] == opponent) score -= bonus;
        }
    return score;
}

/* Count how many opponent pieces adjacent to the destination of action would
 * be converted. Used for move ordering. */
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

// ---------------------------------------------------------------------------
// Move generation
// ---------------------------------------------------------------------------

/* Use memchr (SIMD-optimised in libc) to check for any set move flag. */
static bool any_moves(bool valid_moves[7][7][5][5]) {
    return memchr(valid_moves, 1, sizeof(bool[7][7][5][5])) != NULL;
}

/* Fill valid_moves[7][7][5][5] for player. Flat index matches action encoding:
 * action = piece*25 + move, where piece = y*7+x and move = v*5+u (offsets 0-4,
 * representing delta -2..+2). Offset ranges are pre-clamped to board edges. */
static void get_valid_moves(uint8_t board[7][7], uint8_t player,
                            bool valid_moves[7][7][5][5]) {
    memset(valid_moves, 0, sizeof(bool[7][7][5][5]));
    bool *moves = (bool *)valid_moves;

    for (int y = 0; y < BOARD_SIZE; y++) {
        for (int x = 0; x < BOARD_SIZE; x++) {
            if (board[y][x] != player) continue;
            // Pre-clamp: to_x = x+u-2 must be in [0,6], so u in [max(0,2-x), min(5,9-x))
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

/* Apply action in-place. Clones (distance 1) copy the piece; jumps (distance 2)
 * move it. All opponent pieces adjacent to the destination are converted. */
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

typedef struct { int16_t move; int16_t score; } ScoredMove;

static int compare_scored_moves(const void *a, const void *b) {
    return ((ScoredMove*)b)->score - ((ScoredMove*)a)->score;
}

/* Collect valid moves, score each (TT move gets highest priority; others scored
 * by capture count), sort descending. Returns move count. */
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
// Alpha-beta minimax with transposition table
// ---------------------------------------------------------------------------

/* Recursive alpha-beta search. score_player is fixed throughout a search tree
 * and determines the sign convention: positive = good for score_player.
 * max_player flips each ply to indicate whose turn it is.
 *
 * The hash is XOR'd with BLUE_COLOUR_KEY when score_player == BLUE to give
 * each colour perspective its own TT namespace (scores have opposite signs for
 * the same position depending on perspective). */
static float minimax_cached(uint8_t board[7][7], int depth,
                            float alpha, float beta,
                            bool max_player, uint8_t score_player,
                            uint64_t hash) {
    uint64_t keyed_hash = hash ^ (score_player == BLUE ? BLUE_COLOUR_KEY : 0ULL);

    float tt_score; int16_t tt_best_move = -1;
    if (tt_probe(keyed_hash, depth, alpha, beta, &tt_score, &tt_best_move))
        return tt_score;

    uint8_t colour   = max_player ? score_player : (score_player == BLUE ? GREEN : BLUE);
    uint8_t opponent = (score_player == BLUE) ? GREEN : BLUE;
    float   bias     = max_player ? 0.5f : -0.5f;

    // Proven terminal: one side eliminated. ±SCORE_INF prunes entire subtrees.
    int sc_cells  = count_cells(board, score_player);
    int opp_cells = count_cells(board, opponent);
    if (sc_cells  == 0) { tt_store(keyed_hash, depth, -SCORE_INF, TT_EXACT, -1); return -SCORE_INF; }
    if (opp_cells == 0) { tt_store(keyed_hash, depth,  SCORE_INF, TT_EXACT, -1); return  SCORE_INF; }

    float score = get_score(board, score_player);
    if (depth == 0) {
        score += bias;
        tt_store(keyed_hash, depth, score, TT_EXACT, -1);
        return score;
    }

    bool valid_moves[7][7][5][5];
    get_valid_moves(board, colour, valid_moves);
    if (!any_moves(valid_moves)) {
        // Forced pass: current player has no moves. If opponent also has none, game over.
        bool opp_moves[7][7][5][5];
        get_valid_moves(board, colour == BLUE ? GREEN : BLUE, opp_moves);
        if (!any_moves(opp_moves)) {
            score += bias;
            tt_store(keyed_hash, depth, score, TT_EXACT, -1);
            return score;
        }
        // Decrement depth to prevent pass chains from inflating tree size.
        // XOR 1 into hash so same board with different mover gets a distinct TT slot.
        return minimax_cached(board, depth - 1, alpha, beta,
                              !max_player, score_player, hash ^ 1ULL);
    }

    ScoredMove scored_moves[1225];
    int move_count = order_moves(board, (bool *)valid_moves, colour, tt_best_move, scored_moves);

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
                                    !max_player, score_player, child_hash);

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

    tt_store(keyed_hash, depth, value, bound, best_move);
    return value;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/* Find the best move for player (as_blue selects Blue or Green).
 * Accepts the Python bool[7][7][2] board layout (ch0=green, ch1=blue) and
 * converts to uint8_t internally. Returns the action index (0-1224), or -1
 * if the player has no legal moves. The TT persists across calls in the same
 * process, so earlier searches benefit later ones. */
int find_best_move(bool game_board[7][7][2], int depth, bool as_blue) {
    init_tt();

    uint8_t board[7][7];
    for (int y = 0; y < 7; y++)
        for (int x = 0; x < 7; x++)
            board[y][x] = game_board[y][x][1] ? BLUE
                        : game_board[y][x][0]  ? GREEN
                        :                        EMPTY;

    uint8_t player = as_blue ? BLUE : GREEN;

    bool valid_moves[7][7][5][5];
    get_valid_moves(board, player, valid_moves);

    ScoredMove scored_moves[1225];
    int move_count = order_moves(board, (bool *)valid_moves, player, -1, scored_moves);

    // Initialise below -SCORE_INF so even a proven-loss move updates best.
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
                                        false, player, hash);

        if (score >= SCORE_INF) return i;  // Proven win — take it immediately
        if (score > best_score) { best_score = score; residx = i; }
    }

    // In a proven-loss (all moves scored -SCORE_INF), prefer a clone over a jump.
    // A clone keeps piece count up and may hasten the end; a jump just relocates.
    // scored_moves is already sorted by capture count, so the first clone found
    // is also the best-capturing one.
    if (best_score <= -SCORE_INF && residx >= 0) {
        for (int idx = 0; idx < move_count; idx++) {
            int i = scored_moves[idx].move;
            int mv = i % 25;
            int mv_x = (mv % 5) - 2, mv_y = (mv / 5) - 2;
            if (abs(mv_x) <= 1 && abs(mv_y) <= 1) { residx = i; break; }
        }
    }

    return residx;
}

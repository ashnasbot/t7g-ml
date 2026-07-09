/*
 * tests/test_micro_mcts.c - Unit tests for the micro_mcts.c public API.
 *
 * Tests what the code SHOULD do; do not assume the implementation is correct.
 * Covers: terminal detection, board/turn conversion, value perspective,
 * policy invariants, TT isolation, cross-game pool pollution, bounds safety,
 * and make_move correctness (clone/jump/capture).
 *
 * Build: gcc -O0 -g -Wall micro_mcts.c tests/test_micro_mcts.c -o test_micro_mcts -lm
 * Run:   ./test_micro_mcts
 */
#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * Public API forward declarations (implemented in micro_mcts.c)
 * ------------------------------------------------------------------------- */
void  mcgs_init(void);
void *mcgs_create(int num_simulations, float c_puct, int gumbel_k);
void  mcgs_clear(void *inst);
void  mcgs_destroy(void *inst);
int   mcgs_tt_size(void *inst);
void *mcgs_start_search(void *inst, bool py_board[7][7][2], bool turn);
void  mcgs_search_destroy(void *ss);
int   mcgs_pending_count(void *ss);
int   mcgs_is_done(void *ss);
void  mcgs_get_leaf_board(void *ss, int i, bool out[7][7][2]);
bool  mcgs_get_leaf_turn(void *ss, int i);
void  mcgs_commit_expansion(void *ss, int i, float policy[1225], float value);
int   mcgs_step(void *ss);
void  mcgs_get_result(void *ss, float out[1225]);
float mcgs_get_root_value(void *ss);
void  mcgs_apply_root_dirichlet(void *ss, float alpha, float eps);
int   mcgs_get_pending_boards(void *ss, bool *boards_out, bool *turns_out);
void  mcgs_commit_batch(void *ss, float *policies_flat, float *values, int n);

/* -------------------------------------------------------------------------
 * Minimal test framework
 * ------------------------------------------------------------------------- */
static int  g_tests    = 0;
static int  g_failures = 0;
static const char *g_test_name = NULL;

#define TEST_BEGIN(name) do { g_test_name = (name); g_tests++; } while(0)
#define PASS() do { printf("  PASS  %s\n", g_test_name); return; } while(0)

/* FAIL: print message and return early from the test function */
#define FAIL(msg) do { \
    g_failures++; \
    printf("  FAIL  %s  [line %d]: %s\n", g_test_name, __LINE__, (msg)); \
    return; \
} while(0)

#define CHECK(cond, msg)  do { if (!(cond)) FAIL(msg); } while(0)

#define CHECK_EQ_F(a, b, tol, msg) do { \
    float _a = (float)(a), _b = (float)(b); \
    if (fabsf(_a - _b) > (tol)) { \
        char _buf[256]; \
        snprintf(_buf, sizeof(_buf), "%s: got %.8f expected %.8f", (msg), _a, _b); \
        FAIL(_buf); \
    } \
} while(0)

/* -------------------------------------------------------------------------
 * Board helpers
 * Board format: bool[7][7][2]  -- [y][x][channel]
 *   channel 0 = GREEN piece
 *   channel 1 = BLUE piece
 * Both channels false = EMPTY.
 * ------------------------------------------------------------------------- */
static void board_clear(bool b[7][7][2]) {
    memset(b, 0, sizeof(bool[7][7][2]));
}
static void place_blue(bool b[7][7][2], int x, int y)  { b[y][x][1] = true; }
static void place_green(bool b[7][7][2], int x, int y) { b[y][x][0] = true; }

/* Standard starting board: Blue at (0,0),(6,6)  Green at (6,0),(0,6) */
static void board_start(bool b[7][7][2]) {
    board_clear(b);
    place_blue(b,  0, 0); place_blue(b,  6, 6);
    place_green(b, 6, 0); place_green(b, 0, 6);
}

/* Uniform policy: equal weight over all 1225 actions.
 * mcgs_commit_expansion normalises over legal moves, so any positive
 * distribution works. */
static void uniform_policy(float p[1225]) {
    for (int i = 0; i < 1225; i++) p[i] = 1.0f / 1225.0f;
}

/* Drive a search to completion feeding every pending leaf with the given
 * constant value and a uniform policy.  Returns number of step() calls. */
static int run_full_search(void *ss, float leaf_value) {
    float policy[1225];
    uniform_policy(policy);
    int steps = 0;
    while (!mcgs_is_done(ss)) {
        int n = mcgs_pending_count(ss);
        for (int i = 0; i < n; i++)
            mcgs_commit_expansion(ss, i, policy, leaf_value);
        mcgs_step(ss);
        if (++steps > 20000) break;   /* safety valve against infinite loops */
    }
    return steps;
}

/* =========================================================================
 * Lifecycle
 * ========================================================================= */

static void test_create_destroy(void) {
    TEST_BEGIN("create_destroy");
    void *inst = mcgs_create(16, 1.0f, 8);
    CHECK(inst != NULL, "mcgs_create returned NULL");
    mcgs_destroy(inst);
    PASS();
}

static void test_clear_and_reuse(void) {
    TEST_BEGIN("clear_and_reuse");
    void *inst = mcgs_create(16, 1.0f, 8);
    CHECK(inst != NULL, "mcgs_create returned NULL");
    bool board[7][7][2];
    board_start(board);

    void *ss = mcgs_start_search(inst, board, true);
    CHECK(ss != NULL, "first start_search returned NULL");
    run_full_search(ss, 0.0f);
    mcgs_search_destroy(ss);
    mcgs_clear(inst);

    ss = mcgs_start_search(inst, board, true);
    CHECK(ss != NULL, "start_search after clear returned NULL");
    run_full_search(ss, 0.0f);
    mcgs_search_destroy(ss);
    mcgs_destroy(inst);
    PASS();
}

/* =========================================================================
 * Transposition table size
 * ========================================================================= */

static void test_tt_size_zero_after_create(void) {
    TEST_BEGIN("tt_size_zero_after_create");
    void *inst = mcgs_create(16, 1.0f, 8);
    CHECK(inst != NULL, "mcgs_create returned NULL");
    CHECK(mcgs_tt_size(inst) == 0, "fresh instance must have 0 TT nodes");
    mcgs_destroy(inst);
    PASS();
}

static void test_tt_size_grows_during_search(void) {
    TEST_BEGIN("tt_size_grows_during_search");
    void *inst = mcgs_create(16, 1.0f, 8);
    bool board[7][7][2];
    board_start(board);
    void *ss = mcgs_start_search(inst, board, true);
    CHECK(ss != NULL, "start_search returned NULL");
    run_full_search(ss, 0.0f);
    mcgs_search_destroy(ss);
    CHECK(mcgs_tt_size(inst) > 0, "TT must contain nodes after a search");
    mcgs_destroy(inst);
    PASS();
}

static void test_tt_size_zero_after_clear(void) {
    TEST_BEGIN("tt_size_zero_after_clear");
    void *inst = mcgs_create(16, 1.0f, 8);
    bool board[7][7][2];
    board_start(board);
    void *ss = mcgs_start_search(inst, board, true);
    run_full_search(ss, 0.0f);
    mcgs_search_destroy(ss);
    CHECK(mcgs_tt_size(inst) > 0, "pre-clear: TT should have nodes");
    mcgs_clear(inst);
    CHECK(mcgs_tt_size(inst) == 0,
          "TT must be exactly 0 after clear -- stale nodes cause cross-game Q corruption");
    mcgs_destroy(inst);
    PASS();
}

static void test_tt_nondecreasing_during_search(void) {
    TEST_BEGIN("tt_nondecreasing_during_search");
    void *inst = mcgs_create(32, 1.0f, 8);
    bool board[7][7][2];
    board_start(board);
    float policy[1225];
    uniform_policy(policy);

    void *ss = mcgs_start_search(inst, board, true);
    CHECK(ss != NULL, "start_search returned NULL");
    int prev = mcgs_tt_size(inst);
    int steps = 0;
    while (!mcgs_is_done(ss) && steps++ < 1000) {
        int n = mcgs_pending_count(ss);
        for (int i = 0; i < n; i++)
            mcgs_commit_expansion(ss, i, policy, 0.0f);
        mcgs_step(ss);
        int curr = mcgs_tt_size(inst);
        if (curr < prev) {
            char msg[128];
            snprintf(msg, sizeof(msg),
                     "TT shrunk: %d -> %d at step %d (nodes must not vanish)", prev, curr, steps);
            mcgs_search_destroy(ss);
            mcgs_destroy(inst);
            FAIL(msg);
        }
        prev = curr;
    }
    mcgs_search_destroy(ss);
    mcgs_destroy(inst);
    PASS();
}

/* =========================================================================
 * Board / turn roundtrip through the pending-leaf API
 * ========================================================================= */

static void test_root_leaf_board_roundtrip(void) {
    TEST_BEGIN("root_leaf_board_roundtrip");
    void *inst = mcgs_create(16, 1.0f, 8);
    /* Non-symmetric board to expose any channel or axis swaps */
    bool board_in[7][7][2];
    board_clear(board_in);
    place_blue(board_in,  1, 2);
    place_blue(board_in,  5, 6);
    place_green(board_in, 3, 0);
    place_green(board_in, 6, 4);

    void *ss = mcgs_start_search(inst, board_in, true);
    CHECK(ss != NULL, "start_search returned NULL");
    CHECK(mcgs_pending_count(ss) == 1, "root must be sole pending leaf initially");

    bool board_out[7][7][2];
    memset(board_out, 0xFF, sizeof(board_out));   /* poison - detect partial writes */
    mcgs_get_leaf_board(ss, 0, board_out);

    for (int y = 0; y < 7; y++)
        for (int x = 0; x < 7; x++)
            for (int ch = 0; ch < 2; ch++) {
                if (board_in[y][x][ch] != board_out[y][x][ch]) {
                    char msg[80];
                    snprintf(msg, sizeof(msg),
                             "board mismatch at [y=%d][x=%d][ch=%d]: in=%d out=%d",
                             y, x, ch, board_in[y][x][ch], board_out[y][x][ch]);
                    mcgs_search_destroy(ss);
                    mcgs_destroy(inst);
                    FAIL(msg);
                }
            }

    mcgs_search_destroy(ss);
    mcgs_destroy(inst);
    PASS();
}

static void test_root_leaf_turn_blue(void) {
    TEST_BEGIN("root_leaf_turn_blue");
    void *inst = mcgs_create(16, 1.0f, 8);
    bool board[7][7][2];
    board_start(board);
    void *ss = mcgs_start_search(inst, board, true);
    CHECK(ss != NULL, "start_search returned NULL");
    CHECK(mcgs_pending_count(ss) >= 1, "must have at least one pending leaf");
    bool turn = mcgs_get_leaf_turn(ss, 0);
    mcgs_search_destroy(ss);
    mcgs_destroy(inst);
    CHECK(turn == true, "leaf turn must be true (Blue) when started with turn=true");
    PASS();
}

static void test_root_leaf_turn_green(void) {
    TEST_BEGIN("root_leaf_turn_green");
    void *inst = mcgs_create(16, 1.0f, 8);
    bool board[7][7][2];
    board_start(board);
    void *ss = mcgs_start_search(inst, board, false);
    CHECK(ss != NULL, "start_search returned NULL");
    CHECK(mcgs_pending_count(ss) >= 1, "must have at least one pending leaf");
    bool turn = mcgs_get_leaf_turn(ss, 0);
    mcgs_search_destroy(ss);
    mcgs_destroy(inst);
    CHECK(turn == false, "leaf turn must be false (Green) when started with turn=false");
    PASS();
}

/* =========================================================================
 * Terminal detection
 * ========================================================================= */

static void test_terminal_blue_eliminated(void) {
    TEST_BEGIN("terminal_blue_eliminated");
    void *inst = mcgs_create(16, 1.0f, 8);
    bool board[7][7][2];
    board_clear(board);
    place_green(board, 3, 3);   /* only Green, no Blue */
    void *ss = mcgs_start_search(inst, board, true);   /* Blue's turn */
    CHECK(ss != NULL, "start_search returned NULL");
    CHECK(mcgs_is_done(ss) == 1,
          "Blue eliminated -> terminal, must be immediately done");
    float result[1225];
    mcgs_get_result(ss, result);
    float sum = 0.0f;
    for (int i = 0; i < 1225; i++) sum += fabsf(result[i]);
    CHECK(sum == 0.0f, "terminal position must return all-zero policy");
    mcgs_search_destroy(ss);
    mcgs_destroy(inst);
    PASS();
}

static void test_terminal_green_eliminated(void) {
    TEST_BEGIN("terminal_green_eliminated");
    void *inst = mcgs_create(16, 1.0f, 8);
    bool board[7][7][2];
    board_clear(board);
    place_blue(board, 3, 3);    /* only Blue, no Green */
    void *ss = mcgs_start_search(inst, board, false);   /* Green's turn */
    CHECK(ss != NULL, "start_search returned NULL");
    CHECK(mcgs_is_done(ss) == 1,
          "Green eliminated -> terminal, must be immediately done");
    mcgs_search_destroy(ss);
    mcgs_destroy(inst);
    PASS();
}

/*
 * When ONLY the current mover is stuck but the opponent has moves, the rules
 * require a forced pass -- this is NOT terminal.  The search must NOT return
 * is_done immediately and must eventually complete.
 *
 * Board: Green at (2,2), surrounded by Blue in its entire 5x5 range.
 * Blue has extra pieces at (5,5) with room to move.
 * Green's turn: Green has zero legal moves. Blue has legal moves.
 * => forced pass, not terminal.
 */
static void test_forced_pass_not_terminal(void) {
    TEST_BEGIN("forced_pass_not_terminal");
    void *inst = mcgs_create(8, 1.0f, 4);
    bool board[7][7][2];
    board_clear(board);
    place_green(board, 2, 2);
    /* Fill Green's entire 5x5 range with Blue (all squares within distance 2) */
    for (int y = 0; y <= 4; y++)
        for (int x = 0; x <= 4; x++)
            if (!(x == 2 && y == 2))
                place_blue(board, x, y);
    /* Extra Blue with room to move */
    place_blue(board, 5, 5);

    /* Green to move: Green is completely stuck, Blue has moves */
    void *ss = mcgs_start_search(inst, board, false);
    CHECK(ss != NULL, "start_search returned NULL");
    CHECK(mcgs_is_done(ss) == 0,
          "stuck-but-not-terminal position must NOT be immediately done");
    run_full_search(ss, 0.0f);
    CHECK(mcgs_is_done(ss) == 1, "search must eventually complete");
    mcgs_search_destroy(ss);
    mcgs_destroy(inst);
    PASS();
}

/*
 * Both players stuck AND Blue has more pieces -> terminal, Blue wins.
 * Value perspective: from mover's POV.
 *   Blue to move + Blue wins => value should be reflected as root Q > 0 if Blue
 *   has winning terminal children (indirectly tested via value_perspective tests).
 */
static void test_both_stuck_terminal(void) {
    TEST_BEGIN("both_stuck_terminal");
    void *inst = mcgs_create(8, 1.0f, 4);
    /* Full board: all 49 cells occupied, no legal moves for anyone */
    bool board[7][7][2];
    board_clear(board);
    /* Blue: rows 0-3 (28 cells), Green: rows 4-6 (21 cells) => Blue wins */
    for (int y = 0; y < 7; y++)
        for (int x = 0; x < 7; x++) {
            if (y < 4) place_blue(board, x, y);
            else       place_green(board, x, y);
        }
    /* Both have no moves since the board is full */
    void *ss = mcgs_start_search(inst, board, true);   /* Blue to move */
    CHECK(ss != NULL, "start_search returned NULL");
    CHECK(mcgs_is_done(ss) == 1, "full board with count advantage must be terminal");
    mcgs_search_destroy(ss);
    mcgs_destroy(inst);
    PASS();
}

/* =========================================================================
 * Value perspective
 *
 * Convention: terminal_value and network_value are from the CURRENT MOVER's
 * perspective.  mcgs_get_root_value returns root Q from the root mover's
 * perspective.
 *
 * Test: position one move away from Blue winning.
 * Blue at (3,3), Green at (3,4).  Several Blue clones land adjacent to (3,4)
 * and capture Green -> terminal, value=+1 for Blue.
 * With network_value=0 for non-terminal leaves, root Q must be > 0.
 * ========================================================================= */

static void test_value_perspective_blue_winning(void) {
    TEST_BEGIN("value_perspective_blue_winning");
    void *inst = mcgs_create(64, 1.0f, 16);
    bool board[7][7][2];
    board_clear(board);
    place_blue(board,  3, 3);
    place_green(board, 3, 4);   /* one Green, adjacent to Blue */

    void *ss = mcgs_start_search(inst, board, true);   /* Blue to move */
    CHECK(ss != NULL, "start_search returned NULL");
    run_full_search(ss, 0.0f);

    float root_q = mcgs_get_root_value(ss);
    mcgs_search_destroy(ss);
    mcgs_destroy(inst);
    CHECK(root_q > 0.0f,
          "root Q must be > 0 when Blue can immediately win "
          "(value perspective or capture mechanics broken)");
    PASS();
}

static void test_value_perspective_green_winning(void) {
    TEST_BEGIN("value_perspective_green_winning");
    void *inst = mcgs_create(64, 1.0f, 16);
    bool board[7][7][2];
    board_clear(board);
    place_green(board, 3, 3);
    place_blue(board,  3, 4);   /* one Blue, adjacent to Green */

    void *ss = mcgs_start_search(inst, board, false);  /* Green to move */
    CHECK(ss != NULL, "start_search returned NULL");
    run_full_search(ss, 0.0f);

    float root_q = mcgs_get_root_value(ss);
    mcgs_search_destroy(ss);
    mcgs_destroy(inst);
    CHECK(root_q > 0.0f,
          "root Q must be > 0 when Green can immediately win "
          "(value perspective wrong for Green)");
    PASS();
}

/* =========================================================================
 * Policy invariants
 * ========================================================================= */

static void test_policy_sums_to_one(void) {
    TEST_BEGIN("policy_sums_to_one");
    void *inst = mcgs_create(16, 1.0f, 8);
    bool board[7][7][2];
    board_start(board);
    void *ss = mcgs_start_search(inst, board, true);
    CHECK(ss != NULL, "start_search returned NULL");
    run_full_search(ss, 0.0f);
    float result[1225];
    mcgs_get_result(ss, result);
    float sum = 0.0f;
    for (int i = 0; i < 1225; i++) sum += result[i];
    mcgs_search_destroy(ss);
    mcgs_destroy(inst);
    CHECK_EQ_F(sum, 1.0f, 1e-4f, "policy must sum to 1.0");
    PASS();
}

static void test_policy_nonnegative(void) {
    TEST_BEGIN("policy_nonnegative");
    void *inst = mcgs_create(16, 1.0f, 8);
    bool board[7][7][2];
    board_start(board);
    void *ss = mcgs_start_search(inst, board, true);
    run_full_search(ss, 0.0f);
    float result[1225];
    mcgs_get_result(ss, result);
    for (int i = 0; i < 1225; i++) {
        if (result[i] < 0.0f) {
            char msg[64];
            snprintf(msg, sizeof(msg), "result[%d] = %.8f < 0", i, result[i]);
            mcgs_search_destroy(ss);
            mcgs_destroy(inst);
            FAIL(msg);
        }
    }
    mcgs_search_destroy(ss);
    mcgs_destroy(inst);
    PASS();
}

/*
 * Actions whose piece belongs to the opponent must receive exactly 0 probability.
 * Initial board: Blue at (0,0),(6,6). Green at (6,0),(0,6).
 * piece index for Green at (0,6): y=6,x=0 -> piece=42
 * piece index for Green at (6,0): y=0,x=6 -> piece=6
 * All 25 actions for each Green piece are illegal when Blue is moving.
 */
static void test_policy_zero_on_opponent_pieces(void) {
    TEST_BEGIN("policy_zero_on_opponent_pieces");
    void *inst = mcgs_create(16, 1.0f, 8);
    bool board[7][7][2];
    board_start(board);
    void *ss = mcgs_start_search(inst, board, true);   /* Blue to move */
    run_full_search(ss, 0.0f);
    float result[1225];
    mcgs_get_result(ss, result);

    int green_pieces[2] = {42, 6};  /* piece indices for Green's starting squares */
    for (int p = 0; p < 2; p++) {
        for (int mv = 0; mv < 25; mv++) {
            int action = green_pieces[p] * 25 + mv;
            if (result[action] != 0.0f) {
                char msg[80];
                snprintf(msg, sizeof(msg),
                         "result[%d] = %.8f for Green-piece action (must be 0)",
                         action, result[action]);
                mcgs_search_destroy(ss);
                mcgs_destroy(inst);
                FAIL(msg);
            }
        }
    }
    mcgs_search_destroy(ss);
    mcgs_destroy(inst);
    PASS();
}

/*
 * At least one legal move must receive non-zero probability.
 */
static void test_policy_nonzero_on_legal_moves(void) {
    TEST_BEGIN("policy_nonzero_on_legal_moves");
    void *inst = mcgs_create(16, 1.0f, 8);
    bool board[7][7][2];
    board_start(board);
    void *ss = mcgs_start_search(inst, board, true);
    run_full_search(ss, 0.0f);
    float result[1225];
    mcgs_get_result(ss, result);
    float sum = 0.0f;
    for (int i = 0; i < 1225; i++) sum += result[i];
    mcgs_search_destroy(ss);
    mcgs_destroy(inst);
    CHECK(sum > 0.0f, "at least one legal move must have positive probability");
    PASS();
}

/* =========================================================================
 * Cross-game isolation (the heatmap / pool pollution bug)
 * ========================================================================= */

static void test_tt_fully_cleared_between_games(void) {
    TEST_BEGIN("tt_fully_cleared_between_games");
    void *inst = mcgs_create(32, 1.0f, 8);
    bool board[7][7][2];
    board_start(board);

    /* Game A */
    void *ss = mcgs_start_search(inst, board, true);
    CHECK(ss != NULL, "game A: start_search returned NULL");
    run_full_search(ss, 0.5f);
    mcgs_search_destroy(ss);
    int tt_a = mcgs_tt_size(inst);
    CHECK(tt_a > 0, "game A must populate the TT");

    mcgs_clear(inst);
    CHECK(mcgs_tt_size(inst) == 0,
          "TT must be completely 0 after clear -- any retained node "
          "from game A would corrupt game B's Q estimates");

    /* Game B: first search must see only fresh nodes */
    ss = mcgs_start_search(inst, board, true);
    CHECK(ss != NULL, "game B: start_search returned NULL");
    run_full_search(ss, 0.5f);
    mcgs_search_destroy(ss);
    int tt_b = mcgs_tt_size(inst);

    /* If clear didn't work, tt_b would exceed tt_a (game A nodes still present).
     * After ONE fresh search, tt_b must be <= tt_a. */
    CHECK(tt_b > 0, "game B search must create nodes");
    CHECK(tt_b <= tt_a,
          "game B TT must be <= game A TT: if larger, clear() left stale nodes");

    mcgs_destroy(inst);
    PASS();
}

static void test_repeated_cycles_stay_bounded(void) {
    TEST_BEGIN("repeated_cycles_stay_bounded");
    void *inst = mcgs_create(16, 1.0f, 8);
    bool board[7][7][2];
    board_start(board);
    int sizes[6];

    for (int g = 0; g < 6; g++) {
        mcgs_clear(inst);
        CHECK(mcgs_tt_size(inst) == 0, "TT must be 0 at start of each game");
        void *ss = mcgs_start_search(inst, board, true);
        CHECK(ss != NULL, "start_search returned NULL");
        run_full_search(ss, 0.0f);
        mcgs_search_destroy(ss);
        sizes[g] = mcgs_tt_size(inst);
    }

    /* A working clear means each game starts fresh.  TT sizes should be
     * roughly stable.  A monotonically growing sequence = clear is broken. */
    int min_s = sizes[0], max_s = sizes[0];
    for (int i = 1; i < 6; i++) {
        if (sizes[i] < min_s) min_s = sizes[i];
        if (sizes[i] > max_s) max_s = sizes[i];
    }
    if (max_s > min_s * 5) {
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "TT sizes growing across game cycles (min=%d max=%d) -- clear() broken",
                 min_s, max_s);
        mcgs_destroy(inst);
        FAIL(msg);
    }
    mcgs_destroy(inst);
    PASS();
}

/*
 * After clear, a fresh search for the SAME position must start from visit_count=0.
 * mcgs_get_root_value returns 0.0 when visit_count==0.
 * If the TT node was not truly reset, it would retain its old value_sum/visit_count
 * and return a non-zero root value before any simulation.
 */
static void test_fresh_root_after_clear(void) {
    TEST_BEGIN("fresh_root_after_clear");
    void *inst = mcgs_create(16, 1.0f, 8);
    bool board[7][7][2];
    board_start(board);

    void *ss = mcgs_start_search(inst, board, true);
    run_full_search(ss, 0.8f);   /* bias the value */
    mcgs_search_destroy(ss);

    mcgs_clear(inst);

    /* Start a new search but DON'T run it -- inspect root before any visits */
    ss = mcgs_start_search(inst, board, true);
    CHECK(ss != NULL, "start_search after clear returned NULL");
    float rv = mcgs_get_root_value(ss);
    mcgs_search_destroy(ss);
    mcgs_destroy(inst);

    CHECK(rv == 0.0f,
          "root value must be 0 before any visits -- stale TT node would give non-zero");
    PASS();
}

/* =========================================================================
 * Two-instance isolation (pool slots must not share TT state)
 * ========================================================================= */

static void test_two_instances_independent(void) {
    TEST_BEGIN("two_instances_independent");
    void *inst_a = mcgs_create(16, 1.0f, 8);
    void *inst_b = mcgs_create(16, 1.0f, 8);
    bool board[7][7][2];
    board_start(board);

    CHECK(mcgs_tt_size(inst_a) == 0, "inst_a must start empty");
    CHECK(mcgs_tt_size(inst_b) == 0, "inst_b must start empty");

    void *ss = mcgs_start_search(inst_a, board, true);
    run_full_search(ss, 0.0f);
    mcgs_search_destroy(ss);

    CHECK(mcgs_tt_size(inst_a) > 0, "inst_a must have nodes after search");
    CHECK(mcgs_tt_size(inst_b) == 0,
          "inst_b TT must still be 0 -- instances must not share TT state");

    mcgs_destroy(inst_a);
    mcgs_destroy(inst_b);
    PASS();
}

/* =========================================================================
 * Bounds and null-safety
 * ========================================================================= */

static void test_oob_leaf_index_no_crash(void) {
    TEST_BEGIN("oob_leaf_index_no_crash");
    void *inst = mcgs_create(16, 1.0f, 8);
    bool board[7][7][2];
    board_start(board);
    void *ss = mcgs_start_search(inst, board, true);
    CHECK(ss != NULL, "start_search returned NULL");

    bool out[7][7][2];
    float pol[1225]; uniform_policy(pol);

    /* All of these should silently no-op without crashing */
    mcgs_get_leaf_board(ss, -1, out);
    mcgs_get_leaf_board(ss, 999, out);
    mcgs_get_leaf_turn(ss, -1);
    mcgs_get_leaf_turn(ss, 999);
    mcgs_commit_expansion(ss, -1, pol, 0.0f);
    mcgs_commit_expansion(ss, 999, pol, 0.0f);

    mcgs_search_destroy(ss);
    mcgs_destroy(inst);
    PASS();
}

static void test_null_instance_no_crash(void) {
    TEST_BEGIN("null_instance_no_crash");
    mcgs_clear(NULL);
    mcgs_destroy(NULL);
    CHECK(mcgs_tt_size(NULL) == 0, "mcgs_tt_size(NULL) must return 0");
    PASS();
}

static void test_null_search_state_no_crash(void) {
    TEST_BEGIN("null_search_state_no_crash");
    CHECK(mcgs_pending_count(NULL) == 0, "pending_count(NULL) must return 0");
    CHECK(mcgs_is_done(NULL) == 1,       "is_done(NULL) must return 1 (done)");
    mcgs_search_destroy(NULL);
    float result[1225] = {0};
    mcgs_get_result(NULL, result);
    CHECK(mcgs_get_root_value(NULL) == 0.0f, "root_value(NULL) must return 0.0");
    PASS();
}

/* =========================================================================
 * Edge cases
 * ========================================================================= */

static void test_single_simulation(void) {
    TEST_BEGIN("single_simulation");
    void *inst = mcgs_create(1, 1.0f, 1);
    bool board[7][7][2];
    board_start(board);
    void *ss = mcgs_start_search(inst, board, true);
    CHECK(ss != NULL, "start_search returned NULL");
    run_full_search(ss, 0.0f);
    CHECK(mcgs_is_done(ss) == 1, "N=1 search must complete");
    float result[1225];
    mcgs_get_result(ss, result);
    float sum = 0.0f;
    for (int i = 0; i < 1225; i++) sum += result[i];
    mcgs_search_destroy(ss);
    mcgs_destroy(inst);
    CHECK_EQ_F(sum, 1.0f, 1e-4f, "N=1 policy must sum to 1.0");
    PASS();
}

static void test_k_larger_than_legal_moves(void) {
    TEST_BEGIN("k_larger_than_legal_moves");
    /* K=64 but corner piece at (0,0) has only ~8 legal moves.
     * The code must clip K to the number of legal moves and not crash. */
    void *inst = mcgs_create(64, 1.0f, 64);
    bool board[7][7][2];
    board_clear(board);
    place_blue(board,  0, 0);
    place_green(board, 6, 6);
    void *ss = mcgs_start_search(inst, board, true);
    CHECK(ss != NULL, "start_search returned NULL");
    run_full_search(ss, 0.0f);
    CHECK(mcgs_is_done(ss) == 1, "K>legal search must complete");
    float result[1225];
    mcgs_get_result(ss, result);
    float sum = 0.0f;
    for (int i = 0; i < 1225; i++) sum += result[i];
    mcgs_search_destroy(ss);
    mcgs_destroy(inst);
    CHECK_EQ_F(sum, 1.0f, 1e-4f, "policy must sum to 1.0 when K > legal moves");
    PASS();
}

/*
 * Warm-TT path: second search for the same position reuses the already-expanded
 * root node.  The code takes the PHASE_HALVING path immediately rather than
 * PHASE_ROOT_EXPAND.  Must still produce a valid policy.
 */
static void test_warm_tt_search_valid(void) {
    TEST_BEGIN("warm_tt_search_valid");
    void *inst = mcgs_create(16, 1.0f, 8);
    bool board[7][7][2];
    board_start(board);

    void *ss = mcgs_start_search(inst, board, true);
    run_full_search(ss, 0.0f);
    mcgs_search_destroy(ss);

    /* Second search: root is already expanded and in TT */
    ss = mcgs_start_search(inst, board, true);
    CHECK(ss != NULL, "second start_search returned NULL");
    run_full_search(ss, 0.0f);
    float result[1225];
    mcgs_get_result(ss, result);
    float sum = 0.0f;
    for (int i = 0; i < 1225; i++) sum += result[i];
    mcgs_search_destroy(ss);
    mcgs_destroy(inst);
    CHECK_EQ_F(sum, 1.0f, 1e-4f, "warm-TT policy must sum to 1.0");
    PASS();
}

/*
 * Dirichlet noise must not produce negative priors or NaN in the result.
 */
static void test_dirichlet_keeps_result_valid(void) {
    TEST_BEGIN("dirichlet_keeps_result_valid");
    void *inst = mcgs_create(16, 1.0f, 8);
    bool board[7][7][2];
    board_start(board);
    float policy[1225];
    uniform_policy(policy);

    void *ss = mcgs_start_search(inst, board, true);
    CHECK(ss != NULL, "start_search returned NULL");
    CHECK(mcgs_pending_count(ss) == 1, "root must be first pending leaf");
    mcgs_commit_expansion(ss, 0, policy, 0.0f);
    mcgs_apply_root_dirichlet(ss, 0.3f, 0.25f);
    mcgs_step(ss);

    while (!mcgs_is_done(ss)) {
        int n = mcgs_pending_count(ss);
        for (int i = 0; i < n; i++)
            mcgs_commit_expansion(ss, i, policy, 0.0f);
        mcgs_step(ss);
    }

    float result[1225];
    mcgs_get_result(ss, result);
    float sum = 0.0f;
    for (int i = 0; i < 1225; i++) {
        if (result[i] < 0.0f) {
            char msg[64];
            snprintf(msg, sizeof(msg), "result[%d] = %.8f < 0 after Dirichlet", i, result[i]);
            mcgs_search_destroy(ss);
            mcgs_destroy(inst);
            FAIL(msg);
        }
        if (result[i] != result[i]) {   /* NaN check */
            mcgs_search_destroy(ss);
            mcgs_destroy(inst);
            FAIL("NaN in result after Dirichlet noise");
        }
        sum += result[i];
    }
    mcgs_search_destroy(ss);
    mcgs_destroy(inst);
    CHECK_EQ_F(sum, 1.0f, 1e-4f, "Dirichlet-perturbed policy must sum to 1.0");
    PASS();
}

/* =========================================================================
 * Batch API consistency
 *
 * mcgs_get_pending_boards must return the same boards/turns as iterating
 * mcgs_get_leaf_board/turn per index.
 * ========================================================================= */

static void test_batch_api_matches_individual(void) {
    TEST_BEGIN("batch_api_matches_individual");
    void *inst = mcgs_create(8, 1.0f, 4);
    bool board[7][7][2];
    board_start(board);
    void *ss = mcgs_start_search(inst, board, true);
    CHECK(ss != NULL, "start_search returned NULL");

    int n = mcgs_pending_count(ss);
    CHECK(n > 0, "must have at least one pending leaf for batch test");

    /* Collect via individual API */
    bool ind_boards[64][7][7][2];
    bool ind_turns[64];
    for (int i = 0; i < n && i < 64; i++) {
        mcgs_get_leaf_board(ss, i, ind_boards[i]);
        ind_turns[i] = mcgs_get_leaf_turn(ss, i);
    }

    /* Collect via batch API (98 bools = 7*7*2 per board) */
    bool batch_flat[64 * 98];
    bool batch_turns[64];
    int batch_n = mcgs_get_pending_boards(ss, batch_flat, batch_turns);

    if (batch_n != n) {
        char msg[64];
        snprintf(msg, sizeof(msg),
                 "batch_n=%d != pending_count=%d", batch_n, n);
        mcgs_search_destroy(ss);
        mcgs_destroy(inst);
        FAIL(msg);
    }

    for (int i = 0; i < n && i < 64; i++) {
        bool *batch_board = batch_flat + i * 98;
        for (int j = 0; j < 98; j++) {
            if (((bool *)ind_boards[i])[j] != batch_board[j]) {
                char msg[80];
                snprintf(msg, sizeof(msg),
                         "leaf %d byte %d: individual=%d batch=%d",
                         i, j, (int)((bool *)ind_boards[i])[j], (int)batch_board[j]);
                mcgs_search_destroy(ss);
                mcgs_destroy(inst);
                FAIL(msg);
            }
        }
        if (ind_turns[i] != batch_turns[i]) {
            char msg[64];
            snprintf(msg, sizeof(msg),
                     "leaf %d turn: individual=%d batch=%d",
                     i, (int)ind_turns[i], (int)batch_turns[i]);
            mcgs_search_destroy(ss);
            mcgs_destroy(inst);
            FAIL(msg);
        }
    }
    mcgs_search_destroy(ss);
    mcgs_destroy(inst);
    PASS();
}

/* =========================================================================
 * make_move correctness
 * ========================================================================= */

/*
 * Clone (|dx|<=1 AND |dy|<=1): source piece must remain on the board.
 * Jump (|dx|==2 OR |dy|==2): source piece must be removed.
 *
 * We verify by inspecting child boards returned as pending leaves after
 * the root is expanded.  For any child of Blue at (3,3):
 *   - Count Blue pieces: clone -> 2, jump -> 1.
 *   - Never 0 (destination always gets a piece) or 3+ (only one move was made).
 *   - Never have Blue at a jump destination AND still at the source.
 */
static void test_make_move_clone_vs_jump(void) {
    TEST_BEGIN("make_move_clone_vs_jump");
    void *inst = mcgs_create(8, 1.0f, 4);
    bool board[7][7][2];
    board_clear(board);
    place_blue(board,  3, 3);
    place_green(board, 6, 6);   /* far away, most Blue moves won't capture it */

    void *ss = mcgs_start_search(inst, board, true);
    CHECK(ss != NULL, "start_search returned NULL");
    CHECK(mcgs_pending_count(ss) == 1, "root must be first pending leaf");

    float policy[1225];
    uniform_policy(policy);
    mcgs_commit_expansion(ss, 0, policy, 0.0f);
    mcgs_step(ss);

    int n = mcgs_pending_count(ss);
    if (n == 0) {
        /* All children were already in TT (shouldn't happen on first search) */
        mcgs_search_destroy(ss);
        mcgs_destroy(inst);
        FAIL("expected pending children after root expansion");
    }

    for (int i = 0; i < n; i++) {
        bool child[7][7][2];
        mcgs_get_leaf_board(ss, i, child);

        /* Count Blue and Green pieces */
        int blue_n = 0, green_n = 0;
        for (int y = 0; y < 7; y++)
            for (int x = 0; x < 7; x++) {
                if (child[y][x][1]) blue_n++;
                if (child[y][x][0]) green_n++;
            }

        /* Blue must have 1 or 2 pieces; never 0 or 3+ from a single move */
        if (blue_n < 1 || blue_n > 2) {
            char msg[80];
            snprintf(msg, sizeof(msg),
                     "child %d: blue_count=%d (expected 1 or 2 after one move)", i, blue_n);
            mcgs_search_destroy(ss);
            mcgs_destroy(inst);
            FAIL(msg);
        }

        /* Find destination: Blue piece that is NOT at (3,3) */
        int dest_x = -1, dest_y = -1;
        for (int y = 0; y < 7; y++)
            for (int x = 0; x < 7; x++)
                if (child[y][x][1] && !(x == 3 && y == 3))
                    { dest_x = x; dest_y = y; }

        if (dest_x < 0) {
            /* blue_n==1 and the only Blue is still at (3,3): impossible move */
            mcgs_search_destroy(ss);
            mcgs_destroy(inst);
            FAIL("child has no Blue piece at a new destination");
        }

        int dx = dest_x - 3, dy = dest_y - 3;
        bool is_jump = (dx > 1 || dx < -1 || dy > 1 || dy < -1);

        if (is_jump) {
            /* Jump: source at (3,3) must be gone */
            if (child[3][3][1]) {
                char msg[160];
                snprintf(msg, sizeof(msg),
                         "child %d: jump to (%d,%d) but source (3,3) still has Blue "
                         "(jump must clear source)", i, dest_x, dest_y);
                mcgs_search_destroy(ss);
                mcgs_destroy(inst);
                FAIL(msg);
            }
        } else {
            /* Clone: source at (3,3) must still be present */
            if (!child[3][3][1]) {
                char msg[160];
                snprintf(msg, sizeof(msg),
                         "child %d: clone to (%d,%d) but source (3,3) is gone "
                         "(clone must preserve source)", i, dest_x, dest_y);
                mcgs_search_destroy(ss);
                mcgs_destroy(inst);
                FAIL(msg);
            }
        }

        /* Green must remain at (6,6) if the move didn't land adjacent to it */
        int adj = (abs(dest_x - 6) <= 1 && abs(dest_y - 6) <= 1);
        if (!adj && green_n != 1) {
            char msg[80];
            snprintf(msg, sizeof(msg),
                     "child %d -> (%d,%d): green_count=%d, expected 1",
                     i, dest_x, dest_y, green_n);
            mcgs_search_destroy(ss);
            mcgs_destroy(inst);
            FAIL(msg);
        }
        /* If adjacent: Green may or may not be captured depending on distance */
    }

    mcgs_search_destroy(ss);
    mcgs_destroy(inst);
    PASS();
}

/*
 * Capture correctness: cloning next to an opponent piece must convert it.
 *
 * Position: Blue at (3,3), Green at (3,4) (adjacent, dy=1).
 * Blue clones to (4,4): destination (4,4) is adjacent to (3,4) (dx=-1,dy=0).
 *   -> Green at (3,4) captured -> green_count=0 -> terminal, value=+1 Blue.
 * Blue clones to (2,4): destination (2,4) is adjacent to (3,4) (dx=1,dy=0).
 *   -> same capture.
 *
 * With network_value=0 and 64 simulations, root Q must be > 0 because the
 * search finds and backrefs the winning terminal children.
 * A root Q of 0 indicates captures are silently failing (Green never dies).
 */
static void test_make_move_capture_works(void) {
    TEST_BEGIN("make_move_capture_works");
    void *inst = mcgs_create(64, 1.0f, 16);
    bool board[7][7][2];
    board_clear(board);
    place_blue(board,  3, 3);
    place_green(board, 3, 4);

    void *ss = mcgs_start_search(inst, board, true);
    CHECK(ss != NULL, "start_search returned NULL");
    run_full_search(ss, 0.0f);

    float root_q = mcgs_get_root_value(ss);
    mcgs_search_destroy(ss);
    mcgs_destroy(inst);

    CHECK(root_q > 0.0f,
          "root Q must be > 0 when captures work (Blue immediately wins via capture)");
    PASS();
}

/*
 * Jump clears source AND captures: Blue at (3,3) jumps 2 right to (5,3).
 * Any child board in the search that has Blue at (5,3) must NOT also have Blue
 * at (3,3) (source must be cleared).
 *
 * We run the full search and inspect every pending leaf board encountered.
 */
static void test_make_move_jump_clears_source(void) {
    TEST_BEGIN("make_move_jump_clears_source");
    void *inst = mcgs_create(32, 1.0f, 8);
    bool board[7][7][2];
    board_clear(board);
    place_blue(board,  3, 3);
    place_green(board, 6, 6);

    float policy[1225];
    uniform_policy(policy);

    void *ss = mcgs_start_search(inst, board, true);
    CHECK(ss != NULL, "start_search returned NULL");

    int steps = 0;
    while (!mcgs_is_done(ss) && steps++ < 2000) {
        int n = mcgs_pending_count(ss);
        for (int i = 0; i < n; i++) {
            bool child[7][7][2];
            mcgs_get_leaf_board(ss, i, child);
            /* If Blue landed at (5,3) (a dx=2 jump from (3,3)), source must be clear */
            if (child[3][5][1] && child[3][3][1]) {
                mcgs_search_destroy(ss);
                mcgs_destroy(inst);
                FAIL("child has Blue at both (5,3) and (3,3): jump did not clear source");
            }
            /* General: after one move, no single-piece board should gain 3+ Blue pieces */
            int blue_n = 0;
            for (int y = 0; y < 7; y++)
                for (int x = 0; x < 7; x++)
                    if (child[y][x][1]) blue_n++;
            if (blue_n > 2) {
                char msg[80];
                snprintf(msg, sizeof(msg),
                         "leaf %d has %d Blue pieces after one move (max 2)", i, blue_n);
                mcgs_search_destroy(ss);
                mcgs_destroy(inst);
                FAIL(msg);
            }
            mcgs_commit_expansion(ss, i, policy, 0.0f);
        }
        mcgs_step(ss);
    }

    mcgs_search_destroy(ss);
    mcgs_destroy(inst);
    PASS();
}

/* =========================================================================
 * main
 * ========================================================================= */

int main(void) {
    mcgs_init();
    printf("=== micro_mcts unit tests ===\n\n");

    /* Lifecycle */
    test_create_destroy();
    test_clear_and_reuse();

    /* Transposition table */
    test_tt_size_zero_after_create();
    test_tt_size_grows_during_search();
    test_tt_size_zero_after_clear();
    test_tt_nondecreasing_during_search();

    /* Board / turn roundtrip */
    test_root_leaf_board_roundtrip();
    test_root_leaf_turn_blue();
    test_root_leaf_turn_green();

    /* Terminal detection */
    test_terminal_blue_eliminated();
    test_terminal_green_eliminated();
    test_forced_pass_not_terminal();
    test_both_stuck_terminal();

    /* Value perspective */
    test_value_perspective_blue_winning();
    test_value_perspective_green_winning();

    /* Policy invariants */
    test_policy_sums_to_one();
    test_policy_nonnegative();
    test_policy_zero_on_opponent_pieces();
    test_policy_nonzero_on_legal_moves();

    /* Cross-game isolation */
    test_tt_fully_cleared_between_games();
    test_repeated_cycles_stay_bounded();
    test_fresh_root_after_clear();

    /* Multi-instance isolation */
    test_two_instances_independent();

    /* Bounds and null-safety */
    test_oob_leaf_index_no_crash();
    test_null_instance_no_crash();
    test_null_search_state_no_crash();

    /* Edge cases */
    test_single_simulation();
    test_k_larger_than_legal_moves();
    test_warm_tt_search_valid();
    test_dirichlet_keeps_result_valid();
    test_batch_api_matches_individual();

    /* make_move correctness */
    test_make_move_clone_vs_jump();
    test_make_move_capture_works();
    test_make_move_jump_clears_source();

    printf("\n=== Results: %d/%d passed", g_tests - g_failures, g_tests);
    if (g_failures == 0) printf(" -- ALL PASSED ===\n");
    else                 printf(" (%d FAILED) ===\n", g_failures);

    return g_failures == 0 ? 0 : 1;
}

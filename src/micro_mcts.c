/*
 * micro_mcts.c - C implementation of Gumbel MCGS for Microscope (T7G).
 *
 * Manages the MCTS tree and transposition table in C heap memory
 * to avoid memory churning.
 *
 * Network inference stays in Python.  The step-wise interface lets the Python
 * game pool batch leaf nodes across all concurrent games into one GPU pass.
 *
 * Public API:
 *   mcgs_init()               - init Zobrist tables; call once at startup
 *   mcgs_create(sims,c,k)     - allocate a new MCGS instance
 *   mcgs_clear(inst)          - free all nodes, keep instance alive
 *   mcgs_destroy(inst)        - free instance and all its nodes
 *   mcgs_tt_size(inst)        - transposition table node count (monitoring)
 *   mcgs_start_search(…)      - begin a search, returns MCGSSearchState*
 *   mcgs_search_destroy(ss)   - free the search state struct
 *   mcgs_pending_count(ss)    - number of leaves needing network expansion
 *   mcgs_get_leaf_board(ss,i) - copy i-th pending leaf board (bool[7][7][2])
 *   mcgs_get_leaf_turn(ss,i)  - turn of i-th pending leaf
 *   mcgs_commit_expansion(…)  - feed policy[1225] + value for i-th leaf
 *   mcgs_step(ss)             - backprop + advance; returns new pending count
 *   mcgs_is_done(ss)          - 1 when search is complete
 *   mcgs_get_result(ss, out)  - write action_probs[1225] to out
 *   mcgs_get_root_value(ss)   - root Q (value_sum/visit_count, mover's perspective)
 */
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <stdio.h>

/*  Constants  */
#define BOARD_SIZE     7
#define EMPTY          0
#define GREEN          1
#define BLUE           2
#define PASS_ACTION    1225   /* sentinel: no legal moves but game continues */
#define CLOCK_LIMIT    100    /* libataxx halfmove clock: game is drawn after
                                 100 plies without a clone move.  Jumps and
                                 passes tick the clock; clones reset it.  The
                                 clock is walk-state, not node-state: nodes /
                                 the TT stay keyed on (board, turn) alone,
                                 matching libataxx's choice to exclude the
                                 clock from the zobrist hash.               */

#define MAX_K          64     /* max Gumbel top-K */

/* Gumbel MuZero sigma(q) transform: sigma(q) = (c_visit + max_N) * c_scale * q.
 * Scales Q from [-1,1] into logit space so completed-Q can actually override
 * the prior once the search is confident.  Without it the policy target is
 * softmax(log_prior + q) which the prior dominates (q adds at most +-1 nat)
 * and MCTS degenerates to rubber-stamping the network. */
#define SIGMA_C_VISIT  50.0f
#define SIGMA_C_SCALE  1.0f
#define COMPLETION_N0  50.0f  /* default visit-shrinkage strength, see finish_search */
#define MAX_PATH_DEPTH 128    /* max depth of one simulation path           */
#define MAX_PENDING    64     /* max unique leaves per batch (≤ MAX_K)      */

/* Default arena slab capacities - sized for one game at 250 sims × ~100 moves.
 * The TT persists across moves (advance_tree is a no-op), so the arena grows
 * for the whole game and these caps set how many moves fit before the arena
 * fills and start_search forces a clear.  Measured growth is dead linear in
 * the sim budget: ~0.95 nodes and ~52 edges per simulation, per move.  So
 * 250 sims ≈ 240 nodes/move (a full game fits easily), but 8000 sims is
 * ~7.6k nodes + 450k edges per move and fills these defaults by move ~14.
 *
 * Per instance at the defaults: ~130 MB (node slab 7 MB + edge slab 123 MB +
 * HT 2 MB), of which only the touched pages are ever resident -- a 512-slot
 * self-play pool stays cheap because at 250 sims it touches a few MB each.
 * High-sim evaluation needs a much bigger arena but only a handful of
 * instances, so the caps are per-instance (mcgs_create_ex) rather than
 * compile-time: raising them globally would blow up the training pool's
 * address space for no benefit. */
#define NODE_SLAB_CAP  110000
#define EDGE_SLAB_CAP  7700000
#define HT_BITS        18
#define HT_SIZE        (1 << HT_BITS)   /* 262144 slots - must exceed NODE_SLAB_CAP */

/* Search phase codes */
#define PHASE_ROOT_EXPAND  0
#define PHASE_HALVING      1
#define PHASE_TAIL         2
#define PHASE_DONE         3

/*  Data structures  */

typedef struct {
    int16_t           action;
    float             prior;
    struct MCGSNode_s *child;  /* NULL until first visited */
} MCGSEdge;

typedef struct MCGSNode_s {
    uint64_t  hash;            /* Zobrist hash - open-HT key             */
    uint64_t  green_bb;        /* bitboards, low 49 bits, bit i = (i/7,i%7) */
    uint64_t  blue_bb;
    bool      turn;            /* true = Blue to move                    */
    int       visit_count;
    float     value_sum;
    float     network_value;   /* from most-recent network eval          */
    bool      is_expanded;
    bool      is_terminal;
    float     terminal_value;  /* from current mover's perspective       */
    int       num_edges;
    MCGSEdge *edges;           /* points into inst->edge_slab            */
} MCGSNode;

typedef struct {
    MCGSNode *nodes[MAX_PATH_DEPTH];
    int       len;
    bool      is_cycle;
    bool      is_clock_draw;   /* halfmove clock hit CLOCK_LIMIT on this walk */
    uint8_t   leaf_clock;      /* clock value at the path's leaf              */
} SimPath;

typedef struct {
    /* Open-addressed hash table (fixed size, O(1) clear via memset) */
    MCGSNode **ht;             /* ht_size slots, NULL-terminated probing   */
    uint32_t   ht_size;        /* power of two, must exceed node_cap       */
    uint32_t   ht_mask;        /* ht_size - 1                              */
    /* Arena slabs - bump-pointer allocation, reset on clear             */
    MCGSNode  *node_slab;
    int        node_cap;
    int        node_used;
    MCGSEdge  *edge_slab;
    int        edge_cap;
    int        edge_used;
    /* Search parameters */
    int       num_simulations;
    float     c_puct;
    int       gumbel_k;
    float     sigma_scale;     /* multiplier on sigma(q); 1.0 = paper default */
    float     completion_n0;   /* visit-shrinkage prior strength for the
                                  completed-Q policy target: q~(a) =
                                  (n_a*q_a + n0*v_root) / (n_a + n0).
                                  Damps low-visit Q noise before sigma
                                  amplifies it into target logits. */
    uint64_t  rng;             /* per-instance xorshift64 RNG state        */
    int       clock_in_obs;    /* expose halfmove clock as obs ch3 (default off
                                  so legacy nets keep their all-zero plane) */
} MCGSInstance;

typedef struct {
    MCGSInstance *inst;
    MCGSNode     *root;

    /* Gumbel / halving state */
    float   gumbel[MAX_K];
    float   log_prior[MAX_K];
    int16_t top_k[MAX_K];
    int     K, N, R;
    int16_t active[MAX_K];
    int     num_active;
    int     sims_done, n_r, round_idx;
    int     phase;

    int root_clock;            /* halfmove clock at the root position */

    /* Pending leaves (need expansion before next step) */
    MCGSNode *pending[MAX_PENDING];
    uint8_t   pending_clock[MAX_PENDING];  /* walk clock at each pending leaf */
    int       num_pending;

    /* Current batch of simulation paths (for backprop) */
    SimPath paths[MAX_K];
    int     num_paths;

    /* Completed result */
    float result[1225];
    int   best_action;   /* Sequential Halving winner (-1 if none) */
} MCGSSearchState;

/*  Zobrist / RNG  */

static uint64_t zobrist_board[49][3];
static uint64_t zobrist_turn;
static bool     zobrist_ready = false;

/* Bitboard support tables.  Hash compatibility: the legacy hash XORed
 * zobrist_board[pos][cell] over ALL 49 cells including empties.  With
 * hash_base = XOR of all empty-cell keys and zg/zb = (empty ^ piece) delta
 * keys, the bitboard hash below reproduces the legacy values bit-for-bit,
 * keeping searches byte-identical to the array-board implementation. */
static uint64_t hash_base;
static uint64_t zg[49], zb[49];        /* zobrist delta: empty -> green/blue */
static uint64_t neighbor1_mask[49];    /* Chebyshev dist 1 (clone ring)      */
static uint64_t neighbor2_mask[49];    /* Chebyshev dist <= 2 (all moves)    */
static int16_t  action_of[49][49];     /* [from][to] -> action code, -1      */
static int8_t   move_from_tbl[1225];
static int8_t   move_to_tbl[1225];     /* -1 if destination off-board        */
static bool     move_is_jump_tbl[1225];

static uint64_t xorshift64(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *s = x;
    return x;
}

static void init_zobrist(void) {
    if (zobrist_ready) return;
    uint64_t seed = 0xFEEDC0FFEE123456ULL;
    for (int i = 0; i < 49; i++)
        for (int j = 0; j < 3; j++)
            zobrist_board[i][j] = xorshift64(&seed);
    zobrist_turn  = xorshift64(&seed);

    hash_base = 0;
    for (int i = 0; i < 49; i++) {
        hash_base ^= zobrist_board[i][EMPTY];
        zg[i] = zobrist_board[i][EMPTY] ^ zobrist_board[i][GREEN];
        zb[i] = zobrist_board[i][EMPTY] ^ zobrist_board[i][BLUE];
    }

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
    }

    memset(action_of, -1, sizeof(action_of));
    for (int from = 0; from < 49; from++) {
        for (int mv = 0; mv < 25; mv++) {
            int action = from * 25 + mv;
            int dx = (mv % 5) - 2, dy = (mv / 5) - 2;
            int x  = from % 7 + dx, y = from / 7 + dy;
            move_from_tbl[action]    = (int8_t)from;
            move_is_jump_tbl[action] = (abs(dx) == 2 || abs(dy) == 2);
            if (x < 0 || x >= 7 || y < 0 || y >= 7) {
                move_to_tbl[action] = -1;
            } else {
                int to = y * 7 + x;
                move_to_tbl[action]  = (int8_t)to;
                if (dx != 0 || dy != 0) action_of[from][to] = (int16_t)action;
            }
        }
    }

    zobrist_ready = true;
}

/* [0, 1) uniform float via 53-bit mantissa trick */
static float rng_uniform(uint64_t *rng) {
    float u;
    do {
        u = (float)((xorshift64(rng) >> 11)) * (1.0f / (float)(1ULL << 53));
    } while (u <= 0.0f || u >= 1.0f);
    return u;
}

static float gumbel_noise(uint64_t *rng) {
    float u = rng_uniform(rng);
    return -logf(-logf(u));
}


static uint64_t compute_hash(uint64_t green_bb, uint64_t blue_bb, bool turn) {
    uint64_t h = hash_base;
    uint64_t bbs = green_bb;
    while (bbs) { h ^= zg[__builtin_ctzll(bbs)]; bbs &= bbs - 1; }
    bbs = blue_bb;
    while (bbs) { h ^= zb[__builtin_ctzll(bbs)]; bbs &= bbs - 1; }
    if (turn) h ^= zobrist_turn;
    return h;
}

/*  Board utilities (bitboards, low 49 bits, bit i = cell (i/7, i%7))  */

/* Python bool[7][7][2] (ch0=green, ch1=blue) -> bitboards */
static void convert_board(bool py[7][7][2], uint64_t *green_bb, uint64_t *blue_bb) {
    uint64_t g = 0, b = 0;
    const bool *flat = (const bool *)py;
    for (int pos = 0; pos < 49; pos++) {
        g |= (uint64_t)(flat[pos * 2 + 0] != 0) << pos;
        b |= (uint64_t)(flat[pos * 2 + 1] != 0) << pos;
    }
    *green_bb = g;
    *blue_bb  = b;
}

/* bitboards -> Python bool[7][7][2] */
static void export_board(uint64_t green_bb, uint64_t blue_bb, bool out[7][7][2]) {
    bool *flat = (bool *)out;
    for (int pos = 0; pos < 49; pos++) {
        flat[pos * 2 + 0] = (green_bb >> pos) & 1;
        flat[pos * 2 + 1] = (blue_bb  >> pos) & 1;
    }
}

static bool has_moves_bits(uint64_t mover_bb, uint64_t occupied) {
    uint64_t pieces = mover_bb;
    while (pieces) {
        int sq = __builtin_ctzll(pieces); pieces &= pieces - 1;
        if (neighbor2_mask[sq] & ~occupied) return true;
    }
    return false;
}

/* Emit legal action codes in ascending order (pieces ascend by square; for a
 * fixed source, ascending destination square is ascending action code) -
 * matching the legacy 0..1224 valid-mask scan order exactly. */
static int gen_actions(uint64_t mover_bb, uint64_t occupied, int16_t *out) {
    int count = 0;
    uint64_t pieces = mover_bb;
    while (pieces) {
        int from_sq = __builtin_ctzll(pieces); pieces &= pieces - 1;
        const int16_t *acts = action_of[from_sq];
        uint64_t dests = neighbor2_mask[from_sq] & ~occupied;
        while (dests) {
            int to_sq = __builtin_ctzll(dests); dests &= dests - 1;
            out[count++] = acts[to_sq];
        }
    }
    return count;
}

/* Apply an action for the given mover; returns updated bitboards in place. */
static void make_move_bits(uint64_t *mover_bb, uint64_t *opp_bb, int action) {
    uint64_t m = *mover_bb, o = *opp_bb;
    int to_sq = move_to_tbl[action];
    if (move_is_jump_tbl[action]) m &= ~(1ULL << move_from_tbl[action]);
    m |= (1ULL << to_sq);
    uint64_t captured = neighbor1_mask[to_sq] & o;
    *mover_bb = m | captured;
    *opp_bb   = o ^ captured;
}

/*
 * Check terminal from current mover's perspective.
 * Returns true and sets *value (+1/-1/0) if terminal.
 * Matches Python check_terminal: only terminal when one side has 0 pieces,
 * or BOTH players simultaneously have no legal moves.
 */
static bool check_terminal(uint64_t green_bb, uint64_t blue_bb, bool turn,
                           float *value) {
    if (blue_bb == 0) {
        *value = turn ? -1.0f : 1.0f;
        return true;
    }
    if (green_bb == 0) {
        *value = turn ? 1.0f : -1.0f;
        return true;
    }
    uint64_t occupied = green_bb | blue_bb;
    uint64_t mover = turn ? blue_bb : green_bb;
    uint64_t other = turn ? green_bb : blue_bb;
    if (has_moves_bits(mover, occupied)) return false;
    if (has_moves_bits(other, occupied)) return false;
    /* Both stuck: terminal by count (from mover's perspective) */
    int blue  = __builtin_popcountll(blue_bb);
    int green = __builtin_popcountll(green_bb);
    float raw = (blue > green) ? 1.0f : (green > blue) ? -1.0f : 0.0f;
    *value = turn ? raw : -raw;
    return true;
}

/*  Transposition table (open-addressing, arena-backed)  */

static MCGSNode *tt_find(MCGSInstance *inst, uint64_t h) {
    uint32_t idx = (uint32_t)(h & inst->ht_mask);
    for (;;) {
        MCGSNode *n = inst->ht[idx];
        if (!n) return NULL;
        if (n->hash == h) return n;
        idx = (idx + 1) & inst->ht_mask;
    }
}

static void tt_insert(MCGSInstance *inst, MCGSNode *n) {
    uint32_t idx = (uint32_t)(n->hash & inst->ht_mask);
    while (inst->ht[idx]) idx = (idx + 1) & inst->ht_mask;
    inst->ht[idx] = n;
}

/* O(1) clear: zero the HT bucket array and reset bump pointers. */
static void tt_clear(MCGSInstance *inst) {
    memset(inst->ht, 0, inst->ht_size * sizeof(MCGSNode *));
    inst->node_used = 0;
    inst->edge_used = 0;
}

/* Get or create node for (board, turn), inserting into TT if new. */
static MCGSNode *tt_get_or_create(MCGSInstance *inst, uint64_t green_bb,
                                  uint64_t blue_bb, bool turn) {
    uint64_t h = compute_hash(green_bb, blue_bb, turn);
    MCGSNode *n = tt_find(inst, h);
    if (n) return n;
    if (inst->node_used >= inst->node_cap) {
        fprintf(stderr, "[mcgs] node slab full (%d nodes) - mcgs_start_search will return NULL\n",
                inst->node_cap);
        return NULL;
    }
    n = &inst->node_slab[inst->node_used++];
    memset(n, 0, sizeof(MCGSNode));
    n->hash     = h;
    n->green_bb = green_bb;
    n->blue_bb  = blue_bb;
    n->turn     = turn;
    n->is_terminal = check_terminal(green_bb, blue_bb, turn, &n->terminal_value);
    tt_insert(inst, n);
    return n;
}

/*  PUCT selection helpers  */

static float node_q(MCGSNode *n) {
    return (n->visit_count > 0) ? n->value_sum / (float)n->visit_count : 0.0f;
}

/* Q for a root action from root's perspective. */
static float action_q(MCGSNode *root, int action, float fallback) {
    for (int i = 0; i < root->num_edges; i++) {
        if (root->edges[i].action != (int16_t)action) continue;
        MCGSNode *c = root->edges[i].child;
        if (!c || c->visit_count == 0) return fallback;
        return -node_q(c);
    }
    return fallback;
}

/* Visit count of a root action's child (0 if unexpanded). */
static int action_visits(MCGSNode *root, int action) {
    for (int i = 0; i < root->num_edges; i++) {
        if (root->edges[i].action != (int16_t)action) continue;
        MCGSNode *c = root->edges[i].child;
        return c ? c->visit_count : 0;
    }
    return 0;
}

/* sigma(q) multiplier for the current root: (c_visit + max_a N(a)) * c_scale.
 * max is taken over the top-K candidate actions. */
static float sigma_mult(const MCGSInstance *inst, MCGSNode *root,
                        const int16_t *actions, int n) {
    int max_n = 0;
    for (int i = 0; i < n; i++) {
        int v = action_visits(root, (int)actions[i]);
        if (v > max_n) max_n = v;
    }
    return (SIGMA_C_VISIT + (float)max_n) * inst->sigma_scale;
}

static int best_action_puct(MCGSNode *node, float c_puct) {
    float best = -1e30f;
    int   best_action = -1;
    float sqrt_n = sqrtf((float)node->visit_count);
    for (int i = 0; i < node->num_edges; i++) {
        MCGSEdge *e = &node->edges[i];
        MCGSNode *c = e->child;
        float q    = (c == NULL) ? 0.0f : -node_q(c);
        int   v    = (c == NULL) ? 0    : c->visit_count;
        float s    = q + c_puct * e->prior * sqrt_n / (1.0f + v);
        if (s > best) { best = s; best_action = (int)e->action; }
    }
    return best_action;
}

/* Get or create child node for the given action. */
static MCGSNode *get_or_create_child(MCGSInstance *inst,
                                     MCGSNode *node, int action) {
    for (int i = 0; i < node->num_edges; i++) {
        MCGSEdge *e = &node->edges[i];
        if (e->action != (int16_t)action) continue;
        if (!e->child) {
            uint64_t g = node->green_bb, b = node->blue_bb;
            if (action != PASS_ACTION) {
                if (node->turn) make_move_bits(&b, &g, action);
                else            make_move_bits(&g, &b, action);
            }
            e->child = tt_get_or_create(inst, g, b, !node->turn);
        }
        return e->child;
    }
    return NULL;
}

/* Select one simulation path from root, forcing first_action at the root. */
static void select_path(MCGSInstance *inst, MCGSNode *root, int root_clock,
                        int first_action, SimPath *path) {
    path->len           = 0;
    path->is_cycle      = false;
    path->is_clock_draw = false;

    uint64_t visited[MAX_PATH_DEPTH];
    int      nv = 0;

    MCGSNode *node = root;
    path->nodes[path->len++] = node;
    visited[nv++] = node->hash;
    int clk = root_clock;
    path->leaf_clock = (uint8_t)clk;

    int forced = first_action;
    while (node->is_expanded && !node->is_terminal) {
        int action = (forced >= 0) ? forced : best_action_puct(node, inst->c_puct);
        forced = -1;
        if (action < 0) break;

        MCGSNode *child = get_or_create_child(inst, node, action);
        if (!child) break;

        clk = (action == PASS_ACTION || move_is_jump_tbl[action]) ? clk + 1 : 0;

        /* Cycle detection */
        for (int i = 0; i < nv; i++) {
            if (visited[i] == child->hash) {
                path->nodes[path->len++] = child;
                path->is_cycle = true;
                return;
            }
        }
        path->nodes[path->len++] = child;
        path->leaf_clock = (uint8_t)(clk < 255 ? clk : 255);

        /* Halfmove-clock draw: a real terminal (elimination / both stuck)
         * takes precedence, matching libataxx get_result() ordering. */
        if (!child->is_terminal && clk >= CLOCK_LIMIT) {
            path->is_clock_draw = true;
            return;
        }

        if (nv < MAX_PATH_DEPTH) visited[nv++] = child->hash;
        if (path->len >= MAX_PATH_DEPTH) break;
        node = child;
    }
}

/* Backpropagate a simulation path. */
static void backprop_path(SimPath *path) {
    if (path->len == 0) return;
    MCGSNode *leaf = path->nodes[path->len - 1];
    float value;
    if (path->is_cycle || path->is_clock_draw) {
        value = 0.0f;
    } else if (leaf->is_terminal) {
        value = leaf->terminal_value;  /* current mover's perspective */
    } else {
        value = leaf->network_value;
    }
    for (int i = path->len - 1; i >= 0; i--) {
        path->nodes[i]->visit_count++;
        path->nodes[i]->value_sum += value;
        value = -value;
    }
}

/*  Gumbel / halving helpers  */

typedef struct { int16_t action; float score; float log_prior; } SortEntry;
static int cmp_score_desc(const void *a, const void *b) {
    float fa = ((SortEntry *)a)->score, fb = ((SortEntry *)b)->score;
    return (fb > fa) - (fb < fa);
}

/* Finish: compute completed-Q policy and mark PHASE_DONE.
 *
 * Improved policy = softmax(log_prior + sigma(q~)) over ALL legal actions
 * (full completion, Gumbel MuZero eq. 10), with visit-count shrinkage
 *
 *     q~(a) = (n_a * q(a) + n0 * v_root) / (n_a + n0)
 *
 * so an action's Q can only move the target in proportion to how well it
 * was actually measured.  Unvisited actions get exactly v_root - a constant
 * logit shift - so their relative mass stays the raw prior.  Without the
 * shrinkage, sigma_mult (~200 at 500 sims) turns the ~3x-noisier Q of a
 * 7-visit halving loser into a target-argmax lottery: a measured 15-27%
 * argmax flip rate between two independent searches of the same position,
 * which poisoned both self-play moves and training targets.
 *
 * The Sequential Halving winner is recorded in ss->best_action; play THAT
 * at temperature 0 (the certified action), not argmax(result). */
static void finish_search(MCGSSearchState *ss) {
    MCGSNode *root = ss->root;
    float n0 = ss->inst->completion_n0;
    float v_root = (root->visit_count > 0) ? node_q(root) : root->network_value;
    float sm = sigma_mult(ss->inst, root, ss->top_k, ss->K);

    float logits[1225];
    int16_t acts[1225];
    int n_out = 0;
    float best_logit = -1e30f;
    for (int i = 0; i < root->num_edges; i++) {
        MCGSEdge *e = &root->edges[i];
        if (e->action == (int16_t)PASS_ACTION) continue;
        MCGSNode *c = e->child;
        float n_a = (c != NULL) ? (float)c->visit_count : 0.0f;
        float q_a = (n_a > 0.0f) ? -node_q(c) : v_root;
        float q_shrunk = (n_a + n0 > 0.0f)
                       ? (n_a * q_a + n0 * v_root) / (n_a + n0)
                       : v_root;
        acts[n_out]   = e->action;
        logits[n_out] = logf(fmaxf(e->prior, 1e-9f)) + sm * q_shrunk;
        if (logits[n_out] > best_logit) best_logit = logits[n_out];
        n_out++;
    }

    memset(ss->result, 0, 1225 * sizeof(float));
    if (n_out > 0) {
        float sum = 0.0f;
        for (int i = 0; i < n_out; i++) {
            logits[i] = expf(logits[i] - best_logit);
            sum += logits[i];
        }
        for (int i = 0; i < n_out; i++)
            ss->result[acts[i]] = logits[i] / sum;
    }
    ss->best_action = (ss->num_active > 0) ? (int)ss->active[0] : -1;
    ss->phase       = PHASE_DONE;
    ss->num_pending = 0;
    ss->num_paths   = 0;
}

/* Collect unique unexpanded non-terminal leaves from ss->paths. */
static void collect_pending(MCGSSearchState *ss) {
    ss->num_pending = 0;
    for (int p = 0; p < ss->num_paths && ss->num_pending < MAX_PENDING; p++) {
        SimPath *path = &ss->paths[p];
        if (path->is_cycle || path->is_clock_draw || path->len == 0) continue;
        MCGSNode *leaf = path->nodes[path->len - 1];
        if (leaf->is_terminal || leaf->is_expanded) continue;
        bool dup = false;
        for (int j = 0; j < ss->num_pending; j++)
            if (ss->pending[j] == leaf) { dup = true; break; }
        if (!dup) {
            ss->pending_clock[ss->num_pending] = path->leaf_clock;
            ss->pending[ss->num_pending++]     = leaf;
        }
    }
}

static void select_paths_halving(MCGSSearchState *ss) {
    ss->num_paths = ss->num_active;
    for (int i = 0; i < ss->num_active; i++)
        select_path(ss->inst, ss->root, ss->root_clock, (int)ss->active[i], &ss->paths[i]);
    collect_pending(ss);
}

static void select_paths_tail(MCGSSearchState *ss) {
    if (ss->sims_done >= ss->N) { finish_search(ss); return; }
    int batch = ss->K < (ss->N - ss->sims_done) ? ss->K : (ss->N - ss->sims_done);
    ss->num_paths = batch;
    for (int i = 0; i < batch; i++)
        select_path(ss->inst, ss->root, ss->root_clock, (int)ss->active[0], &ss->paths[i]);
    collect_pending(ss);
}

static void halving_prune(MCGSSearchState *ss) {
    SortEntry ranked[MAX_K];
    MCGSNode *root = ss->root;
    float sm = sigma_mult(ss->inst, root, ss->active, ss->num_active);
    for (int i = 0; i < ss->num_active; i++) {
        /* Sequential Halving ranks by g(a) + logits(a) + sigma(q(a)).
         * ss->gumbel[j] already holds log_prior + gumbel noise from setup. */
        int j = 0;
        while (j < ss->K && ss->top_k[j] != ss->active[i]) j++;
        float gl = (j < ss->K) ? ss->gumbel[j] : ss->log_prior[0];
        ranked[i].action = ss->active[i];
        ranked[i].score  = gl + sm * action_q(root, (int)ss->active[i],
                                              root->network_value);
    }
    qsort(ranked, ss->num_active, sizeof(SortEntry), cmp_score_desc);
    int new_n = (ss->num_active + 1) / 2;  /* ceil(K/2) per Sequential Halving */
    if (new_n < 1) new_n = 1;
    ss->num_active = new_n;
    for (int i = 0; i < new_n; i++) ss->active[i] = ranked[i].action;
    ss->round_idx++;
}

/*
 * Set up Gumbel top-K and Sequential Halving state on a freshly expanded root.
 * Returns false if no legal actions exist (result already set to zeros).
 */
static bool setup_halving(MCGSSearchState *ss) {
    MCGSNode *root = ss->root;
    SortEntry legal[1225];
    int n_legal = 0;
    for (int i = 0; i < root->num_edges; i++) {
        if (root->edges[i].action == (int16_t)PASS_ACTION) continue;
        float lp = logf(fmaxf(root->edges[i].prior, 1e-9f));
        legal[n_legal].action    = root->edges[i].action;
        legal[n_legal].log_prior = lp;
        legal[n_legal].score     = lp + gumbel_noise(&ss->inst->rng);
        n_legal++;
    }
    if (n_legal == 0) {
        memset(ss->result, 0, sizeof(ss->result));
        ss->phase = PHASE_DONE; ss->num_pending = 0;
        return false;
    }
    int K = (n_legal < ss->inst->gumbel_k) ? n_legal : ss->inst->gumbel_k;
    qsort(legal, n_legal, sizeof(SortEntry), cmp_score_desc);
    for (int i = 0; i < K; i++) {
        ss->top_k[i]   = legal[i].action;
        ss->gumbel[i]  = legal[i].score;
        ss->log_prior[i] = legal[i].log_prior;
        ss->active[i]  = legal[i].action;
    }
    ss->K = K; ss->N = ss->inst->num_simulations; ss->num_active = K;
    ss->sims_done = 0; ss->round_idx = 0;
    int R = 1; while ((1 << R) < K) R++; ss->R = R;
    if (K <= 1) {
        ss->phase = PHASE_TAIL;
        select_paths_tail(ss);
    } else {
        int n_r = ss->N / (ss->R * ss->num_active);
        if (n_r < 1) n_r = 1;
        ss->n_r   = n_r;
        ss->phase = PHASE_HALVING;
        select_paths_halving(ss);
    }
    return true;
}

/*  Public API  */

void mcgs_init(void) {
    init_zobrist();
}

/* Full constructor: caller sizes the arena.  node_cap/edge_cap <= 0 mean
 * "use the default".  The HT is grown to the next power of two >= 2*node_cap
 * so the open-addressed table never exceeds 50% load (probing degrades badly
 * as it approaches full, and it must strictly exceed node_cap to terminate). */
MCGSInstance *mcgs_create_ex(int num_simulations, float c_puct, int gumbel_k,
                             int node_cap, int edge_cap) {
    init_zobrist();
    if (node_cap <= 0) node_cap = NODE_SLAB_CAP;
    if (edge_cap <= 0) edge_cap = EDGE_SLAB_CAP;
    MCGSInstance *inst = (MCGSInstance *)calloc(1, sizeof(MCGSInstance));
    if (!inst) return NULL;
    uint32_t ht_size = 1u << HT_BITS;
    while (ht_size < (uint32_t)node_cap * 2u) ht_size <<= 1;
    inst->ht_size   = ht_size;
    inst->ht_mask   = ht_size - 1u;
    inst->node_cap  = node_cap;
    inst->edge_cap  = edge_cap;
    inst->ht        = (MCGSNode **)calloc(ht_size, sizeof(MCGSNode *));
    inst->node_slab = (MCGSNode  *)malloc((size_t)node_cap * sizeof(MCGSNode));
    inst->edge_slab = (MCGSEdge  *)malloc((size_t)edge_cap * sizeof(MCGSEdge));
    if (!inst->ht || !inst->node_slab || !inst->edge_slab) {
        free(inst->ht); free(inst->node_slab); free(inst->edge_slab); free(inst);
        return NULL;
    }
    inst->num_simulations = num_simulations;
    inst->c_puct          = c_puct;
    inst->gumbel_k        = gumbel_k;
    inst->sigma_scale     = SIGMA_C_SCALE;
    inst->completion_n0   = COMPLETION_N0;
    inst->rng             = (uint64_t)time(NULL) ^ (uint64_t)(uintptr_t)inst;
    if (inst->rng == 0) inst->rng = 0xDEADC0DEULL;
    return inst;
}

/* Back-compat constructor: default arena. */
MCGSInstance *mcgs_create(int num_simulations, float c_puct, int gumbel_k) {
    return mcgs_create_ex(num_simulations, c_puct, gumbel_k, 0, 0);
}

void mcgs_clear(MCGSInstance *inst) {
    if (inst) tt_clear(inst);
}

/* Scale sigma(q) by s (default 1.0 = Gumbel MuZero paper setting).  Lower
 * values make the completed-Q policy stickier to the prior. */
void mcgs_set_sigma_scale(MCGSInstance *inst, float s) {
    if (inst) inst->sigma_scale = s;
}

/* Per-search simulation budget.  Takes effect at the next mcgs_start_search;
 * in-flight searches keep the N they were started with.  Used for playout-cap
 * randomization: most self-play moves run a cheap search (value data), a
 * random fraction run the full budget (policy targets). */
void mcgs_set_num_simulations(MCGSInstance *inst, int n) {
    if (inst && n > 0) inst->num_simulations = n;
}

/* Pin the Gumbel-noise RNG (default seed is time^instance-address, i.e.
 * nondeterministic).  For reproducible searches in tests/verification. */
void mcgs_set_rng_seed(MCGSInstance *inst, uint64_t seed) {
    if (inst) inst->rng = seed ? seed : 0xDEADC0DEULL;
}

/* Visit-shrinkage prior strength n0 for the completed-Q target (see
 * finish_search).  Larger = low-visit Q estimates trust the root value
 * more; 0 disables shrinkage (raw q, pre-2026-07-11 behaviour except for
 * the full-support completion). */
void mcgs_set_completion_n0(MCGSInstance *inst, float n0) {
    if (inst) inst->completion_n0 = (n0 < 0.0f) ? 0.0f : n0;
}

/* Expose the halfmove clock as obs channel 3 (clock/100).  Off by default:
 * legacy nets were trained with ch3 = 0 and must keep seeing zeros. */
void mcgs_set_clock_obs(MCGSInstance *inst, int enable) {
    if (inst) inst->clock_in_obs = enable ? 1 : 0;
}

/* Sequential Halving winner of a finished search (-1 if the search ended
 * degenerately: terminal/pass-only root).  This is the action the search
 * budget certified - play it at temperature 0. */
int mcgs_get_best_action(MCGSSearchState *ss) {
    return (ss && ss->phase == PHASE_DONE) ? ss->best_action : -1;
}

void mcgs_destroy(MCGSInstance *inst) {
    if (!inst) return;
    free(inst->ht);
    free(inst->node_slab);
    free(inst->edge_slab);
    free(inst);
}

int mcgs_tt_size(MCGSInstance *inst) {
    return inst ? inst->node_used : 0;
}

/* Edge-slab high-water mark.  Edges dominate the arena footprint (16 B each,
 * ~70 per node), so slab sizing has to be driven by this, not node count. */
int mcgs_edge_used(MCGSInstance *inst) {
    return inst ? inst->edge_used : 0;
}

/*  Search lifecycle  */

MCGSSearchState *mcgs_start_search(MCGSInstance *inst,
                                   bool py_board[7][7][2], bool turn,
                                   int clock) {
    MCGSSearchState *ss = (MCGSSearchState *)calloc(1, sizeof(MCGSSearchState));
    if (!ss) return NULL;
    ss->inst = inst;
    ss->best_action = -1;
    ss->root_clock = (clock < 0) ? 0 : clock;

    uint64_t green_bb, blue_bb;
    convert_board(py_board, &green_bb, &blue_bb);
    ss->root = tt_get_or_create(inst, green_bb, blue_bb, turn);
    if (!ss->root) { free(ss); return NULL; }

    MCGSNode *root = ss->root;

    if (root->is_terminal || ss->root_clock >= CLOCK_LIMIT) {
        memset(ss->result, 0, sizeof(ss->result));
        ss->phase = PHASE_DONE;
    } else if (root->is_expanded) {
        /* Check for pass-only root */
        int real_legal = 0;
        for (int i = 0; i < root->num_edges; i++)
            if (root->edges[i].action != (int16_t)PASS_ACTION) real_legal++;
        if (real_legal == 0) {
            memset(ss->result, 0, sizeof(ss->result));
            ss->phase = PHASE_DONE;
        } else {
            ss->phase = PHASE_HALVING;
            setup_halving(ss);
        }
    } else {
        ss->phase         = PHASE_ROOT_EXPAND;
        ss->pending[0]    = root;
        ss->pending_clock[0] = (uint8_t)(ss->root_clock < 255 ? ss->root_clock : 255);
        ss->num_pending   = 1;
    }
    return ss;
}

void mcgs_search_destroy(MCGSSearchState *ss) {
    free(ss);
}

int mcgs_pending_count(MCGSSearchState *ss) {
    return ss ? ss->num_pending : 0;
}

int mcgs_is_done(MCGSSearchState *ss) {
    return (!ss || ss->phase == PHASE_DONE) ? 1 : 0;
}

void mcgs_get_leaf_board(MCGSSearchState *ss, int i, bool out[7][7][2]) {
    if (!ss || i < 0 || i >= ss->num_pending) return;
    export_board(ss->pending[i]->green_bb, ss->pending[i]->blue_bb, out);
}

bool mcgs_get_leaf_turn(MCGSSearchState *ss, int i) {
    if (!ss || i < 0 || i >= ss->num_pending) return false;
    return ss->pending[i]->turn;
}

/*
 * Commit network expansion for the i-th pending leaf.
 *   policy[1225]: softmax policy probabilities (we filter to legal moves)
 *   value:        network value output (current mover's perspective)
 */
void mcgs_commit_expansion(MCGSSearchState *ss, int i,
                            float policy[1225], float value) {
    if (!ss || i < 0 || i >= ss->num_pending) return;
    MCGSNode *node = ss->pending[i];
    if (node->is_expanded || node->is_terminal) return;

    uint64_t mover_bb = node->turn ? node->blue_bb : node->green_bb;
    uint64_t occupied = node->green_bb | node->blue_bb;

    int16_t legal_actions[1225];
    int n_legal = gen_actions(mover_bb, occupied, legal_actions);

    MCGSInstance *inst = ss->inst;

    if (n_legal == 0) {
        /* Forced pass - no real legal moves */
        if (inst->edge_used + 1 > inst->edge_cap) {
            fprintf(stderr, "[mcgs] edge slab full (%d edges, %d nodes) - forced-pass expansion dropped\n",
                    inst->edge_cap, inst->node_used);
            return;
        }
        node->edges             = &inst->edge_slab[inst->edge_used++];
        node->edges[0].action   = (int16_t)PASS_ACTION;
        node->edges[0].prior    = 1.0f;
        node->edges[0].child    = NULL;
        node->num_edges         = 1;
        /* Keep the net's evaluation: a mover forced to pass is usually in a
         * decided position, and zeroing this injected fake draw backups into
         * exactly the endgame lines where value is most certain. */
        node->network_value     = value;
        node->is_expanded       = true;
        return;
    }

    /* Gather and renormalise policy mass over the legal moves */
    float priors[1225];
    float total = 0.0f;
    for (int j = 0; j < n_legal; j++) {
        float p   = policy[legal_actions[j]];
        priors[j] = (p > 0.0f) ? p : 0.0f;
        total    += priors[j];
    }

    if (total > 0.0f)
        for (int j = 0; j < n_legal; j++) priors[j] /= total;
    else {
        float u = 1.0f / (float)n_legal;
        for (int j = 0; j < n_legal; j++) priors[j] = u;
    }

    if (inst->edge_used + n_legal > inst->edge_cap) {
        fprintf(stderr, "[mcgs] edge slab full (%d edges, %d nodes) - normal expansion dropped (%d legal moves)\n",
                inst->edge_cap, inst->node_used, n_legal);
        return;
    }
    node->edges     = &inst->edge_slab[inst->edge_used];
    inst->edge_used += n_legal;
    node->num_edges  = n_legal;
    for (int j = 0; j < n_legal; j++) {
        node->edges[j].action = (int16_t)legal_actions[j];
        node->edges[j].prior  = priors[j];
        node->edges[j].child  = NULL;
    }
    node->network_value = value;
    node->is_expanded   = true;
}

/*
 * Advance the search by one batch step.
 * Caller must call mcgs_commit_expansion for all pending leaves first.
 * Returns updated pending count (0 means done or another step needed to find leaves).
 */
int mcgs_step(MCGSSearchState *ss) {
    if (!ss || ss->phase == PHASE_DONE) return 0;

    if (ss->phase == PHASE_ROOT_EXPAND) {
        MCGSNode *root = ss->root;
        if (root->is_terminal || !root->is_expanded) {
            memset(ss->result, 0, sizeof(ss->result));
            ss->phase = PHASE_DONE; ss->num_pending = 0;
            return 0;
        }
        int real_legal = 0;
        for (int i = 0; i < root->num_edges; i++)
            if (root->edges[i].action != (int16_t)PASS_ACTION) real_legal++;
        if (real_legal == 0) {
            memset(ss->result, 0, sizeof(ss->result));
            ss->phase = PHASE_DONE; ss->num_pending = 0;
            return 0;
        }
        if (!setup_halving(ss)) return 0;
        return ss->num_pending;
    }

    if (ss->phase == PHASE_HALVING) {
        for (int i = 0; i < ss->num_paths; i++) backprop_path(&ss->paths[i]);
        ss->sims_done += ss->num_active;
        ss->n_r--;
        if (ss->n_r <= 0 || ss->sims_done >= ss->N) {
            halving_prune(ss);
            if (ss->num_active <= 1 || ss->sims_done >= ss->N) {
                ss->phase = PHASE_TAIL;
                select_paths_tail(ss);
            } else {
                int n_r = ss->N / (ss->R * ss->num_active);
                if (n_r < 1) n_r = 1;
                ss->n_r = n_r;
                select_paths_halving(ss);
            }
        } else {
            select_paths_halving(ss);
        }
        return ss->num_pending;
    }

    if (ss->phase == PHASE_TAIL) {
        for (int i = 0; i < ss->num_paths; i++) backprop_path(&ss->paths[i]);
        ss->sims_done += ss->num_paths;
        select_paths_tail(ss);
        return ss->num_pending;
    }

    return 0;  /* PHASE_DONE */
}

void mcgs_get_result(MCGSSearchState *ss, float out[1225]) {
    if (ss) memcpy(out, ss->result, 1225 * sizeof(float));
}

float mcgs_get_root_value(MCGSSearchState *ss) {
    if (!ss || !ss->root || ss->root->visit_count == 0) return 0.0f;
    return ss->root->value_sum / (float)ss->root->visit_count;
}

/* Batch helpers - replace per-leaf ctypes round-trips with one call per slot. */

/* Write all pending leaf boards/turns into caller-provided arrays.
 * boards_out: bool[n * 98]  (n boards of shape 7x7x2, flat)
 * turns_out:  bool[n]
 * Returns n (same as mcgs_pending_count). */
int mcgs_get_pending_boards(MCGSSearchState *ss, bool *boards_out, bool *turns_out) {
    if (!ss) return 0;
    for (int i = 0; i < ss->num_pending; i++) {
        MCGSNode *node = ss->pending[i];
        export_board(node->green_bb, node->blue_bb,
                     (bool (*)[7][2])(boards_out + i * 98));
        turns_out[i] = node->turn;
    }
    return ss->num_pending;
}

/* Write all pending leaves straight into the network's float32 observation
 * layout, with the current-player perspective flip already applied. This
 * folds board_to_obs + the np.where channel swap that Python used to do into
 * the same 49-square loop that would otherwise just unpack the bitboards, so
 * the batch-prep step never materialises the intermediate bool boards.
 *
 * obs_out: float[n * 196], row-major (leaf, y, x, channel) with 4 channels:
 *   ch0 = opponent pieces, ch1 = my pieces (current player),
 *   ch2 = 1.0 (constant),
 *   ch3 = halfmove clock / 100 when clock_in_obs is set, else 0.0
 * matching lib/t7g.board_to_obs exactly.
 * Returns n (same as mcgs_pending_count). */
int mcgs_get_pending_obs(MCGSSearchState *ss, float *obs_out) {
    if (!ss) return 0;
    for (int i = 0; i < ss->num_pending; i++) {
        MCGSNode *node = ss->pending[i];
        float ch3 = ss->inst->clock_in_obs
                  ? (float)ss->pending_clock[i] / (float)CLOCK_LIMIT
                  : 0.0f;
        /* turn == Blue-to-move: mine = blue, opponent = green. */
        uint64_t mine_bb = node->turn ? node->blue_bb  : node->green_bb;
        uint64_t opp_bb  = node->turn ? node->green_bb : node->blue_bb;
        float *o = obs_out + i * 196;
        for (int pos = 0; pos < 49; pos++) {
            o[pos * 4 + 0] = (float)((opp_bb  >> pos) & 1);
            o[pos * 4 + 1] = (float)((mine_bb >> pos) & 1);
            o[pos * 4 + 2] = 1.0f;
            o[pos * 4 + 3] = ch3;
        }
    }
    return ss->num_pending;
}

/* Commit results for all n pending leaves in one call.
 * policies_flat: float[n * 1225]  (row-major, C order)
 * values:        float[n] */
void mcgs_commit_batch(MCGSSearchState *ss, float *policies_flat, float *values, int n) {
    if (!ss) return;
    int lim = (n < ss->num_pending) ? n : ss->num_pending;
    for (int i = 0; i < lim; i++)
        mcgs_commit_expansion(ss, i, policies_flat + i * 1225, values[i]);
}

/* Multi-search batch API - one ctypes call per pool GROUP instead of one per
 * slot.  All take an array of MCGSSearchState* (NULL entries allowed; they
 * behave like the corresponding single-search call on NULL). */

/* Step every search once and record whether it finished. */
void mcgs_step_many(MCGSSearchState **ss_arr, int n, int32_t *done_out) {
    for (int i = 0; i < n; i++) {
        mcgs_step(ss_arr[i]);
        done_out[i] = mcgs_is_done(ss_arr[i]);
    }
}

/* Pending-leaf counts for n searches. */
void mcgs_pending_counts(MCGSSearchState **ss_arr, int n, int32_t *counts_out) {
    for (int i = 0; i < n; i++)
        counts_out[i] = mcgs_pending_count(ss_arr[i]);
}

/* Concatenated pending obs for n searches, rows packed in search order
 * (searches with zero pending leaves contribute nothing).  Returns total rows. */
int mcgs_get_pending_obs_many(MCGSSearchState **ss_arr, int n, float *obs_out) {
    int total = 0;
    for (int i = 0; i < n; i++)
        total += mcgs_get_pending_obs(ss_arr[i], obs_out + (size_t)total * 196);
    return total;
}

/* Commit one flat (policies, values) slab back across n searches; counts[]
 * must be the per-search row counts the slab was assembled with. */
void mcgs_commit_batch_many(MCGSSearchState **ss_arr, const int32_t *counts, int n,
                            float *policies_flat, float *values) {
    size_t off = 0;
    for (int i = 0; i < n; i++) {
        int c = counts[i];
        if (c > 0)
            mcgs_commit_batch(ss_arr[i], policies_flat + off * 1225, values + off, c);
        off += (size_t)(c > 0 ? c : 0);
    }
}

/*
 * micro_mcts.c — C implementation of Gumbel MCGS for Microscope (T7G).
 *
 * Manages the MCTS tree and transposition table in C heap memory so that
 * free() actually returns pages to the OS, eliminating the Python arena
 * allocator leak that accumulates ~8 MB per game at training scale.
 *
 * Network inference stays in Python.  The step-wise interface lets the Python
 * game pool batch leaf nodes across all concurrent games into one GPU pass.
 *
 * Public API:
 *   mcgs_init()               — init Zobrist tables; call once at startup
 *   mcgs_create(sims,c,k)     — allocate a new MCGS instance
 *   mcgs_clear(inst)          — free all nodes, keep instance alive
 *   mcgs_destroy(inst)        — free instance and all its nodes
 *   mcgs_tt_size(inst)        — transposition table node count (monitoring)
 *   mcgs_start_search(…)      — begin a search, returns MCGSSearchState*
 *   mcgs_search_destroy(ss)   — free the search state struct
 *   mcgs_pending_count(ss)    — number of leaves needing network expansion
 *   mcgs_get_leaf_board(ss,i) — copy i-th pending leaf board (bool[7][7][2])
 *   mcgs_get_leaf_turn(ss,i)  — turn of i-th pending leaf
 *   mcgs_commit_expansion(…)  — feed policy[1225] + value for i-th leaf
 *   mcgs_step(ss)             — backprop + advance; returns new pending count
 *   mcgs_is_done(ss)          — 1 when search is complete
 *   mcgs_get_result(ss, out)  — write action_probs[1225] to out
 */
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <stdio.h>

/* ─── Constants ──────────────────────────────────────────────────────────── */
#define BOARD_SIZE     7
#define EMPTY          0
#define GREEN          1
#define BLUE           2
#define PASS_ACTION    1225   /* sentinel: no legal moves but game continues */

#define MAX_K          64     /* max Gumbel top-K */
#define MAX_PATH_DEPTH 128    /* max depth of one simulation path           */
#define MAX_PENDING    64     /* max unique leaves per batch (≤ MAX_K)      */

/* Arena slab capacities — sized for one game at 250 sims × ~100 moves.
 * Peak observed TT ≈ 25k nodes; 30k + 1.5M edges gives a safety margin.
 * Per instance: ~27 MB  (node slab 2.9 MB + edge slab 24 MB + HT 0.5 MB). */
#define NODE_SLAB_CAP  110000
#define EDGE_SLAB_CAP  7700000
#define HT_BITS        18
#define HT_SIZE        (1 << HT_BITS)   /* 262144 slots — must exceed NODE_SLAB_CAP */
#define HT_MASK        (HT_SIZE - 1)

/* Search phase codes */
#define PHASE_ROOT_EXPAND  0
#define PHASE_HALVING      1
#define PHASE_TAIL         2
#define PHASE_DONE         3

/* ─── Data structures ────────────────────────────────────────────────────── */

typedef struct {
    int16_t           action;
    float             prior;
    struct MCGSNode_s *child;  /* NULL until first visited */
} MCGSEdge;

typedef struct MCGSNode_s {
    uint64_t  hash;            /* Zobrist hash — open-HT key             */
    uint8_t   board[7][7];     /* internal: EMPTY=0 / GREEN=1 / BLUE=2   */
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
} SimPath;

typedef struct {
    /* Open-addressed hash table (fixed size, O(1) clear via memset) */
    MCGSNode **ht;             /* HT_SIZE slots, NULL-terminated probing   */
    /* Arena slabs — bump-pointer allocation, reset on clear             */
    MCGSNode  *node_slab;
    int        node_used;
    MCGSEdge  *edge_slab;
    int        edge_used;
    /* Search parameters */
    int       num_simulations;
    float     c_puct;
    int       gumbel_k;
    uint64_t  rng;             /* per-instance xorshift64 RNG state        */
} MCGSInstance;

typedef struct {
    MCGSInstance *inst;
    MCGSNode     *root;

    /* Gumbel / halving state */
    float   gumbel[MAX_K];
    int16_t top_k[MAX_K];
    int     K, N, R;
    int16_t active[MAX_K];
    int     num_active;
    int     sims_done, n_r, round_idx;
    int     phase;

    /* Pending leaves (need expansion before next step) */
    MCGSNode *pending[MAX_PENDING];
    int       num_pending;

    /* Current batch of simulation paths (for backprop) */
    SimPath paths[MAX_K];
    int     num_paths;

    /* Completed result */
    float result[1225];
} MCGSSearchState;

/* ─── Zobrist / RNG ──────────────────────────────────────────────────────── */

static uint64_t zobrist_board[49][3];
static uint64_t zobrist_turn;
static bool     zobrist_ready = false;

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

static uint64_t compute_hash(uint8_t board[7][7], bool turn) {
    uint64_t h = 0;
    for (int pos = 0; pos < 49; pos++) {
        uint8_t cell = board[pos / 7][pos % 7];
        h ^= zobrist_board[pos][cell];
    }
    if (turn) h ^= zobrist_turn;
    return h;
}

/* ─── Board utilities ────────────────────────────────────────────────────── */

/* Python bool[7][7][2] (ch0=green, ch1=blue) → internal uint8_t[7][7] */
static void convert_board(bool py[7][7][2], uint8_t out[7][7]) {
    for (int y = 0; y < 7; y++)
        for (int x = 0; x < 7; x++)
            out[y][x] = py[y][x][1] ? BLUE
                      : py[y][x][0]  ? GREEN
                      :                EMPTY;
}

/* internal uint8_t[7][7] → Python bool[7][7][2] */
static void export_board(uint8_t board[7][7], bool out[7][7][2]) {
    for (int y = 0; y < 7; y++)
        for (int x = 0; x < 7; x++) {
            out[y][x][0] = (board[y][x] == GREEN);
            out[y][x][1] = (board[y][x] == BLUE);
        }
}

/* action = piece*25 + move; piece = y*7+x; move = v*5+u (0..4, delta -2..+2) */
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

static bool any_moves(bool vm[7][7][5][5]) {
    return memchr(vm, 1, sizeof(bool[7][7][5][5])) != NULL;
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

static int count_cells(uint8_t board[7][7], uint8_t player) {
    int n = 0;
    for (int y = 0; y < BOARD_SIZE; y++)
        for (int x = 0; x < BOARD_SIZE; x++)
            if (board[y][x] == player) n++;
    return n;
}

/*
 * Check terminal from current mover's perspective.
 * Returns true and sets *value (+1/-1/0) if terminal.
 * Matches Python check_terminal: only terminal when one side has 0 pieces,
 * or BOTH players simultaneously have no legal moves.
 */
static bool check_terminal(uint8_t board[7][7], bool turn, float *value) {
    int blue  = count_cells(board, BLUE);
    int green = count_cells(board, GREEN);
    if (blue == 0) {
        *value = turn ? -1.0f : 1.0f;
        return true;
    }
    if (green == 0) {
        *value = turn ? 1.0f : -1.0f;
        return true;
    }
    uint8_t mover = turn ? BLUE : GREEN;
    uint8_t other = turn ? GREEN : BLUE;
    bool vm_mover[7][7][5][5], vm_other[7][7][5][5];
    get_valid_moves(board, mover, vm_mover);
    if (any_moves(vm_mover)) return false;
    get_valid_moves(board, other, vm_other);
    if (any_moves(vm_other)) return false;
    /* Both stuck: terminal by count (from mover's perspective) */
    float raw = (blue > green) ? 1.0f : (green > blue) ? -1.0f : 0.0f;
    *value = turn ? raw : -raw;
    return true;
}

/* ─── Transposition table (open-addressing, arena-backed) ───────────────── */

static MCGSNode *tt_find(MCGSInstance *inst, uint64_t h) {
    uint32_t idx = (uint32_t)(h & HT_MASK);
    for (;;) {
        MCGSNode *n = inst->ht[idx];
        if (!n) return NULL;
        if (n->hash == h) return n;
        idx = (idx + 1) & HT_MASK;
    }
}

static void tt_insert(MCGSInstance *inst, MCGSNode *n) {
    uint32_t idx = (uint32_t)(n->hash & HT_MASK);
    while (inst->ht[idx]) idx = (idx + 1) & HT_MASK;
    inst->ht[idx] = n;
}

/* O(1) clear: zero the HT bucket array and reset bump pointers. */
static void tt_clear(MCGSInstance *inst) {
    memset(inst->ht, 0, HT_SIZE * sizeof(MCGSNode *));
    inst->node_used = 0;
    inst->edge_used = 0;
}

/* Get or create node for (board, turn), inserting into TT if new. */
static MCGSNode *tt_get_or_create(MCGSInstance *inst, uint8_t board[7][7], bool turn) {
    uint64_t h = compute_hash(board, turn);
    MCGSNode *n = tt_find(inst, h);
    if (n) return n;
    if (inst->node_used >= NODE_SLAB_CAP) {
        fprintf(stderr, "[mcgs] node slab full (%d nodes) — mcgs_start_search will return NULL\n",
                NODE_SLAB_CAP);
        return NULL;
    }
    n = &inst->node_slab[inst->node_used++];
    memset(n, 0, sizeof(MCGSNode));
    n->hash = h;
    memcpy(n->board, board, sizeof(n->board));
    n->turn = turn;
    n->is_terminal = check_terminal(board, turn, &n->terminal_value);
    tt_insert(inst, n);
    return n;
}

/* ─── PUCT selection helpers ─────────────────────────────────────────────── */

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
            uint8_t cb[7][7]; bool ct;
            if (action == PASS_ACTION) {
                memcpy(cb, node->board, sizeof(cb));
                ct = !node->turn;
            } else {
                memcpy(cb, node->board, sizeof(cb));
                make_move(cb, action, node->turn ? BLUE : GREEN);
                ct = !node->turn;
            }
            e->child = tt_get_or_create(inst, cb, ct);
        }
        return e->child;
    }
    return NULL;
}

/* Select one simulation path from root, forcing first_action at the root. */
static void select_path(MCGSInstance *inst, MCGSNode *root,
                        int first_action, SimPath *path) {
    path->len      = 0;
    path->is_cycle = false;

    uint64_t visited[MAX_PATH_DEPTH];
    int      nv = 0;

    MCGSNode *node = root;
    path->nodes[path->len++] = node;
    visited[nv++] = node->hash;

    int forced = first_action;
    while (node->is_expanded && !node->is_terminal) {
        int action = (forced >= 0) ? forced : best_action_puct(node, inst->c_puct);
        forced = -1;
        if (action < 0) break;

        MCGSNode *child = get_or_create_child(inst, node, action);
        if (!child) break;

        /* Cycle detection */
        for (int i = 0; i < nv; i++) {
            if (visited[i] == child->hash) {
                path->nodes[path->len++] = child;
                path->is_cycle = true;
                return;
            }
        }
        path->nodes[path->len++] = child;
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
    if (path->is_cycle) {
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

/* ─── Gumbel / halving helpers ───────────────────────────────────────────── */

typedef struct { int16_t action; float score; } SortEntry;
static int cmp_score_desc(const void *a, const void *b) {
    float fa = ((SortEntry *)a)->score, fb = ((SortEntry *)b)->score;
    return (fb > fa) - (fb < fa);
}

/* Finish: compute completed-Q policy and mark PHASE_DONE. */
static void finish_search(MCGSSearchState *ss) {
    MCGSNode *root = ss->root;
    float best_logit = -1e30f;
    float logits[MAX_K];
    for (int i = 0; i < ss->K; i++) {
        float q    = action_q(root, (int)ss->top_k[i], root->network_value);
        logits[i]  = ss->gumbel[i] + q;
        if (logits[i] > best_logit) best_logit = logits[i];
    }
    float probs[MAX_K], sum = 0.0f;
    for (int i = 0; i < ss->K; i++) {
        probs[i] = expf(logits[i] - best_logit);
        sum += probs[i];
    }
    memset(ss->result, 0, 1225 * sizeof(float));
    for (int i = 0; i < ss->K; i++)
        ss->result[ss->top_k[i]] = probs[i] / sum;
    ss->phase       = PHASE_DONE;
    ss->num_pending = 0;
    ss->num_paths   = 0;
}

/* Collect unique unexpanded non-terminal leaves from ss->paths. */
static void collect_pending(MCGSSearchState *ss) {
    ss->num_pending = 0;
    for (int p = 0; p < ss->num_paths && ss->num_pending < MAX_PENDING; p++) {
        SimPath *path = &ss->paths[p];
        if (path->is_cycle || path->len == 0) continue;
        MCGSNode *leaf = path->nodes[path->len - 1];
        if (leaf->is_terminal || leaf->is_expanded) continue;
        bool dup = false;
        for (int j = 0; j < ss->num_pending; j++)
            if (ss->pending[j] == leaf) { dup = true; break; }
        if (!dup) ss->pending[ss->num_pending++] = leaf;
    }
}

static void select_paths_halving(MCGSSearchState *ss) {
    ss->num_paths = ss->num_active;
    for (int i = 0; i < ss->num_active; i++)
        select_path(ss->inst, ss->root, (int)ss->active[i], &ss->paths[i]);
    collect_pending(ss);
}

static void select_paths_tail(MCGSSearchState *ss) {
    if (ss->sims_done >= ss->N) { finish_search(ss); return; }
    int batch = ss->K < (ss->N - ss->sims_done) ? ss->K : (ss->N - ss->sims_done);
    ss->num_paths = batch;
    for (int i = 0; i < batch; i++)
        select_path(ss->inst, ss->root, (int)ss->active[0], &ss->paths[i]);
    collect_pending(ss);
}

static void halving_prune(MCGSSearchState *ss) {
    SortEntry ranked[MAX_K];
    MCGSNode *root = ss->root;
    for (int i = 0; i < ss->num_active; i++) {
        ranked[i].action = ss->active[i];
        ranked[i].score  = action_q(root, (int)ss->active[i], root->network_value);
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
        legal[n_legal].action = root->edges[i].action;
        legal[n_legal].score  = lp + gumbel_noise(&ss->inst->rng);
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
        ss->top_k[i]  = legal[i].action;
        ss->gumbel[i] = legal[i].score;
        ss->active[i] = legal[i].action;
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

/* ─── Public API ─────────────────────────────────────────────────────────── */

void mcgs_init(void) {
    init_zobrist();
}

MCGSInstance *mcgs_create(int num_simulations, float c_puct, int gumbel_k) {
    init_zobrist();
    MCGSInstance *inst = (MCGSInstance *)calloc(1, sizeof(MCGSInstance));
    if (!inst) return NULL;
    inst->ht        = (MCGSNode **)calloc(HT_SIZE, sizeof(MCGSNode *));
    inst->node_slab = (MCGSNode  *)malloc(NODE_SLAB_CAP * sizeof(MCGSNode));
    inst->edge_slab = (MCGSEdge  *)malloc(EDGE_SLAB_CAP * sizeof(MCGSEdge));
    if (!inst->ht || !inst->node_slab || !inst->edge_slab) {
        free(inst->ht); free(inst->node_slab); free(inst->edge_slab); free(inst);
        return NULL;
    }
    inst->num_simulations = num_simulations;
    inst->c_puct          = c_puct;
    inst->gumbel_k        = gumbel_k;
    inst->rng             = (uint64_t)time(NULL) ^ (uint64_t)(uintptr_t)inst;
    if (inst->rng == 0) inst->rng = 0xDEADC0DEULL;
    return inst;
}

void mcgs_clear(MCGSInstance *inst) {
    if (inst) tt_clear(inst);
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

/* ─── Search lifecycle ───────────────────────────────────────────────────── */

MCGSSearchState *mcgs_start_search(MCGSInstance *inst,
                                   bool py_board[7][7][2], bool turn) {
    MCGSSearchState *ss = (MCGSSearchState *)calloc(1, sizeof(MCGSSearchState));
    if (!ss) return NULL;
    ss->inst = inst;

    uint8_t board[7][7];
    convert_board(py_board, board);
    ss->root = tt_get_or_create(inst, board, turn);
    if (!ss->root) { free(ss); return NULL; }

    MCGSNode *root = ss->root;

    if (root->is_terminal) {
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
        ss->phase       = PHASE_ROOT_EXPAND;
        ss->pending[0]  = root;
        ss->num_pending = 1;
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
    export_board(ss->pending[i]->board, out);
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

    uint8_t player = node->turn ? BLUE : GREEN;
    bool valid_moves[7][7][5][5];
    get_valid_moves(node->board, player, valid_moves);
    bool *vm = (bool *)valid_moves;

    MCGSInstance *inst = ss->inst;

    if (!any_moves(valid_moves)) {
        /* Forced pass — no real legal moves */
        if (inst->edge_used + 1 > EDGE_SLAB_CAP) {
            fprintf(stderr, "[mcgs] edge slab full (%d edges, %d nodes) — forced-pass expansion dropped\n",
                    EDGE_SLAB_CAP, inst->node_used);
            return;
        }
        node->edges             = &inst->edge_slab[inst->edge_used++];
        node->edges[0].action   = (int16_t)PASS_ACTION;
        node->edges[0].prior    = 1.0f;
        node->edges[0].child    = NULL;
        node->num_edges         = 1;
        node->network_value     = 0.0f;
        node->is_expanded       = true;
        return;
    }

    /* Count legal moves and sum policy mass over them */
    int   legal_actions[1225];
    float priors[1225];
    int   n_legal = 0;
    float total   = 0.0f;
    for (int a = 0; a < 1225; a++) {
        if (!vm[a]) continue;
        legal_actions[n_legal] = a;
        priors[n_legal]        = (policy[a] > 0.0f) ? policy[a] : 0.0f;
        total                 += priors[n_legal];
        n_legal++;
    }

    if (total > 0.0f)
        for (int j = 0; j < n_legal; j++) priors[j] /= total;
    else {
        float u = 1.0f / (float)n_legal;
        for (int j = 0; j < n_legal; j++) priors[j] = u;
    }

    if (n_legal <= 0) return;
    if (inst->edge_used + n_legal > EDGE_SLAB_CAP) {
        fprintf(stderr, "[mcgs] edge slab full (%d edges, %d nodes) — normal expansion dropped (%d legal moves)\n",
                EDGE_SLAB_CAP, inst->node_used, n_legal);
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
    MCGSInstance *inst = ss->inst;

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

/* Batch helpers — replace per-leaf ctypes round-trips with one call per slot. */

/* Write all pending leaf boards/turns into caller-provided arrays.
 * boards_out: bool[n * 98]  (n boards of shape 7x7x2, flat)
 * turns_out:  bool[n]
 * Returns n (same as mcgs_pending_count). */
int mcgs_get_pending_boards(MCGSSearchState *ss, bool *boards_out, bool *turns_out) {
    if (!ss) return 0;
    for (int i = 0; i < ss->num_pending; i++) {
        MCGSNode *node = ss->pending[i];
        export_board(node->board, (bool (*)[7][2])(boards_out + i * 98));
        turns_out[i] = node->turn;
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

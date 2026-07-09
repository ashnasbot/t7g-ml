/*
 * micro_mcts_heuristic.c - MCGS with heuristic leaf evaluation (no neural network).
 *
 * Identical tree mechanics to micro_mcts.c (Gumbel top-K, Sequential Halving,
 * PUCT, Zobrist TT, arena slabs).  The Python NN interface is replaced by:
 *
 *   Policy priors : capture-weighted - moves that flip more pieces get
 *                   proportionally higher prior (1 + flips), normalised.
 *   Leaf value    : material fraction  (player - opponent) / total pieces
 *                   from the current mover's perspective, in (-1, +1).
 *                   Small centrality bonus added (weight 0.05) to break ties.
 *
 * Public API:
 *   hmcts_init()
 *   hmcts_create(sims, c_puct, gumbel_k)  -> void*
 *   hmcts_clear(inst)
 *   hmcts_destroy(inst)
 *   hmcts_find_best_action(inst, py_board, turn)  -> int  (full search, synchronous)
 *
 * DLL entry point (stateless, matches minimax find_best_move style):
 *   hmcts_find_best_move(board_bytes, sims, turn)  -> int
 */
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

/*  Constants  */
#define BOARD_SIZE     7
#define EMPTY          0
#define GREEN          1
#define BLUE           2
#define PASS_ACTION    1225

#define MAX_K          64
#define MAX_PATH_DEPTH 128
#define MAX_PENDING    64

/* Arena slab capacities - sized for ~20k sims per search (fresh instance per call).
 * At 20k sims, ~10k unique node expansions × ~150 legal moves/node ≈ 1.5M edges peak.
 * Use generous headroom: 150k nodes, 10M edges, HT_BITS=18 (262144 > 150k). */
#define NODE_SLAB_CAP  150000
#define EDGE_SLAB_CAP  10000000
#define HT_BITS        18
#define HT_SIZE        (1 << HT_BITS)   /* 262144 slots - must exceed NODE_SLAB_CAP */
#define HT_MASK        (HT_SIZE - 1)

#define PHASE_ROOT_EXPAND  0
#define PHASE_HALVING      1
#define PHASE_TAIL         2
#define PHASE_DONE         3

/*  Data structures  */

typedef struct {
    int16_t           action;
    float             prior;
    struct MCGSNode_s *child;
} MCGSEdge;

typedef struct MCGSNode_s {
    uint64_t  hash;
    uint8_t   board[7][7];
    bool      turn;
    int       visit_count;
    float     value_sum;
    float     network_value;   /* heuristic value at expansion time */
    bool      is_expanded;
    bool      is_terminal;
    float     terminal_value;
    int       num_edges;
    MCGSEdge *edges;
} MCGSNode;

typedef struct {
    MCGSNode *nodes[MAX_PATH_DEPTH];
    int       len;
    bool      is_cycle;
} SimPath;

typedef struct {
    MCGSNode **ht;
    MCGSNode  *node_slab;
    int        node_used;
    MCGSEdge  *edge_slab;
    int        edge_used;
    int        num_simulations;
    float      c_puct;
    int        gumbel_k;
    uint64_t   rng;
} MCGSInstance;

typedef struct {
    MCGSInstance *inst;
    MCGSNode     *root;
    float   gumbel[MAX_K];
    float   log_prior[MAX_K];
    int16_t top_k[MAX_K];
    int     K, N, R;
    int16_t active[MAX_K];
    int     num_active;
    int     sims_done, n_r, round_idx;
    int     phase;
    MCGSNode *pending[MAX_PENDING];
    int       num_pending;
    SimPath paths[MAX_K];
    int     num_paths;
    float   result[1225];
} MCGSSearchState;

/*  Zobrist / RNG  */

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

/*  Board utilities  */

static void convert_board(bool py[7][7][2], uint8_t out[7][7]) {
    for (int y = 0; y < 7; y++)
        for (int x = 0; x < 7; x++)
            out[y][x] = py[y][x][1] ? BLUE
                      : py[y][x][0]  ? GREEN
                      :                EMPTY;
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

static bool check_terminal(uint8_t board[7][7], bool turn, float *value) {
    int blue  = count_cells(board, BLUE);
    int green = count_cells(board, GREEN);
    if (blue == 0) { *value = turn ? -1.0f : 1.0f; return true; }
    if (green == 0) { *value = turn ? 1.0f : -1.0f; return true; }
    uint8_t mover = turn ? BLUE : GREEN;
    uint8_t other = turn ? GREEN : BLUE;
    bool vm_mover[7][7][5][5], vm_other[7][7][5][5];
    get_valid_moves(board, mover, vm_mover);
    if (any_moves(vm_mover)) return false;
    get_valid_moves(board, other, vm_other);
    if (any_moves(vm_other)) return false;
    float raw = (blue > green) ? 1.0f : (green > blue) ? -1.0f : 0.0f;
    *value = turn ? raw : -raw;
    return true;
}

/*  Transposition table  */

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

static void tt_clear(MCGSInstance *inst) {
    memset(inst->ht, 0, HT_SIZE * sizeof(MCGSNode *));
    inst->node_used = 0;
    inst->edge_used = 0;
}

static MCGSNode *tt_get_or_create(MCGSInstance *inst, uint8_t board[7][7], bool turn) {
    uint64_t h = compute_hash(board, turn);
    MCGSNode *n = tt_find(inst, h);
    if (n) return n;
    if (inst->node_used >= NODE_SLAB_CAP) return NULL;
    n = &inst->node_slab[inst->node_used++];
    memset(n, 0, sizeof(MCGSNode));
    n->hash = h;
    memcpy(n->board, board, sizeof(n->board));
    n->turn = turn;
    n->is_terminal = check_terminal(board, turn, &n->terminal_value);
    tt_insert(inst, n);
    return n;
}

/*  PUCT selection  */

static float node_q(MCGSNode *n) {
    return (n->visit_count > 0) ? n->value_sum / (float)n->visit_count : 0.0f;
}

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
        float q = (c == NULL) ? 0.0f : -node_q(c);
        int   v = (c == NULL) ? 0    : c->visit_count;
        float s = q + c_puct * e->prior * sqrt_n / (1.0f + v);
        if (s > best) { best = s; best_action = (int)e->action; }
    }
    return best_action;
}

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

/*  Path selection and backprop  */

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

static void backprop_path(SimPath *path) {
    if (path->len == 0) return;
    MCGSNode *leaf = path->nodes[path->len - 1];
    float value;
    if (path->is_cycle)        value = 0.0f;
    else if (leaf->is_terminal) value = leaf->terminal_value;
    else                        value = leaf->network_value;
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

static void finish_search(MCGSSearchState *ss) {
    MCGSNode *root = ss->root;
    float best_logit = -1e30f;
    float logits[MAX_K];
    for (int i = 0; i < ss->K; i++) {
        float q   = action_q(root, (int)ss->top_k[i], root->network_value);
        logits[i] = ss->log_prior[i] + q;
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
    int new_n = ss->num_active / 2;
    if (new_n < 1) new_n = 1;
    ss->num_active = new_n;
    for (int i = 0; i < new_n; i++) ss->active[i] = ranked[i].action;
    ss->round_idx++;
}

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
        ss->top_k[i]     = legal[i].action;
        ss->gumbel[i]    = legal[i].score;
        ss->log_prior[i] = legal[i].log_prior;
        ss->active[i]    = legal[i].action;
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

/*  Heuristic leaf evaluation  */


/*
 * Count opponent pieces that the player can capture in a single clone move.
 * Mirrors micro_4.c / micro_3.c implementation.
 */
static int conversion_threat_h(uint8_t board[7][7], uint8_t player, uint8_t opponent) {
    bool reachable[7][7] = {0};
    for (int y = 0; y < BOARD_SIZE; y++)
        for (int x = 0; x < BOARD_SIZE; x++) {
            if (board[y][x] != player) continue;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) {
                    if (dy == 0 && dx == 0) continue;
                    int ny = y + dy, nx = x + dx;
                    if (ny < 0 || ny >= BOARD_SIZE || nx < 0 || nx >= BOARD_SIZE) continue;
                    if (board[ny][nx] == EMPTY) reachable[ny][nx] = true;
                }
        }
    int threats = 0;
    for (int y = 0; y < BOARD_SIZE; y++)
        for (int x = 0; x < BOARD_SIZE; x++) {
            if (board[y][x] != opponent) continue;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) {
                    if (dy == 0 && dx == 0) continue;
                    int ny = y + dy, nx = x + dx;
                    if (ny < 0 || ny >= BOARD_SIZE || nx < 0 || nx >= BOARD_SIZE) continue;
                    if (reachable[ny][nx]) { threats++; break; }
                }
        }
    return threats;
}

/* Count empty cells adjacent (dist-1) to any of this player's pieces. */
static int count_clone_squares_h(uint8_t board[7][7], uint8_t player) {
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

/*
 * Heuristic value for a leaf node from the current mover's perspective.
 * Mirrors micro_4.c get_score() then normalises to (-1, +1) via tanh(score/50).
 * The scale factor 50 is chosen so a ~5-piece material lead maps to ~tanh(0.5)≈0.46.
 */
static float heuristic_value(uint8_t board[7][7], bool turn) {
    uint8_t player   = turn ? BLUE : GREEN;
    uint8_t opponent = turn ? GREEN : BLUE;

    float score = (float)(count_cells(board, player) - count_cells(board, opponent)) * 10.0f;

    score += (float)(conversion_threat_h(board, player, opponent)
                   - conversion_threat_h(board, opponent, player)) * 1.0f;

    score += (float)(count_clone_squares_h(board, player)
                   - count_clone_squares_h(board, opponent)) * 0.3f;

    for (int y = 0; y < BOARD_SIZE; y++)
        for (int x = 0; x < BOARD_SIZE; x++) {
            float bonus = (6.0f - (float)(abs(x - 3) + abs(y - 3))) * 0.1f;
            if      (board[y][x] == player)   score += bonus;
            else if (board[y][x] == opponent) score -= bonus;
        }

    return tanhf(score / 50.0f);
}

/*
 * Expand a leaf node inline without a neural network:
 *   - Policy prior: proportional to (1 + captures) per move, normalised.
 *   - Value: heuristic_value().
 */
static void expand_node_inline(MCGSInstance *inst, MCGSNode *node) {
    if (node->is_expanded || node->is_terminal) return;

    uint8_t player = node->turn ? BLUE : GREEN;
    bool valid_moves[7][7][5][5];
    get_valid_moves(node->board, player, valid_moves);
    bool *vm = (bool *)valid_moves;

    if (!any_moves(valid_moves)) {
        if (inst->edge_used + 1 > EDGE_SLAB_CAP) return;
        node->edges           = &inst->edge_slab[inst->edge_used++];
        node->edges[0].action = (int16_t)PASS_ACTION;
        node->edges[0].prior  = 1.0f;
        node->edges[0].child  = NULL;
        node->num_edges       = 1;
        node->network_value   = 0.0f;
        node->is_expanded     = true;
        return;
    }

    int   legal_actions[1225];
    int   n_legal = 0;

    for (int a = 0; a < 1225; a++) {
        if (!vm[a]) continue;
        legal_actions[n_legal++] = a;
    }

    if (n_legal <= 0) return;
    if (inst->edge_used + n_legal > EDGE_SLAB_CAP) return;

    float uniform = 1.0f / (float)n_legal;
    node->edges     = &inst->edge_slab[inst->edge_used];
    inst->edge_used += n_legal;
    node->num_edges  = n_legal;
    for (int j = 0; j < n_legal; j++) {
        node->edges[j].action = (int16_t)legal_actions[j];
        node->edges[j].prior  = uniform;
        node->edges[j].child  = NULL;
    }
    node->network_value = heuristic_value(node->board, node->turn);
    node->is_expanded   = true;
}

/*  Synchronous step  */

/*
 * Expand all pending nodes inline, then advance the search phase.
 * This replaces the async Python round-trip of micro_mcts.c.
 */
static void hmcts_step(MCGSSearchState *ss) {
    if (!ss || ss->phase == PHASE_DONE) return;
    MCGSInstance *inst = ss->inst;

    /* Expand every pending leaf synchronously */
    for (int i = 0; i < ss->num_pending; i++)
        expand_node_inline(inst, ss->pending[i]);
    ss->num_pending = 0;

    if (ss->phase == PHASE_ROOT_EXPAND) {
        MCGSNode *root = ss->root;
        if (ss->root->is_terminal || !root->is_expanded) {
            memset(ss->result, 0, sizeof(ss->result));
            ss->phase = PHASE_DONE;
            return;
        }
        int real_legal = 0;
        for (int i = 0; i < root->num_edges; i++)
            if (root->edges[i].action != (int16_t)PASS_ACTION) real_legal++;
        if (real_legal == 0) {
            memset(ss->result, 0, sizeof(ss->result));
            ss->phase = PHASE_DONE;
            return;
        }
        setup_halving(ss);
        /* Collect new pending from first round of paths */
        for (int i = 0; i < ss->num_pending; i++)
            expand_node_inline(inst, ss->pending[i]);
        ss->num_pending = 0;
        return;
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
        for (int i = 0; i < ss->num_pending; i++)
            expand_node_inline(inst, ss->pending[i]);
        ss->num_pending = 0;
        return;
    }

    if (ss->phase == PHASE_TAIL) {
        for (int i = 0; i < ss->num_paths; i++) backprop_path(&ss->paths[i]);
        ss->sims_done += ss->num_paths;
        select_paths_tail(ss);
        for (int i = 0; i < ss->num_pending; i++)
            expand_node_inline(inst, ss->pending[i]);
        ss->num_pending = 0;
    }
}

/*  Public API  */

void hmcts_init(void) {
    init_zobrist();
}

void *hmcts_create(int num_simulations, float c_puct, int gumbel_k) {
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

void hmcts_clear(void *handle) {
    if (handle) tt_clear((MCGSInstance *)handle);
}

void hmcts_destroy(void *handle) {
    if (!handle) return;
    MCGSInstance *inst = (MCGSInstance *)handle;
    free(inst->ht);
    free(inst->node_slab);
    free(inst->edge_slab);
    free(inst);
}

/*
 * Run a full synchronous MCGS search and return the best action.
 * inst may be reused across moves (TT persists); call hmcts_clear between games.
 */
int hmcts_find_best_action(void *handle, bool py_board[7][7][2], bool turn) {
    MCGSInstance *inst = (MCGSInstance *)handle;

    MCGSSearchState *ss = (MCGSSearchState *)calloc(1, sizeof(MCGSSearchState));
    if (!ss) return PASS_ACTION;
    ss->inst = inst;

    uint8_t board[7][7];
    convert_board(py_board, board);
    ss->root = tt_get_or_create(inst, board, turn);
    if (!ss->root) { free(ss); return PASS_ACTION; }

    MCGSNode *root = ss->root;
    if (root->is_terminal) {
        free(ss); return PASS_ACTION;
    }
    if (root->is_expanded) {
        int real_legal = 0;
        for (int i = 0; i < root->num_edges; i++)
            if (root->edges[i].action != (int16_t)PASS_ACTION) real_legal++;
        if (real_legal == 0) { free(ss); return PASS_ACTION; }
        ss->phase = PHASE_ROOT_EXPAND;  /* setup_halving handles it */
    } else {
        ss->phase       = PHASE_ROOT_EXPAND;
        ss->pending[0]  = root;
        ss->num_pending = 1;
    }

    while (ss->phase != PHASE_DONE)
        hmcts_step(ss);

    /* Best action = argmax of completed-Q policy */
    int   best_action = PASS_ACTION;
    float best_prob   = -1.0f;
    for (int a = 0; a < 1225; a++) {
        if (ss->result[a] > best_prob) {
            best_prob   = ss->result[a];
            best_action = a;
        }
    }

    free(ss);
    return best_action;
}

/*
 * Stateless DLL entry point.
 * board_bytes: raw bytes of bool[7][7][2] (98 bytes, same layout as Python ndarray)
 * sims:        number of MCTS simulations
 * turn:        true = Blue to move
 */
int hmcts_find_best_move(const char *board_bytes, int sims, bool turn) {
    void *inst = hmcts_create(sims, 1.25f, 16);
    if (!inst) return PASS_ACTION;
    int action = hmcts_find_best_action(inst, (bool (*)[7][2])board_bytes, turn);
    hmcts_destroy(inst);
    return action;
}

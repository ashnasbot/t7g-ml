/*
* Optimized solver for the T7G 'microscope' puzzle:
* - Transposition table (hash table cache)
* - Zobrist hashing for fast position hashing
* - Move ordering (captures first, killer moves)
* - Killer move heuristic
*/
#include <stdbool.h>
#include <string.h>
#include <sys/types.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

#define BOARD_SIZE 7
#define GREEN_CELL 1
#define BLUE_CELL 2
#define MAX_DEPTH 20

// Transposition table settings
#define TT_SIZE (1 << 20)  // 1M entries (~32MB)
#define TT_MASK (TT_SIZE - 1)

bool BLUE[2] = {0, 1};
bool GREEN[2] = {1, 0};
bool CLEAR[2] = {0, 0};

// Transposition table entry
typedef struct {
    uint64_t hash;
    float score;
    int8_t depth;
    int8_t flag;  // 0=exact, 1=lower bound, 2=upper bound
    int16_t best_move;
} TTEntry;

// Killer moves (moves that caused beta cutoffs)
typedef struct {
    int16_t moves[MAX_DEPTH][2];  // Two killer moves per depth
} KillerMoves;

// Global transposition table
static TTEntry *tt_table = NULL;
static KillerMoves killer_moves;

// Zobrist hashing tables (for fast incremental hashing)
static uint64_t zobrist_table[49][3];  // [position][piece_type]
static bool zobrist_initialized = false;

// Simple PRNG for Zobrist initialization
static uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

// Initialize Zobrist hashing
void init_zobrist() {
    if (zobrist_initialized) return;

    uint64_t seed = 0x123456789ABCDEF0ULL;
    for (int i = 0; i < 49; i++) {
        for (int j = 0; j < 3; j++) {
            zobrist_table[i][j] = xorshift64(&seed);
        }
    }
    zobrist_initialized = true;
}

// Initialize transposition table
void init_tt() {
    if (tt_table == NULL) {
        tt_table = (TTEntry*)calloc(TT_SIZE, sizeof(TTEntry));
        init_zobrist();
        memset(&killer_moves, 0, sizeof(KillerMoves));
    }
}

// Free transposition table
void free_tt() {
    if (tt_table != NULL) {
        free(tt_table);
        tt_table = NULL;
    }
}

// Clear transposition table
void clear_tt() {
    if (tt_table != NULL) {
        memset(tt_table, 0, TT_SIZE * sizeof(TTEntry));
        memset(&killer_moves, 0, sizeof(KillerMoves));
    }
}

// Compute Zobrist hash for a board position
uint64_t compute_zobrist_hash(bool board[7][7][2]) {
    uint64_t hash = 0;
    for (int pos = 0; pos < 49; pos++) {
        int y = pos / 7;
        int x = pos % 7;

        int piece_type = 0;  // Empty
        if (board[y][x][0] && !board[y][x][1]) piece_type = 1;  // Green
        else if (!board[y][x][0] && board[y][x][1]) piece_type = 2;  // Blue

        if (piece_type > 0) {
            hash ^= zobrist_table[pos][piece_type];
        }
    }
    return hash;
}

// Probe transposition table
bool tt_probe(uint64_t hash, int depth, float alpha, float beta, float *score, int16_t *best_move) {
    TTEntry *entry = &tt_table[hash & TT_MASK];

    if (entry->hash != hash) return false;
    if (entry->depth < depth) return false;

    *best_move = entry->best_move;

    if (entry->flag == 0) {  // Exact score
        *score = entry->score;
        return true;
    } else if (entry->flag == 1) {  // Lower bound
        if (entry->score >= beta) {
            *score = entry->score;
            return true;
        }
    } else if (entry->flag == 2) {  // Upper bound
        if (entry->score <= alpha) {
            *score = entry->score;
            return true;
        }
    }

    return false;
}

// Store in transposition table
void tt_store(uint64_t hash, int depth, float score, int8_t flag, int16_t best_move) {
    TTEntry *entry = &tt_table[hash & TT_MASK];

    // Replace if deeper or same depth
    if (entry->hash != hash || entry->depth <= depth) {
        entry->hash = hash;
        entry->score = score;
        entry->depth = depth;
        entry->flag = flag;
        entry->best_move = best_move;
    }
}

// Store killer move
void store_killer_move(int depth, int16_t move) {
    if (depth >= MAX_DEPTH) return;

    // Shift moves
    if (killer_moves.moves[depth][0] != move) {
        killer_moves.moves[depth][1] = killer_moves.moves[depth][0];
        killer_moves.moves[depth][0] = move;
    }
}

// Check if move is a killer move
bool is_killer_move(int depth, int16_t move) {
    if (depth >= MAX_DEPTH) return false;
    return killer_moves.moves[depth][0] == move ||
           killer_moves.moves[depth][1] == move;
}


// Check if move captures opponent pieces
int count_captures(bool board[7][7][2], int action, bool colour[2]) {
    int piece = action / 25;
    int mv = action % 25;
    int from_x = piece % 7;
    int from_y = piece / 7;
    int mv_x = (mv % 5) - 2;
    int mv_y = (mv / 5) - 2;

    int to_x = from_x + mv_x;
    int to_y = from_y + mv_y;

    if (to_x < 0 || to_x >= 7 || to_y < 0 || to_y >= 7) return 0;

    int captures = 0;
    for (int u = 0; u < 3; u++) {
        for (int v = 0; v < 3; v++) {
            int c_x = to_x + u - 1;
            int c_y = to_y + v - 1;

            if (c_x >= 0 && c_x < 7 && c_y >= 0 && c_y < 7) {
                // Check if it's an opponent piece
                if ((board[c_y][c_x][0] != colour[0] || board[c_y][c_x][1] != colour[1]) &&
                    (board[c_y][c_x][0] != CLEAR[0] || board[c_y][c_x][1] != CLEAR[1])) {
                    captures++;
                }
            }
        }
    }
    return captures;
}

bool any_moves(bool valid_moves[7][7][5][5])
{
    for (int x = 0; x < 7; x++)
    {
        for (int y = 0; y < 7; y++)
        {
            for (int u = 0; u < 5; u++)
            {
                for (int v = 0; v < 5; v++)
                {
                    if(valid_moves[y][x][v][u] != 0) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

int count_cells(bool board[7][7][2], bool colour[2])
{
    int count = 0;
    for (int x = 0; x < BOARD_SIZE; x++)
    {
        for (int y = 0; y < BOARD_SIZE; y++)
        {
            if (board[y][x][0] == colour[0] &&
                board[y][x][1] == colour[1])
            {
                count++;
            }
        }
    }
    return count;
}

float get_score(bool board[7][7][2], bool colour[2])
{
    bool opponent[2];

    if (colour[0] == BLUE[0] &&
        colour[1] == BLUE[1])
    {
        opponent[0] = GREEN[0];
        opponent[1] = GREEN[1];
    } else {
        opponent[0] = BLUE[0];
        opponent[1] = BLUE[1];
    }

    int player_cells = count_cells(board, colour);
    int opponent_cells = count_cells(board, opponent);

    // Material advantage (primary factor, 10x weight)
    float material = (player_cells - opponent_cells) * 10.0f;

    // Center control bonus (secondary factor)
    // Center pieces have more mobility and capture potential
    float center_bonus = 0.0f;
    for (int x = 0; x < BOARD_SIZE; x++)
    {
        for (int y = 0; y < BOARD_SIZE; y++)
        {
            if (board[y][x][0] == colour[0] && board[y][x][1] == colour[1])
            {
                // Manhattan distance from center (3,3)
                int dx = abs(x - 3);
                int dy = abs(y - 3);
                float dist = dx + dy;
                // Closer to center = higher bonus (max 0.6 per piece)
                center_bonus += (6.0f - dist) * 0.1f;
            }
            else if (board[y][x][0] == opponent[0] && board[y][x][1] == opponent[1])
            {
                int dx = abs(x - 3);
                int dy = abs(y - 3);
                float dist = dx + dy;
                center_bonus -= (6.0f - dist) * 0.1f;
            }
        }
    }

    return material + center_bonus;
}

void get_valid_moves(bool board[7][7][2], bool colour[2], bool valid_moves[7][7][5][5])
{
    memset(valid_moves, 0, sizeof(bool[7][7][5][5]));
    bool *moves = (bool *)valid_moves;

    for (int x = 0; x < BOARD_SIZE; x++)
    {
        for (int y = 0; y < BOARD_SIZE; y++)
        {
            if (board[y][x][0] == colour[0] &&
                board[y][x][1] == colour[1])
            {
                for (int u = 0; u < 5; u++)
                {
                    for (int v = 0; v < 5; v++)
                    {
                        int to_x = x + u - 2;
                        int to_y = y + v - 2;

                        if ((0 <= to_x && to_x < 7) &&
                            (0 <= to_y && to_y < 7))
                        {
                            if (board[to_y][to_x][0] == CLEAR[0] &&
                                board[to_y][to_x][1] == CLEAR[1])
                            {
                                int moveid = (25 * ((7 * y) + x)) + (5 *v) + u;
                                moves[moveid] = true;
                            }
                        }
                    }
                }
            }
        }
    }
}

void move(bool board[7][7][2], int action, bool colour[2])
{
    int piece = action / 25;
    int mv = action % 25;
    int from_x = piece % 7;
    int from_y = piece / 7;
    int mv_x = (mv % 5) - 2;
    int mv_y = (mv / 5) - 2;

    int to_x = from_x + mv_x;
    int to_y = from_y + mv_y;

    if (abs(mv_x) == 2 || abs(mv_y) == 2)
    {
        board[from_y][from_x][0] = CLEAR[0];
        board[from_y][from_x][1] = CLEAR[1];
    }
    board[to_y][to_x][0] = colour[0];
    board[to_y][to_x][1] = colour[1];

    for (int u = 0; u < 3; u++)
    {
        for (int v = 0; v < 3; v++)
        {
            int c_x = to_x + u - 1;
            int c_y = to_y + v - 1;

            if ((0 <= c_x && c_x < 7) &&
                (0 <= c_y && c_y < 7))
            {
                if ((board[c_y][c_x][0] != colour[0]  &&
                     board[c_y][c_x][1] != colour[1]) &&
                    !(board[c_y][c_x][0] == CLEAR[0]  &&
                      board[c_y][c_x][1] == CLEAR[1]))
                {
                    board[c_y][c_x][0] = colour[0];
                    board[c_y][c_x][1] = colour[1];
                }
            }
        }
    }
}

// Move ordering structure
typedef struct {
    int16_t move;
    int16_t score;
} ScoredMove;

// Compare function for qsort
int compare_scored_moves(const void *a, const void *b) {
    return ((ScoredMove*)b)->score - ((ScoredMove*)a)->score;
}

// Order moves: TT move first, then captures, then killers, then rest
int order_moves(bool board[7][7][2], bool *moves, bool colour[2], int depth,
                int16_t tt_best_move, ScoredMove *scored_moves) {
    int move_count = 0;

    for (int i = 0; i < 1225; i++) {
        if (moves[i] == false) continue;

        scored_moves[move_count].move = i;
        scored_moves[move_count].score = 0;

        // TT move gets highest priority
        if (i == tt_best_move) {
            scored_moves[move_count].score = 10000;
        }
        // Captures get high priority
        else {
            int captures = count_captures(board, i, colour);
            scored_moves[move_count].score = captures * 100;

            // Killer moves get medium priority
            if (is_killer_move(depth, i)) {
                scored_moves[move_count].score += 50;
            }
        }

        move_count++;
    }

    // Sort moves by score (highest first)
    qsort(scored_moves, move_count, sizeof(ScoredMove), compare_scored_moves);

    return move_count;
}

float minimax_cached(bool board[7][7][2], int depth, float alpha, float beta,
                     bool max_player, bool score_colour[2], uint64_t hash)
{
    // Check transposition table
    float tt_score;
    int16_t tt_best_move = -1;
    if (tt_probe(hash, depth, alpha, beta, &tt_score, &tt_best_move)) {
        return tt_score;
    }

    float score = 0.0f;
    int empty = 0;
    float bias = max_player ? 0.5f : -0.5f;
    bool colour[2];

    bool opponent_colour[2];
    opponent_colour[0] = !score_colour[0];
    opponent_colour[1] = !score_colour[1];

    if (max_player)
    {
        colour[0] = score_colour[0];
        colour[1] = score_colour[1];
    } else {
        colour[0] = opponent_colour[0];
        colour[1] = opponent_colour[1];
    }

    score = get_score(board, score_colour);

    if (depth == 0)
    {
        score += bias;
        tt_store(hash, depth, score, 0, -1);  // Exact score
        return score;
    }

    bool valid_moves[7][7][5][5] = {0};
    get_valid_moves(board, colour, valid_moves);
    if (!any_moves(valid_moves))
    {
        empty = count_cells(board, CLEAR);
        float terminal_score = score - empty;
        tt_store(hash, depth, terminal_score, 0, -1);  // Exact score
        return terminal_score;
    }

    bool *moves = (bool *)valid_moves;

    // Order moves for better alpha-beta pruning
    ScoredMove scored_moves[1225];
    int move_count = order_moves(board, moves, colour, depth, tt_best_move, scored_moves);

    float value = 0.0f;
    int16_t best_move = -1;
    int8_t flag = 2;  // Upper bound by default

    if (max_player == true)
    {
        value = -INFINITY;
        for (int idx = 0; idx < move_count; idx++)
        {
            int i = scored_moves[idx].move;

            bool child[7][7][2];
            memcpy(child, board, sizeof(child));
            move(child, i, colour);

            uint64_t child_hash = compute_zobrist_hash(child);
            float eval = minimax_cached(child, depth-1, alpha, beta, false, score_colour, child_hash);

            if (eval > value) {
                value = eval;
                best_move = i;
            }

            alpha = fmaxf(alpha, eval);
            if (beta <= alpha) {
                store_killer_move(depth, i);  // Store killer move
                flag = 1;  // Lower bound
                break;
            }
        }
        if (value > -INFINITY && value < INFINITY && flag == 2) flag = 0;  // Exact
    } else {
        value = INFINITY;
        for (int idx = 0; idx < move_count; idx++)
        {
            int i = scored_moves[idx].move;

            bool child[7][7][2];
            memcpy(child, board, sizeof(child));
            move(child, i, colour);

            uint64_t child_hash = compute_zobrist_hash(child);
            float eval = minimax_cached(child, depth-1, alpha, beta, true, score_colour, child_hash);

            if (eval < value) {
                value = eval;
                best_move = i;
            }

            beta = fminf(beta, eval);
            if (beta <= alpha) {
                store_killer_move(depth, i);  // Store killer move
                flag = 1;  // Lower bound (in minimizer context)
                break;
            }
        }
        if (value > -INFINITY && value < INFINITY && flag == 2) flag = 0;  // Exact
    }

    // Store in transposition table
    tt_store(hash, depth, value, flag, best_move);

    return value;
}

int find_best_move(bool game_board[7][7][2], int depth, bool as_blue)
{
    // Initialize transposition table if not already done
    init_tt();

    // Clear TT for new search (or keep for iterative deepening)
    // clear_tt();  // Comment out to keep cache between moves

    bool valid_moves[7][7][5][5] = {0};
    bool colour[2];
    float score;

    if (as_blue == true)
    {
        colour[0] = BLUE[0];
        colour[1] = BLUE[1];
    } else {
        colour[0] = GREEN[0];
        colour[1] = GREEN[1];
    }
    get_valid_moves(game_board, colour, valid_moves);
    bool *moves = (bool *)valid_moves;

    float max = -INFINITY;
    int residx = -1;

    // Order moves at root
    ScoredMove scored_moves[1225];
    int move_count = order_moves(game_board, moves, colour, 0, -1, scored_moves);

    for (int idx = 0; idx < move_count; idx++)
    {
        int i = scored_moves[idx].move;

        bool child[7][7][2] = {0};
        memcpy(child, game_board, sizeof(child));
        move(child, i, colour);

        uint64_t hash = compute_zobrist_hash(child);
        score = minimax_cached(child, depth-1, -INFINITY, INFINITY, false, colour, hash);

        if (score >= 120.0f)
        {
            return i;  // Winning move
        }
        if (score > max)
        {
            max = score;
            residx = i;
        }
    }

    return residx;
}

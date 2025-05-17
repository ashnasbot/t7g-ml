#include <stdbool.h>
#include <string.h>
#include <sys/types.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#define BOARD_SIZE 7

bool valid_moves[7][7][5][5] = {0};
bool BLUE[2] = {0, 1};
bool GREEN[2] = {1, 0};
bool CLEAR[2] = {0, 0};

/*
*   Returns true if any valid action currently exists.
*/
bool any_moves(void)
{
    for (int x = 0; x < 7; x++)
    {
        for (int y = 0; y < 7; y++)
        {
            for (int u = 0; u < 7; u++)
            {
                for (int v = 0; v < 7; v++)
                {
                    if(valid_moves[y][y][v][u] != 0) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

/*
*   Count the number of cells of the specified colour on the board.
*/
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


/*
*   Get the current score for the given colour.
*/
int get_score(bool board[7][7][2], bool colour[2])
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

    return player_cells - opponent_cells;
}


/*
*   Update the global valid_moves structure with the valid moves for the
*   given board and player colour.
*/
void update_valid_moves(bool board[7][7][2], bool colour[2])
{
    memset(valid_moves, 0, sizeof(valid_moves));
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
                                moves[moveid] = 1;
                            }
                        }
                    }
                }
            }
        }
    }
}

/*
*   Move a cell using the action index.
*   Handles jumps and capturing opponent peices using the supplied colour.
*/
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
        // Jump
        board[from_y][from_x][0] = CLEAR[0];
        board[from_y][from_x][1] = CLEAR[1];
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

/*
* Alpha beta pruning, modelled off of https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning
* values ripped off from darkshoxx.
*/
float minimax(bool board[7][7][2], int depth, float alpha, float beta, bool max_player)
{
    float score = 0.0;
    int empty = 0;
    float bias = max_player ? 0.5f : -0.5f;
    bool colour[2];

    if (max_player)
    {
        colour[0] = BLUE[0];
        colour[1] = BLUE[1];
    } else {
        colour[0] = GREEN[0];
        colour[1] = GREEN[1];
    }

    if (depth == 0)
    {
        score = get_score(board, GREEN);
        score += bias;
        return score;
    }

    update_valid_moves(board, colour);
    if (!any_moves())
    {
        score = get_score(board, GREEN);
        empty = count_cells(board, CLEAR);
        if (empty == 0)
        {
            if (score > 0)
                score = 100.0f;
            score = -100.0f;
        }
        else
        {
            score = score + (bias * 2 * empty);
        }
        return score;
    }

    // Need to copy here as we'll update actions inside each iteration
    bool moves_cpy[7][7][5][5] = {0};
    bool *moves = (bool *)moves_cpy;
    memcpy(moves, valid_moves, sizeof(valid_moves));

    float value = 0.0f;
    if (max_player)
    {
        value = -100.0f;
        for (int i = 0; i < 1225; i++)
        {
            if (moves[i] == 0)
            {
                continue;
            }
            bool newboard[7][7][2];
            memcpy(newboard, board, sizeof(newboard));
            move(newboard, i, colour);

            value = fmax(value, minimax(newboard, depth-1, alpha, beta, false));
            if (value >= beta)
                break;
            alpha = fmax(alpha, value);
        }
        return value;
    }
    else
    {
        value = 100.0f;
        for (int i = 0; i <  1225; i++)
        {
            if (moves[i] == 0)
            {
                continue;
            }
            bool newboard[7][7][2];
            memcpy(newboard, board, sizeof(newboard));
            move(newboard, i, colour);

            value = fmin(value, minimax(newboard, depth-1, alpha, beta, true));
            if (value <= alpha)
                break;
            beta = fmin(beta, value);
        }
        return value;
    }
    return -999.0f;
}

/*
*   ENTRYPOINT - build as dll.
*
*   Find the best move for the given baord.
*   samples the available legal moves and scores them for efficacy - depth turns in.
*   may take a looong time to compute for complex boards with lots of available moves.
*   Keep depth < 8.
*/
int find_best_move(bool game_board[7][7][2], int depth, bool turn)
{
    update_valid_moves(game_board, GREEN);
    bool *moves = (bool *)valid_moves;

    float res[1225] = {[0 ... 1224] = -200.0f};

    for (int i = 0; i <  1225; i++)
    {
        if (moves[i] == 0)
        {
            continue;
        }
        bool newboard[7][7][2] = {0};
        memcpy(newboard, game_board, 49*sizeof(bool)*2);

        move(newboard, i, GREEN);

        float score = minimax(game_board, depth, -100.0f, 100.0f, !turn);
        res[i] = score;
        if (score == 100.0f)
        {
            return i;
        }
    }
    float max = -100.0f;
    int residx = 1225;
    for (int i = 0; i <  1225; i++)
    {
        if (res[i] > max)
        {
            max = res[i];
            residx = i;
        }
    }

    return residx;
}
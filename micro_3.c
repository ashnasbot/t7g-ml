/*
* A solver for the T7G 'microscope' puzzle.
* Inspired by https://github.com/darkshoxx/Trilobyters
*/
#include <stdbool.h>
#include <string.h>
#include <sys/types.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#define BOARD_SIZE 7

bool BLUE[2] = {0, 1};
bool GREEN[2] = {1, 0};
bool CLEAR[2] = {0, 0};

/*
*   TODO:
*       - Sort moves so that those that take are ranked higher.
*       - Enable evaluating blue moves as well as green
*/

/*
*   Shuffle the elements of array, badly.
*/ 
void shuffle(int *array, size_t n)
{
    if (n > 1) 
    {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          int t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}

/*
*   Returns true if any valid action currently exists.
*/
bool any_moves(bool valid_moves[7][7][5][5])
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

    return (player_cells - opponent_cells) * 1.0f;
}


void print_board(bool board[7][7][2])
{
    printf("[");
    for (int y = 0; y < 7; y++)
    {
        printf("[");
        for (int x = 0; x < 7; x++)
        {
            printf("[%d,%d],", board[y][x][0], board[y][x][1]);
        }
        printf("]\n");
    }
    printf("]\n");
}


/*
*   Update the global valid_moves structure with the valid moves for the
*   given board and player colour.
*/
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
    float score = 0.0f;
    int empty = 0;
    float bias = max_player ? 0.5f : -0.5f;
    bool colour[2];

    if (max_player)
    {
        colour[0] = GREEN[0];
        colour[1] = GREEN[1];
    } else {
        colour[0] = BLUE[0];
        colour[1] = BLUE[1];
    }

    if (depth == 0)
    {
        score = get_score(board, GREEN);
        //printf("score: %f\n", score);
        return score + bias;
    }

    bool valid_moves[7][7][5][5] = {0};
    get_valid_moves(board, colour, valid_moves);
    if (!any_moves(valid_moves))
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

    // 1D iterable for moves where index = move id
    bool *moves = (bool *)valid_moves;

    float value = 0.0f;
    if (max_player == true)
    {
        value = -INFINITY;
        for (int i = 0; i < 1225; i++)
        {
            if (moves[i] == false) { continue; }

            bool child[7][7][2];
            memcpy(child, board, sizeof(child));
            move(child, i, colour);

            float eval = minimax(child, depth-1, alpha, beta, false);
            value = fmaxf(value, eval);
            alpha = fmaxf(alpha, eval);
            if (beta <= alpha)
                break;
        }
    }
    else
    {
        value = INFINITY;
        for (int i = 0; i <  1225; i++)
        {
            if (moves[i] == false) { continue; }

            bool child[7][7][2];
            memcpy(child, board, sizeof(child));
            move(child, i, colour);

            float eval = minimax(child, depth-1, alpha, beta, true);
            value = fminf(value, eval);
            beta = fminf(beta, eval);
            if (beta <= alpha)
                break;
        }
    }
    //if (depth < 2)
    //{
//        printf("depth: %d, alpha: %f, beta: %f, score: %f\n", depth, alpha, beta, value);
    //}
    return value;
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
    bool valid_moves[7][7][5][5] = {0};
    bool colour[2];

    // Non-standard GNU extension
    float res[1225] = {[0 ... 1224] = -INFINITY};
    float score;

    if (turn)
    {
        colour[0] = GREEN[0];
        colour[1] = GREEN[1];
    } else {
        colour[0] = BLUE[0];
        colour[1] = BLUE[1];
    }
    get_valid_moves(game_board, colour, valid_moves);
    bool *moves = (bool *)valid_moves;

    int order[1225] = {0};
    for (int i = 0; i <  1225; i++)
    {  
        order[i] = i;
    }
    shuffle(order, 1225);

    float max = -INFINITY;
    int residx = 1225;
    for (int j = 0; j <  1225; j++)
    {
        int i = order[j];
        if (moves[i] == 0)
        {
            continue;
        }
        bool child[7][7][2] = {0};
        memcpy(child, game_board, sizeof(child));

        move(child, i, colour);

        score = minimax(child, depth-1, -INFINITY, INFINITY, !turn);
        res[i] = score;

        // Early break for winning move
        if (score >= 50.0f)
        {
            return i;
        }
        if (score > max)
        {
            max = score;
            residx = i;
        }
    }

    // Every move we have probably loses, just pick something at random.
    if (residx == 1225)
    {
        for (int j = 0; j <  1225; j++)
        {
            int i = order[j];
            if (moves[i] == 0)
            {
                continue;
            }
            return i;
        }
        // There *are* no moves - usually a terminal state
        return -1;
    }

    return residx;
}

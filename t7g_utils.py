import numpy
from PIL import Image
from term_image.image import AutoImage
import numpy.typing as npt

BLUE = numpy.array([0, 1], dtype=bool)
GREEN = numpy.array([1, 0], dtype=bool)
CLEAR = numpy.array([0, 0], dtype=bool)


def count_cells(board):
    return numpy.sum(numpy.all(board == BLUE, axis=-1)), numpy.sum(numpy.all(board == GREEN, axis=-1))


def show_board(board):
    img_arr = board.astype(dtype=numpy.uint8)
    img_arr = numpy.dstack((numpy.zeros((7, 7), dtype=numpy.uint8), img_arr))
    img_arr[img_arr == 1] = 255
    img = Image.fromarray(img_arr, 'RGB')
    AutoImage(img).draw(h_align="left", pad_height=-6)


def action_to_move(action):
    piece = action // 25
    move = action % 25
    from_x = piece % 7
    from_y = piece // 7
    mv_x = (move % 5) - 2
    mv_y = (move // 5) - 2

    to_x = from_x + mv_x
    to_y = from_y + mv_y

    jump = True if abs(mv_x) == 2 or abs(mv_y) == 2 else False

    return from_x, from_y, to_x, to_y, jump


def move_to_action(x, y, x2, y2):
    piece = y * 7 + x
    move = y2 * 5 + x2

    return piece * 25 + move


def calc_reward(board, as_blue=True):
    reward = 0
    terminated = False
    new_blue, new_green = count_cells(board)

    if as_blue:
        player_cells = new_blue
        opponent_cells = new_green
    else:
        player_cells = new_green
        opponent_cells = new_blue

    if player_cells == 0:  # We have lost
        reward = -100
        terminated = True
    elif opponent_cells == 0:  # We have won!
        print("win!")
        reward = 100
        terminated = True
    else:
        cell_diff = player_cells - opponent_cells

        if cell_diff > 0:
            reward = 1
        elif cell_diff < 0:
            reward = -1
        reward = 0
    return reward, terminated


def action_masks(game_grid: npt.NDArray[numpy.bool], turn: bool):

    player_cell = BLUE if turn else GREEN

    actions = numpy.zeros((7, 7, 5, 5), dtype=numpy.bool)

    # TODO: This could be so much faster if vectorised
    for y, x in numpy.ndindex((7, 7)):
        if numpy.array_equal(game_grid[y, x], player_cell):
            # We're moving our own piece
            for u, v in numpy.ndindex((5, 5)):
                to_x = x + u - 2
                to_y = y + v - 2

                if 0 <= to_x < 7 and\
                   0 <= to_y < 7:
                    if not numpy.any(game_grid[to_y, to_x]):
                        actions[y, x, v, u] = 1

    return actions.flatten()


def is_action_valid(board, action, as_blue):
    if as_blue:
        player_cell = BLUE
    else:
        player_cell = GREEN

    from_x, from_y, to_x, to_y, _ = action_to_move(action)
    if numpy.array_equal(board[from_y, from_x], player_cell):
        # We are trying to move our own piece

        if 0 <= to_x < 7 and\
            0 <= to_y < 7:

            if not any(board[to_y, to_x]):
                # The Dest is free
                return True
    return False

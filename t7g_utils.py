import numpy
from PIL import Image
from term_image.image import AutoImage

BLUE = [0, 0, 1]
GREEN = [0, 1, 0]
CLEAR = [0, 0, 0]


def count_cells(board):
    b = board.reshape(-1, 3)
    blue = 0
    green = 0
    for triplet in b:
        if triplet[2] == 1:
            blue += 1
        elif triplet[1] == 1:
            green += 1

    return blue, green


def show_board(board):
    img_arr = numpy.copy(board)
    img_arr[img_arr == 1] = 255
    img = Image.fromarray(img_arr, 'RGB')
    AutoImage(img).draw(h_align="left")


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

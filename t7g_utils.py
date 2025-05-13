import numpy
from PIL import Image
from term_image.image import AutoImage

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

from ..game.cell import Cell
from ..config import Config


def get_neighbours(board: list[list[Cell]], x: int, y: int, config: Config) -> list[Cell]:
    """Return the neighbours of the cell"""
    neighbours = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            if 0 <= x + dx < config.game.width and 0 <= y + dy < config.game.height:
                neighbours.append(board[y + dy][x + dx])
    return neighbours


def get_neighbours_coords(x: int, y: int, config: Config) -> list[tuple[int, int]]:
    """Return the neighbours coordinates of the cell"""
    neighbours = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            if 0 <= x + dx < config.game.width and 0 <= y + dy < config.game.height:
                neighbours.append((x + dx, y + dy))
    return neighbours


def get_number_of_mines(board: list[list[Cell]], x: int, y: int, config: Config) -> int:
    """Return the number of mines around the cell"""
    count = 0
    for neighbour in get_neighbours(board, x, y, config):
        if neighbour.get_is_mine():
            count += 1
    return count

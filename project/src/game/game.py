from __future__ import annotations
from ..utils.game_func import get_number_of_mines, get_neighbours, get_neighbours_coords
from .cell import Cell, CellState
from typing import TYPE_CHECKING
from ..config import Config
import random

if TYPE_CHECKING:
    from ..manager.manager import Manager


class Game:

    def __init__(self, config: Config, controller: Manager) -> None:
        self.config = config
        self.controller = controller

        self.board = self.create_board()

        self.is_game_over = False
        self.is_game_won = False

        self.reveal_random_cell()

    # Requests
    def create_board(self) -> list[list[Cell]]:
        """Create the board"""

        # board initialization
        board = []
        for _ in range(self.config.game.height):
            row = []
            for _ in range(self.config.game.width):
                row.append(Cell(CellState.HIDDEN, 0, False))
            board.append(row)

        # place mines
        mine_count = self.get_max_mines()
        while mine_count > 0:
            x = random.randint(0, self.config.game.width - 1)
            y = random.randint(0, self.config.game.height - 1)

            if not board[y][x].get_is_mine():
                board[y][x].set_is_mine(True)
                mine_count -= 1

        # update values
        for y in range(self.config.game.height):
            for x in range(self.config.game.width):
                if not board[y][x].get_is_mine():
                    board[y][x].set_value(
                        get_number_of_mines(board, x, y, self.config))

        return board

    def get_board(self) -> list[list[Cell]]:
        """Return the board"""
        return self.board

    def get_board_size(self) -> tuple[int, int]:
        """Return the board size"""
        return self.config.game.width, self.config.game.height

    def check_win(self) -> bool:
        """Check if the game is won"""
        for row in self.board:
            for cell in row:
                if not cell.get_is_mine() and cell.get_state() != CellState.REVEALED:
                    return False
        return True

    def get_cells(self) -> list[Cell]:
        """Return the cells"""
        cells = []
        for row in self.board:
            for cell in row:
                cells.append(cell)
        return cells

    def get_adjacent_cells(self) -> list[tuple[Cell, tuple[int, int]]]:
        """Return hidden cells with revealed adjacent cells"""
        cells = []
        for i, row in enumerate(self.board):
            for j, cell in enumerate(row):
                if cell.get_state() == CellState.HIDDEN:
                    for ncell in get_neighbours(self.board, j, i, self.config):
                        if ncell.get_state() == CellState.REVEALED:
                            cells.append((cell, (j, i)))
                            break
        return cells

    def is_game_end(self) -> bool:
        """Return if the game is over"""
        return self.is_game_over

    def get_exploded_cell(self) -> tuple[int, int]:
        """Return the exploded cell"""
        for i, row in enumerate(self.board):
            for j, cell in enumerate(row):
                if cell.get_state() == CellState.REVEALED and cell.get_is_mine():
                    return j, i
        return -1, -1

    # Commands
    def update(self) -> None:
        """Update the game"""
        move = self.controller.get_move(self)
        if move is not None:
            x, y = move
            self.reveal(x, y)

    def run(self) -> None:
        while not self.is_game_over and not self.is_game_won:
            self.update()

    def reset(self) -> None:
        """Reset the game"""
        self.board = self.create_board()
        self.is_game_over = False
        self.is_game_won = False

    def flag(self, x: int, y: int) -> None:
        """Flag the cell if unflagged, unflag if flagged"""
        if not self.is_game_over:
            cell = self.board[y][x]
            if cell.get_state() == CellState.HIDDEN:
                cell.set_state(CellState.FLAGGED)
            elif cell.get_state() == CellState.FLAGGED:
                cell.set_state(CellState.HIDDEN)

    def reveal(self, x: int, y: int) -> None:
        """Reveal the cell"""
        if not self.is_game_over:
            cell = self.board[y][x]
            if cell.get_state() == CellState.HIDDEN:
                cell.set_state(CellState.REVEALED)

                if cell.get_is_mine():
                    self.is_game_over = True
                    self.reveal_mines()
                    cell.set_state(CellState.REVEALED)
                else:
                    self.is_game_won = self.check_win()
                    if self.is_game_won:
                        self.is_game_over = True
                        self.reveal_mines()

                self.expand(x, y)

    def reveal_mines(self) -> None:
        """Reveal all mines"""
        for row in self.board:
            for cell in row:
                if cell.get_is_mine():
                    cell.set_state(CellState.EXPLODED)

    def expand(self, x: int, y: int) -> None:
        """recursively reveal cells with value 0"""
        if self.board[y][x].get_value() == 0:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    new_x = x + dx
                    new_y = y + dy
                    if 0 <= new_x < self.config.game.width and 0 <= new_y < self.config.game.height:
                        self.reveal(new_x, new_y)

    def reveal_random_cell(self) -> None:
        """Reveal a random cell with value 0"""
        cells_with_position = [
            (cell, (j, i)) for i, row in enumerate(self.board) for j, cell in enumerate(row) if cell.get_value() == 0
        ]
        random.shuffle(cells_with_position)
        for cell, (i, j) in cells_with_position:
            if cell.get_state() == CellState.HIDDEN and cell.get_is_mine() == False:
                self.reveal(i, j)
                break

    # Utils
    def get_max_mines(self) -> int:
        """Return the maximum number of mines"""
        return int(
            self.config.game.width * self.config.game.height *
            self.config.game.mines_percentage
        )

    def display_board(self) -> None:
        """Display the board"""
        for row in self.board:
            for cell in row:
                if cell.get_is_mine():
                    print("X", end=" ")
                else:
                    print(cell.get_value(), end=" ")
            print()
        print()

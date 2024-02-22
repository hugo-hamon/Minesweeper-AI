from ..utils.game_func import get_number_of_mines
from .cell import Cell, CellState
from typing import Callable
from ..config import Config
import random


class Game:

    def __init__(self, config: Config, controller: Callable) -> None:
        self.config = config
        self.controller = controller

        self.board = self.create_board()

        self.is_game_over = False
        self.is_game_won = False

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
                    board[y][x].set_value(get_number_of_mines(board, x, y, self.config))

        return board
    
    def get_board(self) -> list[list[Cell]]:
        """Return the board"""
        return self.board
    
    def get_board_size(self) -> tuple[int, int]:
        """Return the board size"""
        return self.config.game.width, self.config.game.height
    
    # Commands
    def run(self) -> None:
        pass

    def reset(self) -> None:
        """Reset the game"""
        self.board = self.create_board()
        self.is_game_over = False
        self.is_game_won = False

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
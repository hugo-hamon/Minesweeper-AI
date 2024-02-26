from __future__ import annotations
from ..utils.game_func import get_number_of_mines, get_neighbours, get_neighbours_coords
from typing import TYPE_CHECKING, Tuple
from .cell import Cell, CellState
from ..config import Config
import random
import time
import math

if TYPE_CHECKING:
    from ..manager.manager import Manager

CELL_CORD = Tuple[int, int]


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
            row = [
                Cell(CellState.HIDDEN, 0, False)
                for _ in range(self.config.game.width)
            ]
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
            cells.extend(iter(row))
        return cells

    def get_adjacent_cells(self) -> list[tuple[Cell, CELL_CORD]]:
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

    def get_exploded_cell(self) -> CELL_CORD:
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
        self.is_game_won = self.check_win()

    def reset(self) -> None:
        """Reset the game"""
        self.board = self.create_board()
        self.is_game_over = False
        self.is_game_won = False

        self.reveal_random_cell()

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
        if self.is_game_over:
            return
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

    def get_clusters(self) -> list[list[CELL_CORD]]:
        """Return a list of unvisited cells clusters"""
        unvisited_cells = self.get_unvisited_cells()
        clusters = []

        while unvisited_cells:
            start_cell = unvisited_cells.pop()
            cluster = [start_cell]
            to_visit = [start_cell]
            visited_cells = {start_cell}

            while to_visit:
                current_cell = to_visit.pop(0)
                neigbours = self.get_unvisited_neighbours(current_cell)
                new_neigbours = [
                    neigbour for neigbour in neigbours if neigbour in unvisited_cells and neigbour not in visited_cells
                ]
                to_visit.extend(new_neigbours)
                unvisited_cells -= set(new_neigbours)
                visited_cells.update(new_neigbours)
                cluster.extend(new_neigbours)

            clusters.append(cluster)

        return clusters

    def get_unvisited_cells(self) -> set[CELL_CORD]:
        """Return a set of unvisited cells"""
        unvisited_cells = set()
        for i, row in enumerate(self.board):
            for j, cell in enumerate(row):
                if cell.get_state() == CellState.HIDDEN:
                    unvisited_cells.add((j, i))
        return unvisited_cells

    def get_unvisited_neighbours(self, cell: CELL_CORD) -> list[CELL_CORD]:
        """Return a list of unvisited neighbours"""
        neighbours = get_neighbours_coords(cell[0], cell[1], self.config)
        return [
            neighbour for neighbour in neighbours
            if self.board[neighbour[1]][neighbour[0]].get_state()
            == CellState.HIDDEN
        ]

    def is_valid(self, arrangement: list[tuple[int, CELL_CORD]], value_cells: set[CELL_CORD]) -> bool:
        """Return if the arrangement is valid or not"""
        for value_cell in value_cells:
            value = self.board[value_cell[1]][value_cell[0]].get_value()
            neighbours = get_neighbours_coords(value_cell[0], value_cell[1], self.config)
            revealed_neighbours = 0
            for _, cell in arrangement:
                if cell in neighbours:
                    revealed_neighbours += 1
            
            if revealed_neighbours > value:
                return False
        return True
    
    def is_complete(self, arrangement: list[tuple[int, CELL_CORD]], value_cells: set[CELL_CORD]) -> bool:
        """Return if the arrangement is complete or not"""
        for value_cell in value_cells:
            value = self.board[value_cell[1]][value_cell[0]].get_value()
            neighbours = get_neighbours_coords(value_cell[0], value_cell[1], self.config)
            revealed_neighbours = 0
            arrangement_cells = [cell for _, cell in arrangement]
            for neighbour in neighbours:
                cells_state = self.board[neighbour[1]][neighbour[0]].get_state()
                if neighbour in arrangement_cells or cells_state == CellState.FLAGGED:
                    revealed_neighbours += 1
            
            if revealed_neighbours != value:
                return False
        return True

    def get_arrangement_rec(
        self,
        current_arrangement: list[tuple[int, CELL_CORD]], remaining_cells: list[tuple[int, CELL_CORD]],
        arrangements: list[list[tuple[int, CELL_CORD]]], value_cells: set[CELL_CORD]
    ):
        """Return all possible arrangements recursively"""
        if self.is_complete(current_arrangement, value_cells):
            arrangements.append(current_arrangement[:])
            return

        for i in range(len(remaining_cells)):
            idx = current_arrangement[-1][0]
            if remaining_cells[i][0] <= idx:
                continue
            cell = remaining_cells[i]
            new_arrangement = current_arrangement + [cell]
            if self.is_valid(new_arrangement, value_cells):
                self.get_arrangement_rec(
                    new_arrangement, remaining_cells[:i] +
                    remaining_cells[i+1:], arrangements, value_cells
                )

    def get_arrangements(self, cluster: list[CELL_CORD]) -> list[list[tuple[int, CELL_CORD]]]:
        """Return all possible arrangements"""
        unvisited_cells = self.get_adjacent_cells_from_cluster(cluster)
        index_cells = list(enumerate(unvisited_cells))
        value_cells = self.get_value_cells(unvisited_cells)
        arrangements = []
        for index_cell in index_cells:
            new_index_cells = index_cells[:]
            new_index_cells.remove(index_cell)
            self.get_arrangement_rec(
                [index_cell], new_index_cells, arrangements, value_cells
            )

        return arrangements

    def get_adjacent_cells_from_cluster(self, cluster: list[CELL_CORD]) -> list[CELL_CORD]:
        """Return hidden cells with revealed adjacent cells"""
        adjacent_cells = []
        for cell in cluster:
            neighbours = get_neighbours_coords(cell[0], cell[1], self.config)
            for neighbour in neighbours:
                ncell = self.board[neighbour[1]][neighbour[0]]
                if ncell.get_state() == CellState.REVEALED:
                    adjacent_cells.append(cell)
                    break
        return adjacent_cells

    def get_value_cells(self, unvisited_cells: list[CELL_CORD]) -> set[CELL_CORD]:
        """Return the value cells neighbours of the unvisited cells"""
        value_cells = set()
        for cell in unvisited_cells:
            neighbours = get_neighbours_coords(cell[0], cell[1], self.config)
            for neighbour in neighbours:
                value_cell = self.board[neighbour[1]][neighbour[0]]
                if value_cell.get_value() > 0 and value_cell.get_state() == CellState.REVEALED:
                    value_cells.add(neighbour)
        return value_cells

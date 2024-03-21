from ..utils.game_func import get_neighbours_coords
from ..game.cell import CellState
from .manager import Manager
from ..game.game import Game
from typing import Optional
from ..config import Config
import numpy as np
import operator
import random


class RuleBasedManager(Manager):

    def __init__(self, config: Config, use_random_move=True) -> None:
        super().__init__(config)
        self.current_move = None
        self.use_random_move = use_random_move

        self.visited = np.zeros(
            (self.config.game.height, self.config.game.width))

    def set_move(self, game: Game, move: tuple[int, int]) -> None:
        """Set the current move"""
        self.current_move = move

    def get_move(self, game: Game) -> Optional[tuple[int, int]]:
        """Return the current move"""
        if game.is_game_end():
            return None
        self.basic_strategy_flag(game)
        self.group_strategy_flag(game)

        move = self.basic_strategy_reveal(game)
        if move is not None:
            return move

        move = self.group_strategy_reveal(game)
        if move is not None:
            return move

        move, flag = self.cluster_strategy(game)
        if move is not None:
            return move
        if flag:
            return None

        return self.random_move(game) if self.use_random_move else None

    def basic_strategy_flag(self, game: Game) -> None:
        """Flag cells based on basic strategy"""
        # Check if there is a cell were is value is equal to the number of unrevealed cells
        board = game.get_board()
        for y in range(self.config.game.height):
            for x in range(self.config.game.width):
                cell = board[y][x]
                if self.visited[y][x] != 0:
                    continue
                if cell.get_state() == CellState.REVEALED and cell.get_value() > 0:
                    neighbours = get_neighbours_coords(x, y, self.config)
                    unrevealed_neighbours = []
                    count = 0
                    for nx, ny in neighbours:
                        ncell = board[ny][nx]
                        if ncell.get_state() == CellState.HIDDEN:
                            unrevealed_neighbours.append((nx, ny))
                            count += 1
                        elif ncell.get_state() == CellState.FLAGGED:
                            count += 1
                    if count == cell.get_value():
                        for nx, ny in unrevealed_neighbours:
                            game.flag(nx, ny)
                        self.visited[y][x] = 1

    def basic_strategy_reveal(self, game: Game) -> Optional[tuple[int, int]]:
        """Reveal cells based on basic strategy"""
        # If there are already as many mines as the number, remaining cells are safe
        board = game.get_board()
        for y in range(self.config.game.height):
            for x in range(self.config.game.width):
                cell = board[y][x]
                if self.visited[y][x] != 0:
                    continue
                if cell.get_state() == CellState.REVEALED and cell.get_value() > 0:
                    neighbours = get_neighbours_coords(x, y, self.config)
                    flagged_neighbours = 0
                    for nx, ny in neighbours:
                        ncell = board[ny][nx]
                        if ncell.get_state() == CellState.FLAGGED:
                            flagged_neighbours += 1
                    if flagged_neighbours == cell.get_value():
                        for nx, ny in neighbours:
                            ncell = board[ny][nx]
                            if ncell.get_state() == CellState.HIDDEN:
                                return (nx, ny)
        return None

    def group_strategy_flag(self, game: Game) -> None:
        """Flag cells based on group strategy"""
        pairs = self.get_pairs(game)
        board = game.get_board()
        for x1, y1, x2, y2 in pairs:
            nfn_a = self.get_non_flagged_neighbours(x1, y1, game)
            nfn_b = self.get_non_flagged_neighbours(x2, y2, game)
            a_value = board[y1][x1].get_value(
            ) - len(self.get_flagged_neighbours(x1, y1, game))
            b_value = board[y2][x2].get_value(
            ) - len(self.get_flagged_neighbours(x2, y2, game))
            # only nfn_a not in nfn_b
            intersection_a = list(set(nfn_a).difference(nfn_b))
            if a_value - b_value == len(intersection_a):
                for nx, ny in intersection_a:
                    game.flag(nx, ny)
            # only nfn_b not in nfn_a
            intersection_b = list(set(nfn_b).difference(nfn_a))
            if b_value - a_value == len(intersection_b):
                for nx, ny in intersection_b:
                    game.flag(nx, ny)

    def group_strategy_reveal(self, game: Game) -> Optional[tuple[int, int]]:
        pairs = self.get_pairs(game)
        board = game.get_board()
        for x1, y1, x2, y2 in pairs:
            nfn_a = self.get_non_flagged_neighbours(x1, y1, game)
            nfn_b = self.get_non_flagged_neighbours(x2, y2, game)
            a_value = board[y1][x1].get_value(
            ) - len(self.get_flagged_neighbours(x1, y1, game))
            b_value = board[y2][x2].get_value(
            ) - len(self.get_flagged_neighbours(x2, y2, game))
            # only nfn_a not in nfn_b
            intersection_a = list(set(nfn_a).difference(nfn_b))
            intersection_b = list(set(nfn_b).difference(nfn_a))
            if a_value - b_value == len(intersection_a):
                for nx, ny in intersection_b:
                    return (nx, ny)
            # only nfn_b not in nfn_a
            if b_value - a_value == len(intersection_b):
                for nx, ny in intersection_a:
                    return (nx, ny)
        return None

    def cluster_strategy(self, game: Game) -> tuple[Optional[tuple[int, int]], bool]:
        """Flag cells based on cluster strategy"""
        clusters = game.get_clusters()
        for cluster in clusters:
            adjacent_cells = game.get_adjacent_cells_from_cluster(cluster)
            adjactent_cells_size = len(adjacent_cells)
            if adjactent_cells_size <= 20 and adjactent_cells_size > 0:
                return self.probability_move(game, cluster, adjacent_cells)
        return None, False

    def probability_move(self, game: Game, cluster: list[tuple[int, int]], adjacent_cells: list[tuple[int, int]]) -> tuple[Optional[tuple[int, int]], bool]:
        """Return a move based on the probability of a cell being a mine"""
        arrangements = game.get_arrangements(cluster)
        if len(arrangements) == 0:
            return None, False
        probs = {
            cell_coords: 0. for cell_coords in adjacent_cells
        }
        total = 0
        for arrangement in arrangements:
            cells = [cell_coords for _, cell_coords in arrangement]
            for x, y in cells:
                if (x, y) not in probs:
                    probs[(x, y)] = 0
                probs[(x, y)] += 1
                total += 1
        probs = {cell_coords: prob /
                 total for cell_coords, prob in probs.items()}
        mean = sum(probs.values()) / len(probs)

        # Check for a zero probability
        eps = 1e-3
        for cell_coords, prob in probs.items():
            if prob < eps:
                return cell_coords, False

        mean_diff = [
            (cell_coords, abs(prob - mean), prob > mean) for cell_coords, prob in probs.items()
        ]

        mean_diff.sort(key=operator.itemgetter(1), reverse=True)
        coords, prob_diff, to_flag = mean_diff[0]
        if to_flag:
            game.flag(*coords)
            return None, True

        return coords, False

    def random_move(self, game: Game) -> Optional[tuple[int, int]]:
        """Return a random move"""
        board = game.get_board()
        unrevealed_cells = []
        for y in range(self.config.game.height):
            for x in range(self.config.game.width):
                cell = board[y][x]
                if cell.get_state() == CellState.HIDDEN:
                    unrevealed_cells.append((x, y))
        return None if not unrevealed_cells else random.choice(unrevealed_cells)

    def get_pairs(self, game: Game) -> list[tuple[int, int, int, int]]:
        """Return the pairs of cells"""
        pairs = []
        board = game.get_board()
        for y in range(self.config.game.height):
            for x in range(self.config.game.width):
                cell = board[y][x]
                if self.visited[y][x] != 0:
                    continue
                if cell.get_state() == CellState.REVEALED and cell.get_value() > 0:
                    neighbours = get_neighbours_coords(x, y, self.config)
                    for nx, ny in neighbours:
                        ncell = board[ny][nx]
                        if ncell.get_state() == CellState.REVEALED and ncell.get_value() > 0 and ((nx, ny, x, y) not in pairs and (x, y, nx, ny) not in pairs):
                            pairs.append((x, y, nx, ny))
        return pairs

    def get_non_flagged_neighbours(self, x: int, y: int, game: Game) -> list[tuple[int, int]]:
        """Return the non-flagged neighbours"""
        board = game.get_board()
        neighbours = get_neighbours_coords(x, y, self.config)
        non_flagged_neighbours = []
        for nx, ny in neighbours:
            ncell = board[ny][nx]
            if ncell.get_state() == CellState.HIDDEN:
                non_flagged_neighbours.append((nx, ny))
        return non_flagged_neighbours

    def get_flagged_neighbours(self, x: int, y: int, game: Game) -> list[tuple[int, int]]:
        """Return the flagged neighbours"""
        board = game.get_board()
        neighbours = get_neighbours_coords(x, y, self.config)
        flagged_neighbours = []
        for nx, ny in neighbours:
            ncell = board[ny][nx]
            if ncell.get_state() == CellState.FLAGGED:
                flagged_neighbours.append((nx, ny))
        return flagged_neighbours

    def reset(self) -> None:
        """Reset the manager"""
        self.current_move = None
        self.visited = np.zeros(
            (self.config.game.height, self.config.game.width))

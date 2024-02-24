from ..trainer.cnn_trainer import CNNTrainer
from .manager import Manager
from ..game.game import Game
from typing import Optional
from ..config import Config
import numpy as np


class CNNManager(Manager):

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.current_move = None

        self.model = CNNTrainer(config)
        self.model.load_model("model/cnn/model.pth")

    def set_move(self, game: Game, move: tuple[int, int]) -> None:
        """Set the current move"""
        self.current_move = move

    def get_move(self, game: Game) -> Optional[tuple[int, int]]:
        """Return the current move"""
        adjacent_cells = game.get_adjacent_cells()
        adjactent_mask = np.zeros((self.config.game.height, self.config.game.width))
        for _, pos in adjacent_cells:
            adjactent_mask[pos[1]][pos[0]] = 1

        states = self.model.get_state(game)
        values = self.model.predict(states)

        values = np.multiply(values, adjactent_mask)
        move = (0, 0)
        max_value = -1
        for y in range(self.config.game.height):
            for x in range(self.config.game.width):
                if values[0][y][x] > max_value:
                    max_value = values[0][y][x]
                    move = (x, y)
        return move

    
    def reset(self) -> None:
        """Reset the manager"""
        self.current_move = None
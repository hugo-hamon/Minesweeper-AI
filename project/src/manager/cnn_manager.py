from ..trainer.cnn_trainer import CNNTrainer
from .manager import Manager
from ..game.game import Game
from typing import Optional
from ..config import Config
import time


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
        states = []
        for cell, pos in adjacent_cells:
            state = self.model.get_state(game, pos)
            states.append(state)

        values = self.model.predict(states)
        print(values)
        move = (0, 0)
        max_value = 1
        for value, (cell, pos) in zip(values, adjacent_cells):
            if 1 - value[0] < max_value:
                max_value = 1 - value[0]
                move = pos
        print(move)
        return move

    
    def reset(self) -> None:
        """Reset the manager"""
        self.current_move = None
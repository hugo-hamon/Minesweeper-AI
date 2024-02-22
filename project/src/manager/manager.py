from abc import ABC, abstractmethod
from ..game.game import Game
from ..config import Config
from typing import Optional


class Manager(ABC):

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def get_move(self, game: Game) -> Optional[tuple[int, int]]:
        """Return a direction for a given algorithm"""
        return NotImplemented
    
    @abstractmethod
    def set_move(self, game: Game, x: int, y: int) -> None:
        """Set a move for a given algorithm"""
        return NotImplemented
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the manager"""
        return NotImplemented
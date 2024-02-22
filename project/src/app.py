from .graphics.graphic_game import GraphicGame
from .utils.manager_func import match_manager
from .config import load_config
from .game.game import Game
import logging


class App:

    def __init__(self, config_path: str) -> None:
        self.config = load_config(config_path)

    def run(self) -> None:
        """Run the app with the given config"""

        manager = match_manager(self.config)
        logging.info(
            f"Starting the game with {self.config.user.algorithm} algorithm"
        )

        if self.config.graphics.graphic_enabled:
            game = GraphicGame(self.config, manager.get_move)
        else:
            game = Game(self.config, manager.get_move)
        
        game.run()

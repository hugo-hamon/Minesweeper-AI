from .trainer.cnn_rule_based_trainer import CNNRuledBasedTrainer
from .graphics.graphic_game import GraphicGame
from .utils.manager_func import match_manager
from .trainer.cnn_trainer import CNNTrainer
from .config import load_config
from .game.game import Game
import logging


class App:

    def __init__(self, config_path: str) -> None:
        self.config = load_config(config_path)

    def run(self) -> None:
        """Run the app with the given config"""
        if self.config.cnn.train_enabled:
            trainer = CNNTrainer(self.config)
            trainer.train()
        if self.config.cnn_rule_based.train_enabled:
            trainer = CNNRuledBasedTrainer(self.config)
            trainer.train()

        manager = match_manager(self.config)
        logging.info(
            f"Starting the game with {self.config.user.algorithm} algorithm"
        )

        if self.config.graphics.graphic_enabled:
            game = GraphicGame(self.config, manager)
        else:
            game = Game(self.config, manager)
        
        game.run()

    def test_performance(self) -> None:
        """Test the performance of the algorithm"""
        epochs = 10000
        manager = match_manager(self.config)
        game = Game(self.config, manager)
        wins = 0
        for k in range(epochs):
            manager.reset()
            game.reset()
            game.run()
            if game.is_game_won:
                wins += 1
            if (k + 1) % 10 == 0:
                logging.info(f"Game {k + 1}, win rate: {wins/(k + 1):.5f*100}%")
        
        logging.info(f"Win rate: {wins/epochs:.5f*100}%")
        
                

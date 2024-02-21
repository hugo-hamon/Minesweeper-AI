from .utils.manager_func import match_manager
from .config import load_config
from .game.game import Game



class App:

    def __init__(self, config_path: str) -> None:
        self.config = load_config(config_path)

    def run(self) -> None:
        """Run the app with the given config"""
        
        manager = match_manager(self.config)
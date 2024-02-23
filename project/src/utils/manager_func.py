from ..manager.human_manager import HumanManager
from ..manager.cnn_manager import CNNManager
from ..manager.manager import Manager
from ..config import Config


def match_manager(config: Config) -> Manager:
    """Return the manager based on the config"""
    match config.user.algorithm:
        case "human":
            return HumanManager(config)
        case "cnn":
            return CNNManager(config)
        case _:
            raise ValueError("Invalid algorithm")

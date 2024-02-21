from ..manager.manager import Manager
from ..config import Config


def match_manager(config: Config) -> Manager:
    """Return the manager based on the config"""
    match config.user.algorithm:
        case "human":
            return HumanManager(config)

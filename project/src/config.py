from dataclasses import dataclass
from dacite.core import from_dict
import toml


@dataclass
class User:
    algorithm: str


@dataclass
class Graphics:
    width: int
    height: int
    fps: int
    title: str
    icon_path: str


@dataclass
class Game:
    width: int
    height: int
    mines_percentage: float


@dataclass
class Config:
    user: User
    graphics: Graphics
    game: Game


def load_config(config_path: str) -> Config:
    """Load the config"""
    return from_dict(data_class=Config, data=toml.load(config_path))
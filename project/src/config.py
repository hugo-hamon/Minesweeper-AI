from dataclasses import dataclass
from dacite.core import from_dict
import toml


@dataclass
class User:
    algorithm: str


@dataclass
class Graphics:
    graphic_enabled: bool
    screen_width: int
    screen_height: int
    fps: int
    title: str
    icon_path: str
    delay: float
    screen_capture: bool


@dataclass
class Game:
    width: int
    height: int
    mines_percentage: float


@dataclass
class CNN:
    train_enabled: bool


@dataclass
class Config:
    user: User
    graphics: Graphics
    game: Game
    cnn: CNN


def load_config(config_path: str) -> Config:
    """Load the config"""
    return from_dict(data_class=Config, data=toml.load(config_path))
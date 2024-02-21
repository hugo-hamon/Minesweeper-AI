from ..game.game import Game
from ..config import Config
from typing import Callable
import pygame as pg


class GraphicGame(Game):

    def __init__(self, config: Config, controller: Callable) -> None:
        super().__init__(config, controller)
        pg.init()

        self.screen_width = self.config.graphics.width
        self.screen_height = self.config.graphics.height

        self.screen = pg.display.set_mode(
            (self.screen_width, self.screen_height)
        )
        pg.display.set_caption(self.config.graphics.title)

        self.clock = pg.time.Clock()
        self.canvas = pg.Surface((self.screen_width, self.screen_height))

        # load icon
        icon = pg.image.load(self.config.graphics.icon_path)
        pg.display.set_icon(icon)

        self.running = False

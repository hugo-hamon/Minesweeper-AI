from __future__ import annotations
from ..game.cell import CellState
from typing import TYPE_CHECKING
from ..game.game import Game
from ..config import Config
import pygame as pg
import time
import os

if TYPE_CHECKING:
    from ..manager.manager import Manager


BACKGROUND_COLOR = (66, 71, 105)


class GraphicGame(Game):

    def __init__(self, config: Config, controller: Manager) -> None:
        super().__init__(config, controller)
        pg.init()

        self.screen_width, self.screen_height, self.tile_padding = self.compute_screen_size()
        self.screen = pg.display.set_mode(
            (self.screen_width, self.screen_height)
        )
        pg.display.set_caption(self.config.graphics.title)

        self.clock = pg.time.Clock()
        self.canvas = pg.Surface((self.screen_width, self.screen_height))

        # load icon
        icon = pg.image.load(self.config.graphics.icon_path)
        pg.display.set_icon(icon)

        # load images
        self.images = {
            f"{i}": pg.image.load(f"asset/images/{i}.png") for i in range(1, 9)
        }
        self.images["mine"] = pg.image.load("asset/images/mine.png")
        self.images["flag"] = pg.image.load("asset/images/flag.png")
        self.images["empty"] = pg.image.load("asset/images/empty.png")
        self.images["red_mine"] = pg.image.load("asset/images/red_mine.png")
        self.images["undiscovered"] = pg.image.load(
            "asset/images/undiscovered.png")

        self.running = False
        self.last_update = 0

        png_dir = 'asset/screenshot'
        if not os.path.exists(png_dir):
            os.makedirs(png_dir)

    # Requests
    def get_tile_width(self) -> float:
        """Return the width of a tile"""
        row_number, _ = self.get_board_size()
        tiles_width = self.screen_width - self.tile_padding * (row_number + 1)
        return tiles_width / row_number

    def get_tile_height(self) -> float:
        """Return the height of a tile"""
        _, column_number = self.get_board_size()
        tiles_height = self.screen_height - \
            self.tile_padding * (column_number + 1)
        return tiles_height / column_number

    # Commands

    def run(self) -> None:
        """Run the graphic game"""
        self.running = True
        start_time = time.time()
        need_screenshot = False
        while self.running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False
                elif event.type == pg.MOUSEBUTTONDOWN:
                    if event.button == 3:
                        x, y = self.get_tile(*event.pos)
                        if x != -1 and y != -1:
                            self.flag(x, y)
                    if event.button == 1:
                        x, y = self.get_tile(*event.pos)
                        if x != -1 and y != -1:
                            self.controller.set_move(self, (x, y))

            # Update title
            if self.is_game_won or self.is_game_over:
                # Affichage de la fin de la partie
                pg.display.set_caption(
                    f"{self.config.graphics.title} - {elapsed_time} seconds - {'You won' if self.is_game_won else 'You lost'}")
            else:
                elapsed_time = int(time.time() - start_time)
                pg.display.set_caption(
                    f"{self.config.graphics.title} - {elapsed_time} seconds")

            # update game
            if time.time() - self.last_update > self.config.graphics.delay:
                self.update()
                self.last_update = time.time()
                need_screenshot = True

            # draw tiles
            self.draw_tiles()
            self.screen.blit(self.canvas, (0, 0))
            if need_screenshot and self.config.graphics.screen_capture:
                pg.image.save(self.canvas, f"asset/screenshot/{time.time()}.png")
                need_screenshot = False

            self.canvas.fill(BACKGROUND_COLOR)
            pg.display.update()
            self.clock.tick(self.config.graphics.fps)

        pg.quit()

    def draw_tiles(self) -> None:
        """Draw the tiles on the canvas"""
        tile_width = self.get_tile_width()
        tile_height = self.get_tile_height()

        mouse_x, mouse_y = pg.mouse.get_pos()

        for y, row in enumerate(self.board):
            for x, tile in enumerate(row):
                state = tile.get_state()
                value = tile.get_value()

                tile_x = self.tile_padding + \
                    (x * (tile_width + self.tile_padding))
                tile_y = self.tile_padding + \
                    (y * (tile_height + self.tile_padding))

                # Draw image
                if state == CellState.FLAGGED:
                    image = self.images["flag"]
                elif state == CellState.HIDDEN:
                    image = self.images["undiscovered"]
                elif state == CellState.EXPLODED:
                    image = self.images["mine"]
                elif state == CellState.REVEALED and tile.get_is_mine():
                    image = self.images["red_mine"]
                elif state == CellState.REVEALED and value == 0:
                    image = self.images["empty"]
                else:
                    image = self.images[f"{value}"]

                scale = 1
                if tile_x < mouse_x < tile_x + tile_width and tile_y < mouse_y < tile_y + tile_height:
                    scale = 1.1
                image = pg.transform.scale(
                    image, (int(tile_width * scale), int(tile_height * scale))
                )
                self.canvas.blit(image, (
                    tile_x - (tile_width * (scale - 1) / 2),
                    tile_y - (tile_height * (scale - 1) / 2)
                ))

    # Utils
    def compute_screen_size(self) -> tuple[int, int, int]:
        """Compute the screen size based on the board size"""
        max_width, max_height = self.config.graphics.screen_width, self.config.graphics.screen_height
        row_number, column_number = self.get_board_size()

        if row_number > column_number:
            width = max_width
            height = int((max_width / row_number) * column_number)
        else:
            height = max_height
            width = int((max_height / column_number) * row_number)

        if height > max_height:
            height = max_height
            width = int((max_height / column_number) * row_number)
        if width > max_width:
            width = max_width
            height = int((max_width / row_number) * column_number)

        tile_padding = int(0.005 * max(width, height))
        tile_padding = max(1, min(10, tile_padding))

        return width, height, tile_padding

    def get_tile(self, mx: int, my: int) -> tuple[int, int]:
        """Get the tile at the given x and y, where x and y are the mouse position"""
        tile_width = self.get_tile_width()
        tile_height = self.get_tile_height()

        for y, row in enumerate(self.board):
            for x in range(len(row)):
                tile_x = self.tile_padding + \
                    (x * (tile_width + self.tile_padding))
                tile_y = self.tile_padding + \
                    (y * (tile_height + self.tile_padding))
                if tile_x < mx < tile_x + tile_width and tile_y < my < tile_y + tile_height:
                    return x, y
        return -1, -1

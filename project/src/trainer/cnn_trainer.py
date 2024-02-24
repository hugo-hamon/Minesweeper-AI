from torch.utils.data import DataLoader, TensorDataset
from ..manager.human_manager import HumanManager
from ..game.cell import CellState
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ..game.game import Game
from ..config import Config
import torch.nn as nn
import seaborn as sns
import numpy as np
import logging
import random
import torch
import os

NUM_HIDDEN = 75
GAME_SIMULATION = 1000
TRAINING_ITERATIONS = 200


class CNN(nn.Module):

    def __init__(self, input_size: int, num_hidden: int) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            input_size, num_hidden, kernel_size=5, stride=1, padding="same"
        )
        self.conv2 = nn.Conv2d(
            num_hidden, num_hidden, kernel_size=5, stride=1, padding="same"
        )
        self.conv3 = nn.Conv2d(
            num_hidden, num_hidden, kernel_size=5, stride=1, padding="same"
        )
        self.conv4 = nn.Conv2d(
            num_hidden, 1, kernel_size=1, stride=1, padding="same"
        )

        model_tmp_path = "model/cnn/temp"
        if not os.path.exists(model_tmp_path):
            os.makedirs(model_tmp_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.softsign(self.conv4(x))
        return x


class CNNTrainer:

    def __init__(self, config: Config) -> None:
        self.config = config

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logging.info(f"Using {self.device}")

        self.model = CNN(10, NUM_HIDDEN).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self) -> None:
        states, values = self.generate_data(GAME_SIMULATION)
        logging.info(f"Data generated, {len(states)} states")

        dataset = TensorDataset(
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(values), dtype=torch.float32)
        )
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        losses = []
        for i in range(TRAINING_ITERATIONS):
            logging.info(f"----Training iteration {i}----")
            for state, value in dataloader:
                self.optimizer.zero_grad()
                state = state.to(self.device)
                value = value.to(self.device)
                output = self.model(state)
                loss = self.criterion(output, value)
                loss.backward()
                self.optimizer.step()
            losses.append(loss.item())
            logging.info(f"Loss: {loss.item()}")
            if i % 10 == 0:
                sns.set_theme()
                plt.plot(losses)
                plt.savefig("model/cnn/loss.png")
            if i % 50 == 0:
                self.save_model(f"model/cnn/temp/model_{i}.pth")

        logging.info("Training done")

        sns.set_theme()
        plt.plot(losses)
        plt.savefig("model/cnn/loss.png")

        self.save_model("model/cnn/model.pth")

    def predict(self, states: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            output = self.model(torch.tensor(
                states, dtype=torch.float32).to(self.device))
            output = output.cpu()
            return output.numpy()

    def generate_data(self, game_simulation: int) -> tuple[list, list]:
        states = []
        values = []
        for _ in range(game_simulation):
            manager = HumanManager(self.config)
            game = Game(self.config, manager)
            while not game.is_game_end():
                adjacent_cells = game.get_adjacent_cells()
                actions = []
                state = self.get_state(game)
                value = np.zeros(
                    (1, self.config.game.height, self.config.game.width)
                )
                for cell, pos in adjacent_cells:
                    if not cell.get_is_mine():
                        actions.append(pos)
                        value[0][pos[1]][pos[0]] = 1

                states.append(state)
                values.append(value)

                if len(actions) == 0:
                    break
                manager.set_move(game, random.choice(actions))
                game.update()
        return states, values

    def get_state(self, game: Game) -> np.ndarray:
        state = np.zeros((10, self.config.game.height, self.config.game.width))
        for y in range(game.config.game.height):
            for x in range(game.config.game.width):
                cell = game.get_board()[y][x]
                if cell.get_state() == CellState.REVEALED:
                    state[cell.get_value()][y][x] = 1
                elif cell.get_state() == CellState.HIDDEN:
                    state[9][y][x] = 1

        return state

    def save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path))

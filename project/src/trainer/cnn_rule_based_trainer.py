from ..manager.rule_based_manager import RuleBasedManager
from torch.utils.data import DataLoader, TensorDataset
from ..game.cell import CellState
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ..game.game import Game
from ..config import Config
from .cnn_model import CNN
import torch.nn as nn
import seaborn as sns
import numpy as np
import logging
import random
import torch
import os

NUM_HIDDEN = 75
TRAINING_ITERATIONS = 1000


class CNNRuledBasedTrainer:

    def __init__(self, config: Config) -> None:
        self.config = config

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logging.info(f"Using {self.device}")

        self.model = CNN(11, NUM_HIDDEN).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        model_tmp_path = "model/cnn_rule_based/temp"
        if not os.path.exists(model_tmp_path):
            os.makedirs(model_tmp_path)


    def train(self) -> None:
        """Train the model by playing games and generating data"""
        states = []
        values = []
        # Generate data
        for iteration in range(TRAINING_ITERATIONS):
            if iteration % 10 == 0:
                logging.info(f"----Iteration {iteration}----")
            manager = RuleBasedManager(self.config, use_random_move=False)
            game = Game(self.config, manager)
            while not game.is_game_end():
                move = manager.get_move(game)
                if move is not None:
                    x, y = move
                    game.reveal(x, y)
                else:
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

                    if not actions:
                        break
                    move = random.choice(actions)
                    game.reveal(move[0], move[1])

        logging.info(f"Data generated for {len(states)} states")
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
                plt.savefig("model/cnn_rule_based/cnn_rule_based_loss.png")
            if i % 50 == 0:
                self.save_model(f"model/cnn_rule_based/temp/cnn_rule_based_model_{i}.pth")

        logging.info("Training done")

        sns.set_theme()
        plt.plot(losses)
        plt.savefig("model/cnn/cnn_rule_based_loss.png")
        self.save_model("model/cnn/cnn_rule_based_model.pth")


    def predict(self, states: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            output = self.model(torch.tensor(
                states, dtype=torch.float32).to(self.device))
            output = output.cpu()
            return output.numpy()

    
    def get_state(self, game: Game) -> np.ndarray:
        state = np.zeros((11, self.config.game.height, self.config.game.width))
        for y in range(game.config.game.height):
            for x in range(game.config.game.width):
                cell = game.get_board()[y][x]
                if cell.get_state() == CellState.REVEALED:
                    state[cell.get_value()][y][x] = 1
                elif cell.get_state() == CellState.HIDDEN:
                    state[9][y][x] = 1
                elif cell.get_state() == CellState.FLAGGED:
                    state[10][y][x] = 1

        return state
    
    def save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path))
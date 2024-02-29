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

        self.model = CNN(11, NUM_HIDDEN).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

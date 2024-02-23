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
import torch

WINDOW_SIZE = 5
NUM_HIDDEN = 64
GAME_SIMULATION = 1000
TRAINING_ITERATIONS = 200


class CNN(nn.Module):

    def __init__(self, state_size: int, input_size: int, num_hidden: int) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_size, num_hidden,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden,
                               kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_hidden, num_hidden,
                               kernel_size=3, padding=1)
        self.fc = nn.Linear(num_hidden * state_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CNNTrainer:

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model = CNN(WINDOW_SIZE ** 2, 1, NUM_HIDDEN)
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
                output = self.model(state)
                loss = self.criterion(output, value)
                loss.backward()
                self.optimizer.step()
            losses.append(loss.item())
            logging.info(f"Loss: {loss.item()}")
        logging.info("Training done")

        sns.set_theme()
        plt.plot(losses)
        plt.savefig("model/cnn/loss.png")

        self.save_model("model/cnn/model.pth")


    def predict(self, states: list[np.ndarray]) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            output = self.model(torch.tensor(states, dtype=torch.float32))
            return output.numpy()
                

    def generate_data(self, game_simulation: int) -> tuple[list, list]:
        states = []
        values = []
        for _ in range(game_simulation):
            manager = HumanManager(self.config)
            game = Game(self.config, manager)
            while not game.is_game_end():
                adjacent_cells = game.get_adjacent_cells()
                action = None
                for cell, pos in adjacent_cells:
                    state = self.get_state(game, pos)
                    states.append(state)
                    if cell.get_is_mine():
                        value = np.array([0, 1])
                    else:
                        value = np.array([1, 0])
                        action = pos
                    values.append(value)
                if action is None:
                    break
                manager.set_move(game, action)
                game.update()
        return states, values
    
    def get_state(self, game: Game, pos: tuple[int, int]) -> np.ndarray:
        state = np.zeros((1, WINDOW_SIZE, WINDOW_SIZE))
        x, y = pos
        for i in range(-WINDOW_SIZE // 2, WINDOW_SIZE // 2 + 1):
            for j in range(-WINDOW_SIZE // 2, WINDOW_SIZE // 2 + 1):
                if 0 <= x + i < game.config.game.width and 0 <= y + j < game.config.game.height:
                    cell = game.get_board()[y + j][x + i]
                    if cell.get_state() == CellState.REVEALED:
                        state[0][j + WINDOW_SIZE // 2][i + WINDOW_SIZE // 2] = cell.get_value()

        return state
    
    def save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path))
                

if __name__ == "__main__":
    """
    size = 5

    model = CNN(size ** 2, 1, 64)

    array = np.array([[[[0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 2],
                        [0, 0, 1, 1, 0],
                        [0, 0, 0, 0, 0]]]], dtype=np.float32)

    x = torch.tensor(array)
    print(model(x))
    y = np.array([0, 1])
    # training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for _ in tqdm(range(1000)):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, torch.tensor(y, dtype=torch.float32))
        loss.backward()
        optimizer.step()
    print("done")

    # testing
    model.eval()
    with torch.no_grad():
        output = model(x)
        print(output)
        print(np.round(output.numpy()))
    """

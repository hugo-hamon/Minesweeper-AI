import torch.nn.functional as F
import torch.nn as nn
import torch
import os


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

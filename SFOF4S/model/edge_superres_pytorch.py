import torch
import torch.nn as nn

# Define the equivalent PyTorch model
class PyTorchModel(nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        self.conv1 = nn.Conv2d(1,   64,   kernel_size=5, padding="same")
        self.conv2 = nn.Conv2d(64,  64,  kernel_size=3, padding="same")
        self.conv3 = nn.Conv2d(64,  32,  kernel_size=3, padding="same")
        self.conv4 = nn.Conv2d(32,  9,   kernel_size=3, padding="same")
        self.pixel_shuffle = nn.PixelShuffle(3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pixel_shuffle(x)
        return x
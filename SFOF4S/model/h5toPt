import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Define the DepthToSpace layer
class DepthToSpace(layers.Layer):
    def __init__(self, upscale_factor, **kwargs):
        super(DepthToSpace, self).__init__(**kwargs)
        self.upscale_factor = upscale_factor

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.upscale_factor)

# Load TensorFlow model and weights
def load_tf_model(weight_address):
    upscale_factor = 3
    channels = 1
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    inputs = keras.Input(shape=(None, None, channels))
    x = layers.Conv2D(64, 5, **conv_args)(inputs)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(32, 3, **conv_args)(x)
    x = layers.Conv2D(channels * (upscale_factor ** 2), 3, **conv_args)(x)
    outputs = DepthToSpace(upscale_factor=upscale_factor)(x)
    
    model = keras.Model(inputs, outputs)
    model.load_weights(weight_address)
    return model

# Define the equivalent PyTorch model
class PyTorchModel(nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 9, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pixel_shuffle(x)
        return x

# Convert TensorFlow weights to PyTorch
def convert_tf_to_torch(tf_model, torch_model):
    tf_layers = [layer for layer in tf_model.layers if isinstance(layer, layers.Conv2D)]
    torch_layers = [torch_model.conv1, torch_model.conv2, torch_model.conv3, torch_model.conv4]
    
    for tf_layer, torch_layer in zip(tf_layers, torch_layers):
        tf_weights = tf_layer.get_weights()
        torch_layer.weight.data = torch.tensor(np.transpose(tf_weights[0], (3, 2, 0, 1)), dtype=torch.float32)
        torch_layer.bias.data = torch.tensor(tf_weights[1], dtype=torch.float32)
    
    return torch_model

# Main function to execute conversion
def main(tf_weight_path, torch_output_path):
    tf_model = load_tf_model(tf_weight_path)
    torch_model = PyTorchModel()
    torch_model = convert_tf_to_torch(tf_model, torch_model)
    torch.save(torch_model.state_dict(), torch_output_path)
    print(f"Model weights converted and saved to {torch_output_path}")

if __name__ == "__main__":
    # Example usage
    main("path_to_tf_model.h5", "converted_model.pt")

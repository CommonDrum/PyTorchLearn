import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_neural_net(ax, layer_sizes):
    '''
    Draw a neural network cartoon using matplotlib.
    
    :param ax: matplotlib.axes.Axes instance
    :param layer_sizes: list of int, list containing the number of neurons for each layer
    '''
    n_layers = len(layer_sizes)
    v_spacing = 1 / float(max(layer_sizes))
    h_spacing = 1 / float(len(layer_sizes) - 1)
    
    # Nodes
    for n in range(n_layers):
        layer_size = layer_sizes[n]
        layer_top = 1 - (v_spacing * layer_size) / 2
        for m in range(layer_size):
            circle = patches.Circle((n * h_spacing, layer_top - m * v_spacing), v_spacing / 4,
                                    edgecolor='k', facecolor='w', zorder=4)
            ax.add_patch(circle)
    
    # Edges
    for n in range(n_layers - 1):
        layer_size_a = layer_sizes[n]
        layer_size_b = layer_sizes[n + 1]
        layer_top_a = 1 - (v_spacing * layer_size_a) / 2
        layer_top_b = 1 - (v_spacing * layer_size_b) / 2
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = patches.ConnectionPatch(xyA=(n * h_spacing, layer_top_a - m * v_spacing),
                                               xyB=((n + 1) * h_spacing, layer_top_b - o * v_spacing),
                                               coordsA='data', coordsB='data',
                                               axesA=ax, axesB=ax,
                                               color='k', arrowstyle='-', zorder=1)
                ax.add_patch(line)

# Define the PyTorch model
class SimpleNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNeuralNet, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

# Specify the size of each layer
input_size = 10
hidden_size = 5
output_size = 2

# Create the neural network
net = SimpleNeuralNet(input_size, hidden_size, output_size)

# Extract the layer sizes from the PyTorch model
layer_sizes = [input_size] + [hidden_size] + [output_size]

# Create the figure
fig, ax = plt.subplots(figsize=(12, 12))
ax.axis('off')
draw_neural_net(ax, layer_sizes)
plt.show()

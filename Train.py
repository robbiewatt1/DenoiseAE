import torch

from Network import DenoiseAE
from Dataset import EdgeRadDataset


def train(network_layer, network_args, learning_rate, data_loader):
    """
    Training function for de-noising autoencoder.
    :param network_layer: List of network layers
    :param network_args: required parameters for layers
    :param learning_rate: Adam optimiser learning rate
    :param data_loader: Edge radiation data loader
    """

    # Check what device to run on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Setup network / optimiser / loss
    network = DenoiseAE(network_layer, network_args)
    network.to(device)
    loss_func = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(network.parameters(), lr=learning_rate)



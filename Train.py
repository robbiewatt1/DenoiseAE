import matplotlib.pyplot as plt
import torch
from Network import DenoiseAE
from Dataset import EdgeRadDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision

"""
Functions for training and tuning network
"""


def train(network_layer, network_args, learning_rate, data_loader, epochs):
    """
    Training function for de-noising autoencoder.
    :param network_layer: List of network layers
    :param network_args: required parameters for layers
    :param learning_rate: Adam optimiser learning rate
    :param data_loader: Edge radiation data loader
    :param epochs: Number of loops through the data
    """

    # Check what device to run on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup network / optimiser / loss
    network = DenoiseAE(network_layer, network_args)
    network.to(device)
    loss_func = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(network.parameters(), lr=learning_rate)

    writer = SummaryWriter()

    # Start training loop
    losses = []
    for i in range(epochs):
        for j, (image_in, image_out) in enumerate(data_loader):

            # Send data to device
            image_in, image_out = image_in.to(device), image_out.to(device)

            # Calculate loss and take step
            optimiser.zero_grad()
            net_out = network(image_in)
            loss = loss_func(image_out, net_out)
            loss.backward()
            optimiser.step()

            # Save loss
            losses.append(loss.item())
            if (j + 1) % 10 == 0:
                writer.add_scalar("Loss", loss.item(), j)

            # Print image
            if(j + 1) % 10 == 0:
                network.eval()
                net_out = network(image_in[0][None, :, :, :])

                fig, ax = plt.subplots(1, 3, figsize=(12, 3))
                print(net_out[0, 0].detach().cpu().numpy().shape)
                ax[0].pcolor(image_in[0, 0].detach().cpu().numpy(), cmap='jet')
                ax[1].pcolor(image_out[0, 0].detach().cpu().numpy(), cmap='jet')
                ax[2].pcolor(net_out[0, 0].detach().cpu().numpy(), cmap='jet')
                writer.add_figure('Fig1', fig, j)
                network.train()


if __name__ == "__main__":

    # Set dataloader
    dataset = EdgeRadDataset(
        "/home/robbie/Documents/Edge-Radiation/ML/Data/denoise-data.h5",
        35.67, 25)
    loader = DataLoader(dataset, batch_size=1, shuffle=True,
                        num_workers=8)

    layers = ['Conv', 'Conv', 'Pool', 'Conv', 'Conv',  'Pool', 'InvConv',
              'InvConv', 'InvConv', 'Output']
    layer_params = [[1, 64, 5], [64, 64, 5], [2], [64, 64, 5], [64, 64, 5], [2],
                    [64, 64, 5], [64, 64, 5], [64, 64, 5],  [64, 1, 5]]


    train(layers, layer_params, 1e-3, loader, 1)

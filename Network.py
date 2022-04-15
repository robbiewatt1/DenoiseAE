import torch
from torchsummary import summary

"""
File containing neural network classes and sub blocks
"""


class ConvBlock(torch.nn.Module):
    """
    Convolutional block consisting of 2d-convolution, relu activation and
    batch-normalisation. Padding is set to maintain same input shape.
    """
    def __init__(self, block_params, activation=True, batch_norm=True):
        """
        :param block_params: List of block parameter [filters_in,
            filters_out, filter_shape]
        """
        super(ConvBlock, self).__init__()
        layers = [torch.nn.Conv2d(block_params[0], block_params[1],
                                  block_params[2], padding='same')]
        if activation:
            layers.append(torch.nn.ReLU())
        if batch_norm:
            layers.append(torch.nn.BatchNorm2d(block_params[1]))
        self.block = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: input tensor
        :return: output tensor
        """
        return self.block(x)


class InvConvBlock(torch.nn.Module):
    """
    Inverse convolutional block consisting of 2d transpose convolution, relu
    activation and batch-normalisation. Padding is set to maintain same input
    shape.
    """
    def __init__(self, block_params, activation=True, batch_norm=True):
        """
        :param block_params: List of block parameter (filters_in,
            filters_out, filter_shape)
        """
        super(InvConvBlock, self).__init__()
        padding = int((block_params[2] - 1) / 2)
        layers = [torch.nn.ConvTranspose2d(block_params[0], block_params[1],
                                           block_params[2], padding=padding)]
        if activation:
            layers.append(torch.nn.ReLU())
        if batch_norm:
            layers.append(torch.nn.BatchNorm2d(block_params[1]))
        self.block = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: input tensor
        :return: output tensor
        """
        return self.block(x)


class DenoiseAE(torch.nn.Module):
    """
    Convolutional autoencoder for de-noising edge radiation image.
    """
    def __init__(self, layers, layer_params):
        """
        :param layers: List of layers
        :param layer_params: list of layer params
        """
        super(DenoiseAE, self).__init__()
        layer_list = []
        for i, layer in enumerate(layers):
            print(i, layer)
            if layer == "Conv":
                layer_list.append(ConvBlock(layer_params[i]))
            elif layer == "InvConv":
                layer_list.append(InvConvBlock(layer_params[i]))
            elif layer == "Pool":
                layer_list.append(torch.nn.MaxPool2d(layer_params[i]))
            elif layer == "Output":
                layer_list.append(InvConvBlock(layer_params[i], False, False))
        self.network = torch.nn.Sequential(*layer_list)

    def forward(self, x):
        """
        :param x: input tensor
        :return: output tensor
        """
        return self.network(x)

    def summary(self, shape):
        """
        Prints a summary of the network
        :param shape: Shape of input tensor
        """
        print(summary(self, shape))

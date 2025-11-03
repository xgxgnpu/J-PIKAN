import torch
import torch.nn as nn
import numpy as np

class FourierKANLayer(torch.nn.Module):
    """
    Implements a single layer of the KAN network using Fourier basis functions.
    """

    def __init__(
        self, inputdim, outdim, gridsize, addbias=True, smooth_initialization=False
    ):
        """
        Initializes the Fourier-based KAN layer.

        Args:
            inputdim (int): Number of input features.
            outdim (int): Number of output features.
            gridsize (int): Number of Fourier coefficients.
            addbias (bool): Whether to include a bias term.
            smooth_initialization (bool): If True, scales initialization to emphasize lower frequencies.
        """
        super(FourierKANLayer, self).__init__()
        self.gridsize = gridsize  # Number of Fourier modes
        self.addbias = addbias  # Flag to add bias
        self.inputdim = inputdim  # Number of input dimensions
        self.outdim = outdim  # Number of output dimensions

        # Normalization factor for initialization
        grid_norm_factor = (
            (torch.arange(gridsize) + 1) ** 2
            if smooth_initialization
            else np.sqrt(gridsize)
        )

        # Fourier coefficients for sine and cosine terms
        self.fouriercoeffs = torch.nn.Parameter(
            torch.randn(2, outdim, inputdim, gridsize)
            / (np.sqrt(inputdim) * grid_norm_factor)
        )

        # Optional bias term for each output dimension
        if self.addbias:
            self.bias = torch.nn.Parameter(torch.zeros(1, outdim))

    def forward(self, x):
        """
        Forward pass through the layer.

        Args:
            x (torch.Tensor): Input tensor of shape (..., inputdim).

        Returns:
            torch.Tensor: Output tensor of shape (..., outdim).
        """
        xshp = x.shape  # Save original shape of input
        outshape = xshp[0:-1] + (self.outdim,)  # Output shape after transformation
        x = torch.reshape(x, (-1, self.inputdim))  # Flatten the input for processing

        # Generate wave numbers for Fourier modes
        k = torch.reshape(
            torch.arange(1, self.gridsize + 1, device=x.device),
            (1, 1, 1, self.gridsize)
        )
        # Reshape input for element-wise multiplication with Fourier coefficients
        xrshp = torch.reshape(x, (x.shape[0], 1, x.shape[1], 1))

        # Compute cosine and sine components
        c = torch.cos(k * xrshp)  # Cosine basis functions
        s = torch.sin(k * xrshp)  # Sine basis functions

        # Compute the sum of Fourier interpolations
        y = torch.sum(c * self.fouriercoeffs[0:1], (-2, -1))  # Cosine contributions
        y += torch.sum(s * self.fouriercoeffs[1:2], (-2, -1))  # Sine contributions

        # Add bias if enabled
        if self.addbias:
            y += self.bias

        # Reshape output to match the expected output shape
        y = torch.reshape(y, outshape)
        return y


class FourierPINN(nn.Module):
    """
    Implements a Physics-Informed Neural Network (PINN) using NaiveFourierKANLayer.
    """

    def __init__(self, network, degree=4):
        """
        Initializes the PINNKAN network.

        Args:
            network (list[int]): List defining the number of nodes in each layer.
            activation (nn.Module): Activation function (currently unused).
            degree (int): Number of Fourier coefficients for each layer.
        """
        super().__init__()
        self.network = network  # Network architecture
        self.layers = nn.ModuleList()  # List of layers

        # Define layers based on the network architecture
        for i in range(len(network) - 1):
            self.layers.append(FourierKANLayer(network[i], network[i + 1], degree))

    def forward(self, x):
        """
        Forward pass through the PINNKAN network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        x = x.view(-1, self.network[0])  # 将输入展平
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
        return x

import torch
import torch.nn as nn


class ProteinCNN(nn.Module):
    """
    A 1D convolutional module to capture sequential residue-level patterns
    from GNN-generated node embeddings.

    This module performs two stacked 1D convolutions (customizable dimensions),
    followed by ReLU and Dropout, matching the design from GraphPBSP.
    """

    def __init__(
        self,
        in_channels: int,
        cnn_channels: int,
        cnn_kernel_size: int,
        cnn_dropout: float,
    ):
        """
        Initialize the ProteinCNN.

        Args:
            in_channels (int): Input channels (dimensionality of GNN embeddings).
            cnn_channels (int): Output channels after convolution.
            cnn_kernel_size (int): Size of the convolutional filter.
            cnn_dropout (float): Dropout rate applied after first convolution.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                cnn_channels,
                kernel_size=cnn_kernel_size,
                padding=cnn_kernel_size // 2,
            ),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(cnn_dropout),
            nn.Conv1d(
                cnn_channels,
                cnn_channels,
                kernel_size=cnn_kernel_size,
                padding=cnn_kernel_size // 2,
            ),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply 1D convolution to a sequence of node embeddings.

        Args:
            x (Tensor): Input tensor of shape [1, in_channels, seq_len].

        Returns:
            Tensor: Output tensor of shape [1, cnn_channels, seq_len].
        """
        return self.net(x)

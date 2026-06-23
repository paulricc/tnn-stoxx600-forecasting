import torch
import torch.nn as nn


class TimeAttention(nn.Module):
    """Time attention mechanism that assigns weights to each time step.

    Attributes:
        attention_fc: Small neural network that generates attention scores.
        softmax: Normalizes attention scores so they sum to 1.
    """

    def __init__(self, input_size: int):
        """Initialize the time attention module.

        Args:
            input_size: Number of features per time step (kernel filter output size).
        """
        super().__init__()
        self.attention_fc = nn.Linear(input_size, input_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply time attention to a sequence of features.

        Args:
            features: Tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            Weighted features of shape (batch_size, sequence_length, input_size).
        """
        attention_scores = self.attention_fc(features)
        attention_weights = self.softmax(attention_scores)
        weighted_features = attention_weights * features
        return weighted_features


class TNN(nn.Module):
    """Time-series Neural Network combining a Kernel Filter and Time Attention.

    Attributes:
        kernel_filter: 1D convolution that extracts local temporal patterns.
        time_attention: Assigns importance weights to each time step.
        fc1: First fully connected layer.
        fc2: Output fully connected layer.
        dropout: Dropout layer for regularization.
    """

    def __init__(
        self,
        input_size: int,
        kernel_output_size: int,
        kernel_size: int,
        hidden_size: int,
        dropout: float = 0.2,
    ):
        """Initialize the TNN model.

        Args:
            input_size: Number of input features per time step.
            kernel_output_size: Number of output channels of the kernel filter.
            kernel_size: Size of the convolution kernel.
            hidden_size: Hidden size of the first fully connected layer.
            dropout: Dropout probability.
        """
        super().__init__()
        self.kernel_filter = nn.Conv1d(
            in_channels=input_size,
            out_channels=kernel_output_size,
            kernel_size=kernel_size,
        )
        self.time_attention = TimeAttention(kernel_output_size)
        self.fc1 = nn.Linear(kernel_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            Predicted values of shape (batch_size, 1).
        """
        x = x.permute(0, 2, 1)
        x = self.kernel_filter(x)
        x = torch.relu(x)

        x = x.permute(0, 2, 1)
        x = self.time_attention(x)
        x = torch.sum(x, dim=1)

        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

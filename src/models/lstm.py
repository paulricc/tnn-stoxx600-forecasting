import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """LSTM-based model for stock price forecasting.

    Attributes:
        lstm: The LSTM layer that processes the input sequence.
        fc: The fully connected layer that produces the final prediction.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        """Initialize the LSTM model.

        Args:
            input_size: Number of input features per time step.
            hidden_size: Number of hidden units in the LSTM.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout probability between LSTM layers.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            Predicted values of shape (batch_size, 1).
        """
        output, _ = self.lstm(x)
        last_step = output[:, -1, :]
        prediction = self.fc(last_step)
        return prediction

"""
model.py — Two-layer stacked LSTM for binary contact detection.

Architecture:
    LSTM (2 layers, hidden=128, dropout=0.3)
    → take last timestep hidden state
    → Linear(128 → 1)
    → (sigmoid applied at inference; BCEWithLogitsLoss used at training)
"""

import torch
import torch.nn as nn


class ContactLSTM(nn.Module):
    """
    Stacked 2-layer LSTM binary classifier for robot contact detection.

    Parameters
    ----------
    input_size  : number of telemetry features per timestep
    hidden_size : LSTM hidden dimension (default 128)
    num_layers  : number of stacked LSTM layers (default 2)
    dropout     : dropout probability between LSTM layers (default 0.3)
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Additional dropout before the classifier head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_size)

        Returns
        -------
        logits : (batch,)   raw (pre-sigmoid) scores
        """
        # lstm_out: (batch, seq, hidden)
        lstm_out, _ = self.lstm(x)

        # Use the last timestep's output as the sequence representation
        last_hidden = lstm_out[:, -1, :]           # (batch, hidden)
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden).squeeze(-1)  # (batch,)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities (inference only)."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    @staticmethod
    def from_checkpoint(path: str, device=None):
        """Load a saved checkpoint produced by train.py."""
        device = device or torch.device("cpu")
        ckpt = torch.load(path, map_location=device)
        model = ContactLSTM(
            input_size=ckpt["input_size"],
            hidden_size=ckpt.get("hidden_size", 128),
            num_layers=ckpt.get("num_layers", 2),
            dropout=ckpt.get("dropout", 0.3),
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()
        return model, ckpt

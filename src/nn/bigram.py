"""Basline models for the language modeling task."""

import torch
from torch import nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    """Bigram language model."""

    def __init__(self, vocab_size):
        super().__init__()
        # Each token reads the logits for the next token from the lookup table
        self.lookup = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """Forward pass of the model."""
        # idx and targets are both of shape (batch_size, sequence_length)
        logits = self.lookup(idx)

        if targets is None:
            loss = None
        else:
            # (B, T, C) (batch_size, sequence_length, vocab_size)
            b, t, c = logits.shape  # Shape (32, 8, 527)
            logits_ = logits.view(b * t, c)  # Reshape to (256, 527)
            targets_ = targets.view(-1)  # Reshape from (32, 8) to (256)
            loss = F.cross_entropy(logits_, targets_)
        return logits, loss

    def generate(self, idx, length):
        """Generate text using the model."""
        with torch.no_grad():
            for _ in range(length):
                # Get the predictions for the all the tokens
                logits, _ = self.forward(idx)  # (B, T, C)
                # Get the last token
                logits = logits[:, -1, :]  # (B, C)
                # Apply softmax to get the probabilities
                probs = F.softmax(logits, dim=-1)  # (B, C)
                # Sample the next token
                next_token = torch.multinomial(probs, 1)  # (B, 1)
                # Append the next token to the sequence
                idx = torch.cat([idx, next_token], dim=-1)  # (B, T+1)
        return idx

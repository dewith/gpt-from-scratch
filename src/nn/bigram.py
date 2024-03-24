"""Basline models for the language modeling task."""

import torch
from torch import nn
from torch.nn import functional as F
from torchviz import make_dot


class BigramLanguageModel(nn.Module):
    """Bigram language model."""

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
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

    def viz(self):
        """Visualize the model."""
        x = torch.randint(self.vocab_size, (1, 8))
        y = torch.randint(self.vocab_size, (1, 8))
        viz = make_dot(
            self(x, y)[1],
            params=dict(self.named_parameters()),
            show_attrs=False,
            show_saved=False,
        )
        return viz


class BigramLanguageModelV2(nn.Module):
    """Bigram language model."""

    def __init__(self, vocab_size, num_embeds=64, block_size=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_embeds = num_embeds
        self.block_size = block_size
        # Each token reads the logits for the next token from the lookup table
        self.token_embed_table = nn.Embedding(vocab_size, num_embeds)
        self.position_embed_table = nn.Embedding(block_size, num_embeds)
        self.linear_head = nn.Linear(num_embeds, vocab_size)

    def forward(self, idx, targets=None):
        """Forward pass of the model."""
        b, t = idx.shape

        # idx and targets are both of shape (batch_size, sequence_length)
        tok_emb = self.token_embed_table(idx)  # (B, T, E)
        pos_emb = self.position_embed_table(torch.arange(t).to(idx.device))  # (T, E)
        x = tok_emb + pos_emb  # (B, T, E)
        logits = self.linear_head(x)

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
        generated_sequence = idx.clone()
        with torch.no_grad():
            for _ in range(length):
                # Ensure idx does not exceed the block size
                idx = idx[:, -self.block_size :]

                # Get the predictions for the current set of tokens
                logits, _ = self.forward(idx)  # (B, T, C)
                logits = logits[:, -1, :]  # Get the last token's logits (B, C)
                probs = F.softmax(logits, dim=-1)  # (B, C)
                next_token = torch.multinomial(probs, 1)  # (B, 1)

                # Append the next token to the generated sequence (B, T+1)
                generated_sequence = torch.cat(
                    tensors=[generated_sequence, next_token], dim=-1
                )
                # Append the next token to idx for the next iteration
                idx = torch.cat([idx, next_token], dim=-1)
        return generated_sequence

    def viz(self):
        """Visualize the model."""
        x = torch.randint(self.vocab_size, (1, self.block_size))
        y = torch.randint(self.vocab_size, (1, self.block_size))
        viz = make_dot(
            self(x, y)[1],
            params=dict(self.named_parameters()),
            show_attrs=False,
            show_saved=False,
        )
        return viz

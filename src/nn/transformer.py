"""Transformer model for language modeling."""

import torch
from torch import nn
from torch.nn import functional as F
from torchviz import make_dot


class Head(nn.Module):
    """Attetion head."""

    # pylint: disable=too-few-public-methods

    def __init__(self, block_size=8, num_embeds=64, head_size=8):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(num_embeds, head_size, bias=False)
        self.query = nn.Linear(num_embeds, head_size, bias=False)
        self.value = nn.Linear(num_embeds, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        """Forward pass of the model."""
        _, t, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # Compute the attention scores
        scores = (q @ k.transpose(-2, -1)) / (self.head_size**0.5)
        scores_masked = scores.masked_fill(self.tril[:t, :t] == 0, float("-inf"))
        weights = torch.softmax(scores_masked, dim=-1)

        # Compute the weighted sum of the values
        output = weights @ v
        return output


class MultiHeadAttention(nn.Module):
    """Multiple heads of attention."""

    # pylint: disable=too-few-public-methods

    def __init__(self, num_heads, block_size, num_embeds, head_size):
        super().__init__()
        self.heads = [Head(block_size, num_embeds, head_size) for _ in range(num_heads)]

    def forward(self, x):
        """Forward pass of the model."""
        return torch.cat([head(x) for head in self.heads], dim=-1)


class FeedForward(nn.Module):
    """Feed forward network for the transformer."""

    # pylint: disable=too-few-public-methods

    def __init__(self, num_embeds):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embeds, num_embeds),
            nn.ReLU(),
        )

    def forward(self, x):
        """Forward pass of the model."""
        return self.net(x)


class Transformer(nn.Module):
    """Transformer model for language modeling."""

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments

    def __init__(
        self,
        vocab_size: int,
        num_embeds: int = 32,
        block_size: int = 8,
        num_heads: int = 4,
        head_size: int = 8,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_embeds = num_embeds
        self.block_size = block_size
        self.head_size = head_size
        # Each token reads the logits for the next token from the lookup table
        self.token_embed_table = nn.Embedding(vocab_size, num_embeds)
        self.position_embed_table = nn.Embedding(block_size, num_embeds)
        self.attention_heads = MultiHeadAttention(
            num_heads, block_size, num_embeds, head_size
        )
        self.feed_forward = FeedForward(num_embeds)
        self.linear_head = nn.Linear(num_embeds, vocab_size)

    def forward(self, idx, targets=None):
        """Forward pass of the model."""
        b, t = idx.shape

        # idx and targets are both of shape (batch_size, sequence_length)
        tok_emb = self.token_embed_table(idx)  # (B, T, E)
        pos_emb = self.position_embed_table(torch.arange(t).to(idx.device))  # (T, E)
        x = tok_emb + pos_emb  # (B, T, E)
        x_att = self.attention_heads(x)  # (B, T, E)
        x_ffwd = self.feed_forward(x_att)  # (B, T, E)
        logits = self.linear_head(x_ffwd)

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

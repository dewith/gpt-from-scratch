"""
This script trains a bigram language model using a small dataset.
The model is trained using a small dataset and then used to generate text.

Date created: 2024-03-05
"""

import torch
from torch import nn
from torch.nn import functional as F

from src.utils.logging import get_logger
from src.utils.data import get_corpus_text, get_data_split, Tokenizer, Batcher
from src.utils.evaluation import estimate_loss


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


def main():
    """Main function for training the bigram language model."""
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements
    # Data
    LOGGER.info("├── Loading the dataset")
    dataset_path = "dewithsan/secop_corpus_clean"
    corpus = get_corpus_text(dataset_path)
    LOGGER.info("│   ├── Downloaded from %s", dataset_path)
    tokenizer = Tokenizer(corpus)
    train_data, val_data = get_data_split(corpus, tokenizer)
    LOGGER.info("│   └── Data tokenized and splitted")
    LOGGER.info("│")

    # Hyperparameters
    batch_size = 32
    block_size = 8
    learning_rate = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_steps = 20000
    step_loss_interval = 200
    eval_interval = 1000
    eval_iters = 200

    # Model definition
    LOGGER.info("├── Defining the model")
    vocab_size = tokenizer.vocab_size
    model = BigramLanguageModel(vocab_size)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    batcher = Batcher(train_data, val_data, batch_size, block_size)
    LOGGER.info("│   ├── Model created with vocab of %s", vocab_size)
    LOGGER.info("│   └── Batcher initialized with size %s", batch_size)
    LOGGER.info("│")

    # Model training
    LOGGER.info("├── Training the model with %s steps", max_steps)
    for step in range(max_steps + 1):
        # Forward pass
        xb, yb = batcher.get_batch("train")
        xb, yb = xb.to(device), yb.to(device)
        _, loss = model(xb, yb)

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % step_loss_interval == 0:
            loss_str = f"{loss.item():.4f}"
            LOGGER.info("│   ├── Step %s ~ Loss: %s", step, loss_str)

        # Evaluate the model and log the losses
        if step % eval_interval == 0:
            losses = estimate_loss(model, batcher, eval_iters)
            train_loss = f"{losses['train']:.4f}"
            val_loss = f"{losses['val']:.4f}"
            LOGGER.info("│   │   ├── Train loss: %s", train_loss)
            LOGGER.info("│   │   └── Val loss:   %s", val_loss)

    LOGGER.info("│   └── Model training completed")
    LOGGER.info("│")

    LOGGER.info("├── Generating text")
    context = torch.randint(tokenizer.vocab_size, (1, 1)).to(device)
    generated_tokens = model.generate(context, 80)
    generated_text = tokenizer.decode(generated_tokens[0].tolist())
    LOGGER.info("│   ├── Text generated")
    LOGGER.info("│   └── %s", generated_text.replace("\n", " "))
    LOGGER.info("│")

    LOGGER.info("└── Saving the model")
    local_path = "data/04_models/bigram_model.pth"
    torch.save(model.state_dict(), local_path)
    LOGGER.info("    └── Model saved at %s", local_path)


if __name__ == "__main__":
    LOGGER = get_logger()
    LOGGER.info("Training the Bigram Language Model")
    main()

"""
This script trains a bigram language model using a small dataset.
The model is trained using a small dataset and then used to generate text.

Date created: 2024-03-05
"""

import logging
import os
import re
import sys

import torch
from torch import nn
from torch.nn import functional as F
from datasets import load_dataset
from unidecode import unidecode


def get_logger():
    """Get the logger for the script."""
    file_name = os.path.basename(sys.argv[0]).replace(".py", "")
    file_handler = logging.FileHandler(filename=f"logs/{file_name}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    format_ = "%(asctime)s L%(lineno)s %(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=format_, handlers=handlers)
    logger = logging.getLogger(file_name)
    return logger


def get_corpus_text(dataset_path):
    """Get the corpus text"""
    dataset = load_dataset(dataset_path)
    corpus_df = dataset["train"].to_pandas()
    corpus_text = "\n".join(corpus_df["doc_text"])

    corpus_text_clean = corpus_text.replace("\n", " ").replace("\r", " ")
    corpus_text_clean = re.sub(r" +", " ", corpus_text_clean)
    pat = r'[^\w\s!"·$%&/()=?¿\\|@#+,\.-^\*;:_\[\]\{\} !¡¿?,\.@#$%^&\*]'
    corpus_text_clean = re.sub(pat, "", corpus_text_clean)
    corpus_text_clean = corpus_text_clean.lower()
    corpus_text_clean = unidecode(corpus_text_clean)
    return corpus_text_clean


def get_data_split(corpus_text, tokenizer):
    """Get the data split into training and validation sets."""
    data = torch.tensor(tokenizer.encode(corpus_text), dtype=torch.long)
    train_size = int(len(data) * 0.90)
    train_data = data[:train_size]
    val_data = data[train_size:]
    return train_data, val_data


class Tokenizer:
    """Tokenizer class for encoding and decoding text."""

    def __init__(self, corpus_text):
        stoi, itos = self._get_token_maps(corpus_text)
        self.stoi = stoi
        self.itos = itos
        self.vocab_size = len(stoi)

    def encode(self, text: str) -> list:
        """Encode text to integers."""
        return [self.stoi.get(char, self.stoi["[UNK]"]) for char in text]

    def decode(self, integers: list) -> str:
        """Decode list of integers to text."""
        return "".join([self.itos[i] for i in integers])

    def _get_token_maps(self, corpus_text):
        chars = sorted(list(set(corpus_text)))
        itos = dict(enumerate(chars, start=0))
        itos[max(itos) + 1] = "[UNK]"
        stoi = {char: i for i, char in enumerate(chars, start=0)}
        stoi["[UNK]"] = max(stoi)
        return stoi, itos


class Batcher:
    """Batcher class for generating batches of data."""

    # pylint: disable=too-few-public-methods

    def __init__(self, train_data, val_data, batch_size, block_size):
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.block_size = block_size

    def get_batch(self, split):
        """Generates a small batch of data of inputs x and targets y"""
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])
        return x, y


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


@torch.no_grad()
def estimate_loss(model, batcher, eval_iters):
    """Estimate the loss of the model on the training and validation sets."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = batcher.get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


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

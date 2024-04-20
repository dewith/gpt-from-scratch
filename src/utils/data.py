"""
This module provides utility functions for working with data.
The `get_corpus_text` function is used to get the corpus text from a dataset.
The `get_data_split` function is used to split the data into train and val sets.
The `Tokenizer` class is used to encode and decode text.
The `Batcher` class is used to generate batches of data.
"""

import re
import logging

import pandas as pd
import torch
from datasets import load_dataset
from unidecode import unidecode

LOGGER = logging.getLogger(__name__)


def get_corpus_text(local_path=None, hf_path=None, is_clean=False):
    """Get the corpus text"""
    if not is_clean:
        if local_path is None and hf_path is None:
            raise ValueError("At least one of local or hf path must be passed.")

        if local_path is not None and hf_path is None:
            corpus_df = pd.read_csv(local_path)
            path = local_path
        elif local_path is None and hf_path is not None:
            dataset = load_dataset(hf_path)
            corpus_df = dataset["train"].to_pandas()
            path = hf_path
        else:
            try:
                corpus_df = pd.read_csv(local_path)
                path = local_path
            except Exception as e:  # pylint: disable=broad-except
                LOGGER.warning("│   ├── Local path raised exception:")
                LOGGER.error("│   ├── \t%s", e)
                LOGGER.warning("│   ├── Loading from Hugging Face dataset.")
                dataset = load_dataset(hf_path)
                corpus_df = dataset["train"].to_pandas()
                path = hf_path
        LOGGER.info("│   ├── Loaded from %s", path)

        corpus_text = "\n".join(corpus_df["doc_text"])
        corpus_text_clean = corpus_text.replace("\n", " ").replace("\r", " ")
        corpus_text_clean = re.sub(r" +", " ", corpus_text_clean)
        pat = r'[^\w\s!"·$%&/()=?¿\\|@#+,\.-^\*;:_\[\]\{\} !¡¿?,\.@#$%^&\*]'
        corpus_text_clean = re.sub(pat, "", corpus_text_clean)
        corpus_text_clean = corpus_text_clean.lower()
        corpus_text_clean = unidecode(corpus_text_clean)
        LOGGER.info("│   ├── Corpues cleaned and normalized")

        with open("data/02_primary/corpus.txt", "w", encoding="utf-8") as f:
            f.write(corpus_text_clean)
    else:
        with open(local_path, "r", encoding="utf-8") as f:
            corpus_text_clean = f.read()
        LOGGER.info("│   ├── Loaded already clean corpus from %s", local_path)

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
    # pylint: disable=too-many-arguments

    def __init__(self, train_data, val_data, batch_size, block_size, device="cpu"):
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device

    def get_batch(self, split):
        """Generates a small batch of data of inputs x and targets y"""
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])
        return x.to(self.device), y.to(self.device)

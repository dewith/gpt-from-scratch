"""
This script trains a transformer model using a small dataset.

Date created: 2024-03-31
"""

# pylint: disable=duplicate-code
import time
import pickle
import torch

from src.utils.logging import get_logger
from src.utils.data import get_corpus_text, get_data_split, Tokenizer, Batcher
from src.nn.transformer import Transformer
from src.nn.evaluation import estimate_loss


def main():
    """Main function for training the bigram language model."""
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements
    LOGGER.info("Training the transformer model with self attention")

    # Data
    LOGGER.info("├── Loading the dataset")
    try:
        dataset_local_path = "data/02_primary/corpus.txt"
        corpus = get_corpus_text(dataset_local_path, is_clean=True)
    except FileNotFoundError:
        dataset_local_path = "data/01_raw/secop_corpus.csv"
        dataset_hf_path = "dewithsan/secop_corpus_clean"
        corpus = get_corpus_text(dataset_local_path, dataset_hf_path)

    tokenizer_path = "data/04_models/transformer_tokenizer.pkl"
    try:
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
        LOGGER.info("│   ├── Tokenizer loaded from file")
    except FileNotFoundError:
        tokenizer = Tokenizer(corpus)
        with open(tokenizer_path, "wb") as f:
            pickle.dump(tokenizer, f)
        LOGGER.info("│   ├── Tokenizer created and saved at %s", tokenizer_path)

    train_data, val_data = get_data_split(corpus, tokenizer)
    LOGGER.info("│   └── Data tokenized and train-val splitted")
    LOGGER.info("│")

    # Hyperparameters
    LOGGER.info("├── Defining hyperparameters")
    batch_size = 64
    block_size = 256
    num_embeds = 384
    num_heads = 6
    head_size = num_embeds // num_heads
    num_layers = 6
    dropout = 0.2
    learning_rate = 3e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_steps = 25000
    step_loss_interval = 100
    eval_interval = 500
    eval_iters = 100

    # Model definition
    LOGGER.info("├── Defining the model")
    vocab_size = tokenizer.vocab_size
    model = Transformer(
        vocab_size,
        num_embeds,
        block_size,
        num_heads,
        head_size,
        num_layers,
        dropout,
        device,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    batcher = Batcher(train_data, val_data, batch_size, block_size, device)
    LOGGER.info("│   ├── Model created with vocab of %s", vocab_size)
    LOGGER.info("│   ├── Optimizer: AdamW with lr %s", learning_rate)
    LOGGER.info("│   ├── Device: %s", device)
    LOGGER.info("│   └── Batcher size: %s", batch_size)
    LOGGER.info("│")

    LOGGER.info("├── Visualizing the model")
    viz_path = "data/04_models/transformer.png"
    x_viz, y_viz = batcher.get_batch("train")
    x_viz, y_viz = x_viz.to(device), y_viz.to(device)
    model_viz = model.viz(x_viz, y_viz)
    model_viz.render(
        filename=viz_path.rsplit(".", maxsplit=1)[0],
        format=viz_path.rsplit(".", maxsplit=1)[-1],
        cleanup=True,
    )
    LOGGER.info("│   └── Model visualization saved at %s", viz_path)

    # Model training
    LOGGER.info("├── Training the model with %s steps", max_steps)
    time_start = time.time()
    for step in range(max_steps + 1):
        # Forward pass
        xb, yb = batcher.get_batch("train")
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

    time_end = time.time()
    time_elapsed = (time_end - time_start) / 60
    LOGGER.info("│   └── Model training completed in %s minutes.", time_elapsed)
    LOGGER.info("│")

    LOGGER.info("├── Generating text for testing")
    random_start = False
    if random_start:
        context = torch.randint(tokenizer.vocab_size, (1, 1)).to(device)
    else:
        start_token = tokenizer.encode("La")
        context = torch.tensor([start_token]).to(device)
    generated_tokens = model.generate(context, 200)
    generated_text = tokenizer.decode(generated_tokens[0].tolist())
    LOGGER.info("│   └── %s", generated_text.replace("\n", " "))
    LOGGER.info("│")

    LOGGER.info("└── Saving the model")
    local_path = "data/04_models/transformer_model.pth"
    torch.save(model.state_dict(), local_path)
    LOGGER.info("    └── Model saved at %s", local_path)


if __name__ == "__main__":
    LOGGER = get_logger()
    main()

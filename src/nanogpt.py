"""
This script trains a transformer model using a small dataset.

Date created: 2024-03-31
"""

# pylint: disable=duplicate-code
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
    tokenizer = Tokenizer(corpus)
    train_data, val_data = get_data_split(corpus, tokenizer)
    LOGGER.info("│   └── Data tokenized and train/val splitted")
    LOGGER.info("│")

    # Hyperparameters
    LOGGER.info("├── Defining hyperparameters")
    batch_size = 16
    block_size = 8
    num_embeds = 32
    num_heads = 4
    head_size = num_embeds // num_heads
    learning_rate = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_steps = 10000
    step_loss_interval = 200
    eval_interval = 1000
    eval_iters = 200

    # Model definition
    LOGGER.info("├── Defining the model")
    vocab_size = tokenizer.vocab_size
    model = Transformer(vocab_size, num_embeds, block_size, num_heads, head_size)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    batcher = Batcher(train_data, val_data, batch_size, block_size)
    LOGGER.info("│   ├── Model created with vocab of %s", vocab_size)
    LOGGER.info("│   ├── Optimizer: AdamW with lr %s", learning_rate)
    LOGGER.info("│   ├── Device: %s", device)
    LOGGER.info("│   └── Batcher size: %s", batch_size)
    LOGGER.info("│")

    LOGGER.info("├── Visualizing the model")
    viz_path = "data/04_models/transformer.png"
    model_viz = model.viz()
    model_viz.render(
        filename=viz_path.rsplit(".", maxsplit=1)[0],
        format=viz_path.rsplit(".", maxsplit=1)[-1],
        cleanup=True,
    )
    LOGGER.info("│   └── Model visualization saved at %s", viz_path)

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
    local_path = "data/04_models/transformer_model.pth"
    torch.save(model.state_dict(), local_path)
    LOGGER.info("    └── Model saved at %s", local_path)


if __name__ == "__main__":
    LOGGER = get_logger()
    main()

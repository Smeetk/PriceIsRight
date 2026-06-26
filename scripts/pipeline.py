"""
pipeline.py — Full end-to-end pipeline:
    1. Load raw Amazon product data
    2. Assign IDs and split into train / val / test
    3. Summarise items via OpenAI Batch API
    4. Build training prompts and push to HuggingFace Hub
    5. Train the DeepNeuralNetwork price predictor
    6. Run a quick inference demo
"""

import random
from pathlib import Path
from pricer import Item, ItemLoader, Batch, DeepNeuralNetworkRunner

# ── Configuration ─────────────────────────────────────────────────────────── #

CATEGORIES = [
    "Toys_and_Games",      # ~100k good for a quick test run
    "Musical_Instruments", # ~50k
]

DATASET_NAME = "Smitkumarmistry/amazon-pricer" 
MODEL_PATH = Path("pricer_model.pt")

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
# TEST_FRAC implied as the remainder

SEED = 42

# ── Step 1: Load ──────────────────────────────────────────────────────────── #

def load_items() -> list[Item]:
    all_items: list[Item] = []
    for category in CATEGORIES:
        loader = ItemLoader(category)
        all_items.extend(loader.load())
    print(f"\nTotal items loaded: {len(all_items):,}")
    return all_items


# ── Step 2: Assign IDs and split ──────────────────────────────────────────── #

def split_items(items: list[Item]) -> tuple[list[Item], list[Item], list[Item]]:
    random.seed(SEED)
    random.shuffle(items)

    for i, item in enumerate(items):
        item.id = i

    n = len(items)
    train_end = int(n * TRAIN_FRAC)
    val_end = train_end + int(n * VAL_FRAC)

    train = items[:train_end]
    val = items[train_end:val_end]
    test = items[val_end:]

    print(f"Split — train: {len(train):,}  val: {len(val):,}  test: {len(test):,}")
    return train, val, test


# ── Step 3: Batch summarise ───────────────────────────────────────────────── #

def summarise(items: list[Item], lite: bool = False):
    """Submit all items to the OpenAI Batch API and poll until complete."""
    Batch.create(items, lite=lite)
    Batch.run()

    state_file = Path("batches.pkl")
    Batch.save(state_file)

    print("Waiting for batches to complete…")
    while True:
        Batch.fetch()
        pending = [b for b in Batch.batches if not b.done]
        if not pending:
            break
        print(f"{len(pending)} batches still pending — checking again in 60s…")
        import time; time.sleep(60)

    print("All batches complete.")


# ── Step 4: Build prompts and push to Hub ────────────────────────────────── #

def build_and_push(
    train: list[Item],
    val: list[Item],
    test: list[Item],
    dataset_name: str,
):
    for split in (train, val, test):
        for item in split:
            if item.summary:
                item.make_prompt(item.summary)

    Item.push_to_hub(dataset_name, train, val, test)
    print(f"Dataset pushed to Hub: {dataset_name}")


# ── Step 5: Train ─────────────────────────────────────────────────────────── #

def train_model(train: list[Item], val: list[Item]) -> DeepNeuralNetworkRunner:
    runner = DeepNeuralNetworkRunner(train, val)
    runner.setup()
    runner.train(epochs=5)
    runner.save(MODEL_PATH)
    return runner


# ── Step 6: Demo inference ───────────────────────────────────────────────── #

def demo(runner: DeepNeuralNetworkRunner, test: list[Item], n: int = 5):
    print("\n── Inference demo ──")
    for item in random.sample(test, n):
        pred = runner.inference(item)
        print(f"{item!r}  →  predicted ${pred:.2f}")


# ── Main ──────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    items = load_items()
    train, val, test = split_items(items)

    summarise(items, lite=False)

    build_and_push(train, val, test, DATASET_NAME)

    runner = train_model(train, val)
    demo(runner, test)

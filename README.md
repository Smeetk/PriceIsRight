# Pricer — Amazon Product Price Predictor

An end-to-end ML pipeline that:

1. **Loads** raw Amazon product metadata (McAuley-Lab dataset)
2. **Cleans** product text (strips part numbers, trims lengths)
3. **Summarises** each product via the Groq Batch API (`gpt-oss-20b`)
4. **Trains** a deep residual network on the summaries to predict price
5. **Pushes** the processed dataset to HuggingFace Hub

---

## Project Structure

```
pricer_project/
├── pricer/
│   ├── __init__.py       # Package exports
│   ├── items.py          # Item pydantic model + Hub helpers
│   ├── parser.py         # Raw datapoint scrubbing
│   ├── loader.py         # Parallel ItemLoader
│   ├── preprocessor.py   # Single-item LLM summariser (LiteLLM)
│   ├── batch.py          # Groq Batch API orchestration
│   └── model.py          # DeepNeuralNetwork + Runner
├── scripts/
│   └── pipeline.py       # Full end-to-end script
├── .env.example
├── pyproject.toml
└── requirements.txt
```

---

## Setup

```bash
# 1. Clone and enter the project
cd pricer_project

# 2. Create and activate a virtualenv
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install
pip install -e .

# 4. Set your API keys
cp .env.example .env
# Edit .env and add your GROQ_API_KEY and HUGGINGFACE_TOKEN
```

---

## Usage

### Full pipeline
```bash
python scripts/pipeline.py
```

### Individual modules

```python
from pricer import ItemLoader, Batch, Item, DeepNeuralNetworkRunner

# Load a single category
items = ItemLoader("Electronics").load()

# Summarise via Groq Batch API
Batch.create(items)
Batch.run()
# … wait for jobs … then:
Batch.fetch()

# Train the model
train, val = items[:8000], items[8000:]
runner = DeepNeuralNetworkRunner(train, val)
runner.setup()
runner.train(epochs=5)
runner.save("model.pt")

# Predict
price = runner.inference(val[0])
print(f"Predicted: ${price:.2f}  Actual: ${val[0].price:.2f}")
```

---

## Pipeline Architecture

```
Amazon Reviews Dataset
        │
        ▼
   ItemLoader  ──────────────────────────────────── parallel parsing
        │
        ▼
    parser.py  ──── scrub(), get_weight()  ──────── clean raw text
        │
        ▼
  Groq Batch API  ─── gpt-oss-20b ──────────────── structured summary
        │                                           Title / Category /
        ▼                                           Brand / Description
  Item.summary populated                           / Details
        │
        ├── Item.push_to_hub()  ─────────────────── HuggingFace Hub
        │
        ▼
  DeepNeuralNetworkRunner
        │  HashingVectorizer (5000 features)
        │  Log-normalised targets
        │  ResidualBlock × 8  (hidden=4096)
        │  AdamW + CosineAnnealingLR
        ▼
    model.pt  ──── inference()  ─────────────────── predicted price $
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `HashingVectorizer` over embeddings | Fast, memory-efficient, no API calls at inference time |
| Log-normalise prices | Price distribution is log-normal; improves L1 loss convergence |
| Residual blocks | Depth without vanishing gradients in a simple MLP setting |
| Groq Batch API | ~10× cheaper than synchronous calls for bulk summarisation |
| `reasoning_effort="low"` | Sufficient for structured reformatting; saves tokens |

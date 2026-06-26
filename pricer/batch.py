import os
import json
import pickle
import time
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"          # cheap + fast; swap to gpt-4o if quality matters
BATCHES_FOLDER = "batches"
OUTPUT_FOLDER = "output"
STATE_FILE = Path("batches.pkl")

SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""


class Batch:
    """
    Manages a single batch of items via the OpenAI Batch API.

    Full lifecycle: build JSONL → upload file → submit batch job →
    poll for completion → download output → apply summaries to Items.

    OpenAI Batch API gives 50% cost reduction vs synchronous calls
    with up to 24h turnaround — ideal for large-scale preprocessing.
    """

    BATCH_SIZE = 1_000
    batches: list["Batch"] = []

    def __init__(self, items: list, start: int, end: int, lite: bool):
        self.items = items
        self.start = start
        self.end = end
        self.filename = f"{start}_{end}.jsonl"
        self.file_id: str | None = None
        self.batch_id: str | None = None
        self.output_file_id: str | None = None
        self.done = False

        folder = Path("lite") if lite else Path("full")
        self.batches_dir = folder / BATCHES_FOLDER
        self.output_dir = folder / OUTPUT_FOLDER
        self.batches_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Instance methods                                                     #
    # ------------------------------------------------------------------ #

    def _make_jsonl_line(self, item) -> str:
        body = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item.full},
            ],
            "max_tokens": 200,
        }
        line = {
            "custom_id": str(item.id),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        return json.dumps(line)

    def make_file(self):
        """Write this batch's items to a JSONL file on disk."""
        batch_file = self.batches_dir / self.filename
        with batch_file.open("w", encoding="utf-8") as f:
            for item in self.items[self.start : self.end]:
                f.write(self._make_jsonl_line(item))
                f.write("\n")

    def send_file(self):
        """Upload the JSONL file to OpenAI Files API and store the file_id."""
        batch_file = self.batches_dir / self.filename
        with batch_file.open("rb") as f:
            response = client.files.create(file=f, purpose="batch")
        self.file_id = response.id

    def submit_batch(self):
        """Submit the uploaded file as a batch job and store the batch_id."""
        response = client.batches.create(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id=self.file_id,
        )
        self.batch_id = response.id

    def is_ready(self) -> bool:
        """Poll batch status. Returns True and stores output_file_id when done."""
        response = client.batches.retrieve(self.batch_id)
        status = response.status
        if status == "completed":
            self.output_file_id = response.output_file_id
        elif status == "failed":
            raise RuntimeError(f"Batch {self.batch_id} failed: {response.errors}")
        return status == "completed"

    def fetch_output(self):
        """Download completed batch output to disk."""
        output_path = self.output_dir / self.filename
        file_response = client.files.content(self.output_file_id)
        output_path.write_bytes(file_response.content)

    def apply_output(self):
        """Parse downloaded output and apply summaries back to Item objects."""
        output_path = self.output_dir / self.filename
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                json_line = json.loads(line)
                item_id = int(json_line["custom_id"])
                summary = json_line["response"]["body"]["choices"][0]["message"]["content"]
                self.items[item_id].summary = summary
        self.done = True

    # ------------------------------------------------------------------ #
    # Class methods (manage all batches)                                  #
    # ------------------------------------------------------------------ #

    @classmethod
    def create(cls, items: list, lite: bool = False):
        """Partition items into BATCH_SIZE chunks and create Batch objects."""
        cls.batches = []
        for start in range(0, len(items), cls.BATCH_SIZE):
            end = min(start + cls.BATCH_SIZE, len(items))
            cls.batches.append(Batch(items, start, end, lite))
        print(f"Created {len(cls.batches)} batches of up to {cls.BATCH_SIZE} items each")

    @classmethod
    def run(cls):
        """Make, upload, and submit all batches."""
        for batch in tqdm(cls.batches, desc="Submitting batches"):
            batch.make_file()
            batch.send_file()
            batch.submit_batch()
        print(f"Submitted {len(cls.batches)} batches — results arrive within 24h")

    @classmethod
    def fetch(cls, poll_interval: int = 60):
        """
        Poll until all batches complete, then download and apply output.

        Args:
            poll_interval: Seconds to wait between status checks.
        """
        while True:
            for batch in cls.batches:
                if not batch.done and batch.is_ready():
                    batch.fetch_output()
                    batch.apply_output()

            finished = sum(1 for b in cls.batches if b.done)
            pending = len(cls.batches) - finished
            print(f"Progress: {finished}/{len(cls.batches)} batches done")

            if pending == 0:
                break
            print(f"  {pending} pending — retrying in {poll_interval}s…")
            time.sleep(poll_interval)

    @classmethod
    def save(cls, path: Path = STATE_FILE):
        """Pickle batch state (without items) to disk."""
        items = cls.batches[0].items if cls.batches else None
        for batch in cls.batches:
            batch.items = None
        with path.open("wb") as f:
            pickle.dump(cls.batches, f)
        for batch in cls.batches:
            batch.items = items
        print(f"Saved {len(cls.batches)} batch states to {path}")

    @classmethod
    def load(cls, items: list, path: Path = STATE_FILE):
        """Restore pickled batch state and re-attach the items list."""
        with path.open("rb") as f:
            cls.batches = pickle.load(f)
        for batch in cls.batches:
            batch.items = items
        print(f"Loaded {len(cls.batches)} batch states from {path}")

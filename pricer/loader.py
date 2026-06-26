from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ProcessPoolExecutor
from pricer.parser import parse
from pricer.items import Item
import os

CHUNK_SIZE = 1000

cpu_count = os.cpu_count()
WORKERS = max(cpu_count - 1, 1)


class ItemLoader:
    """
    Loads and processes a single Amazon product category from the
    McAuley-Lab/Amazon-Reviews-2023 dataset.
    """

    def __init__(self, category: str):
        self.category = category
        self.dataset = None

    def from_datapoint(self, datapoint: dict) -> Item | None:
        """
        Try to create an Item from this datapoint.
        Returns the Item if successful, or None if it shouldn't be included.
        """
        return parse(datapoint, self.category)

    def from_chunk(self, chunk) -> list[Item]:
        """
        Create a list of Items from this chunk of dataset elements.
        """
        batch = [self.from_datapoint(datapoint) for datapoint in chunk]
        return [item for item in batch if item is not None]

    def chunk_generator(self):
        """
        Iterate over the Dataset, yielding chunks of datapoints at a time.
        """
        size = len(self.dataset)
        for i in range(0, size, CHUNK_SIZE):
            yield self.dataset.select(range(i, min(i + CHUNK_SIZE, size)))

    def load_in_parallel(self, workers: int) -> list[Item]:
        """
        Use ProcessPoolExecutor to farm out chunk processing.
        Speeds up processing significantly, but ties up CPU while running.
        """
        results = []
        chunk_count = (len(self.dataset) // CHUNK_SIZE) + 1
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for batch in tqdm(
                pool.map(self.from_chunk, self.chunk_generator()),
                total=chunk_count,
                desc=f"Processing {self.category}",
            ):
                results.extend(batch)
        return results

    def load(self, workers: int = WORKERS) -> list[Item]:
        """
        Load and scrub this dataset category.

        Args:
            workers: Number of parallel processes. Defaults to CPU count - 1.

        Returns:
            List of valid Item objects.
        """
        start = datetime.now()
        print(f"Loading dataset {self.category}", flush=True)
        self.dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{self.category}",
            split="full",
        )
        results = self.load_in_parallel(workers)
        finish = datetime.now()
        elapsed = (finish - start).total_seconds() / 60
        print(
            f"Completed {self.category} with {len(results):,} datapoints in {elapsed:.1f} mins",
            flush=True,
        )
        return results

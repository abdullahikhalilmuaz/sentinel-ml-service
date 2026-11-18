
import csv
from pathlib import Path

from src.train import generate_synthetic_normal_samples
from src.feature_schema import FEATURE_NAMES

# Reproduce the original training dataset
SAMPLE_COUNT = 5000
RANDOM_SEED = 42

# Generate the same synthetic records
rows = generate_synthetic_normal_samples(
    count=SAMPLE_COUNT,
    seed=RANDOM_SEED
)

# Create output directory
output_dir = Path("data")
output_dir.mkdir(parents=True, exist_ok=True)

# CSV destination
output_file = output_dir / "sentinel_synthetic_training_data.csv"

# Export using the original feature schema
with output_file.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=FEATURE_NAMES
    )

    writer.writeheader()
    writer.writerows(rows)

print("=" * 55)
print("SENTINEL SYNTHETIC TRAINING DATA EXPORTED")
print("=" * 55)
print(f"Total records: {len(rows)}")
print(f"Features: {len(FEATURE_NAMES)}")
print(f"Random seed: {RANDOM_SEED}")
print(f"CSV location: {output_file.resolve()}")
print("=" * 55)


from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DATA_DIR / "ai4i2020.csv"

def main():
    print("Downloading AI4I dataset from UCI...")

    dataset = fetch_ucirepo(id=601)

    # Prefer the original dataset when available.
    df = dataset.data.original

    if df is None:
        features = dataset.data.features
        targets = dataset.data.targets

        if targets is not None:
            df = pd.concat([features, targets], axis=1)
        else:
            df = features.copy()

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved dataset to: {OUTPUT_FILE}")
    print(f"Rows: {len(df):,}")
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nFirst five records:")
    print(df.head())

if __name__ == "__main__":
    main()

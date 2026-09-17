from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/SMSSpamCollection")


def load_data() -> pd.DataFrame:
    """Load and prepare the raw SMS Spam Collection dataset."""

    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        names=["label", "message"]
    )

    df = df.drop_duplicates().reset_index(drop=True)

    return df


if __name__ == "__main__":
    data = load_data()

    print("FIRST FIVE ROWS")
    print(data.head())

    print("\nDATASET SHAPE")
    print(data.shape)

    print("\nCOLUMN INFORMATION")
    data.info()

    print("\nLABEL DISTRIBUTION")
    print(data["label"].value_counts())

    print("\nMISSING VALUES")
    print(data.isnull().sum())

    print("\nDUPLICATE ROWS")
    print(data.duplicated().sum())
import yaml
import pandas as pd
from pathlib import Path
from src.preprocess_data import clean_text


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(raw_data_path: str) -> pd.DataFrame:
    return pd.read_csv(raw_data_path)


def clean_dataset(df: pd.DataFrame, steps: list[str], text_column: str) -> pd.DataFrame:
    df_copy = df.copy()
    if not steps:
         steps = [
             "remove_special_tokens",
             "lowercase",
             "remove_punctuation",
             "remove_stopwords",
             "remove_extra_whitespace"
         ]
    df_copy[text_column] = df_copy[text_column].apply(lambda text: clean_text(text, steps))
    return df_copy


def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    project_root = Path(__file__).parent.parent.resolve()
    config_path = project_root / "config.yaml"
    config = load_config(str(config_path))
    local_raw_data_path = (project_root / config["data"]["local_raw_data_path"]).resolve()
    text_column = config["task"]["text_column"]

    print(f"Loading data from {local_raw_data_path}...")
    df = load_data(str(local_raw_data_path))
    print(df.head())
    print(f"Loaded {len(df)} rows")

    print("Cleaning dataset...")
    steps = ["remove_special_tokens", "lowercase", "remove_punctuation", "remove_stopwords", "remove_extra_whitespace"]
    df_cleaned = clean_dataset(df, steps, text_column)

    output_path = str(local_raw_data_path).replace("train_aggressiveness.csv", "train_aggressiveness_preprocessed.csv")
    print(f"Saving cleaned data to {output_path}...")
    save_cleaned_data(df_cleaned, output_path)
    print("Saved cleaned data successfully!")
    
    print("Sample of cleaned data:")
    print(df_cleaned.head())
    print("Done!")


if __name__ == "__main__":
    main()

from pathlib import Path
from src.utils import load_config, get_config_path, get_project_root, load_csv
from src.preprocess_data import clean_text


def preprocess_dataset(df: pd.DataFrame, steps: list[str], text_column: str) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy[text_column] = df_copy[text_column].apply(lambda text: clean_text(text, steps))
    return df_copy


def save_data(df: pd.DataFrame, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    config_path = get_config_path()
    config = load_config(str(config_path))
    project_root = get_project_root()
    local_raw_data_path = (project_root / config["data"]["local_raw_data_path"]).resolve()
    local_preprocessed_data_path = (project_root / config["data"]["local_preprocessed_data_path"]).resolve()
    text_column = config["task"]["text_column"]

    print(f"Loading data from {local_raw_data_path}...")
    df = load_csv(str(local_raw_data_path))
    print(df.head())
    print(f"Loaded {len(df)} rows")

    print("Preprocessing dataset...")
    steps = config["task"]["preprocessing_steps"]
    df_cleaned = preprocess_dataset(df, steps, text_column)

    output_path = str(local_preprocessed_data_path)
    print(f"Saving preprocessed data to {output_path}...")
    save_data(df_cleaned, output_path)
    print("Saved preprocessed data successfully!")
    
    print("Sample of preprocessed data:")
    print(df_cleaned.head())
    print("Done!")


if __name__ == "__main__":
    main()

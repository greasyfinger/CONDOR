import pandas as pd

FILE_PATH = "data/edos_labelled_aggregated.csv"


def clean_text(df):
    df.text = df.text.str.replace(r"\bRT\b", " ", regex=True)
    df.text = df.text.str.replace("(@[^\s]*\s|\s?@[^\s]*$)", " ", regex=True)
    df.text = df.text.str.replace("https?://[^\s]*(\s|$)", " ", regex=True)
    df.text = df.text.str.strip()
    return df


def clean_label(df):
    df.label = df.label.str.replace(r"^\d+\.\s*", "", regex=True)
    df.label = df.label.str.strip()
    df.label = df.label.replace("none", "not sexist")
    return df


def get_dataset(file_path=FILE_PATH, task="B"):
    df = pd.read_csv(file_path)

    if task == "B":
        df = df.rename(columns={"label_category": "label"})
        df = clean_text(df)
        df = clean_label(df)
        df = df[df["label"] != "not sexist"]
    elif task == "A":
        df = df.rename(columns={"label_sexist": "label"})
        df = clean_text(df)
        df = clean_label(df)
    train_df = df.loc[df["split"] == "train"][["text", "label"]]
    dev_df = df.loc[df["split"] == "dev"][["text", "label"]]
    test_df = df.loc[df["split"] == "test"][["text", "label"]]

    return train_df, dev_df, test_df


if __name__ == "__main__":
    get_dataset()

import pandas as pd

from .config import EMAILS_PATH, URLS_PATH


def load_urls() -> pd.DataFrame:
    """Kolumny: url_id, url (znormalizowany), url_original, host, label, source, ..."""
    return pd.read_csv(URLS_PATH)


def load_emails() -> pd.DataFrame:
    """Dodaje kolumnę `text` = subject + body."""
    df = pd.read_csv(EMAILS_PATH, low_memory=False)
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")
    df["text"] = (df["subject"] + "\n" + df["body"]).str.strip()
    return df

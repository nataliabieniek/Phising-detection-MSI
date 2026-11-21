import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .config import LABEL_COL, SEED, TEST_SIZE, VAL_SIZE


def stratified_split(df: pd.DataFrame, seed: int = SEED):
    """Train/val/test stratyfikowany po etykiecie (maile)."""
    train_val, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[LABEL_COL], random_state=seed
    )
    train, val = train_test_split(
        train_val,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        stratify=train_val[LABEL_COL],
        random_state=seed,
    )
    return train, val, test


def group_split(df: pd.DataFrame, group_col: str = "host", seed: int = SEED):
    """Train/val/test, w którym grupa (host) występuje tylko w jednym zbiorze (URL)."""
    outer = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    tv_idx, test_idx = next(outer.split(df, groups=df[group_col]))
    train_val, test = df.iloc[tv_idx], df.iloc[test_idx]
    inner = GroupShuffleSplit(
        n_splits=1, test_size=VAL_SIZE / (1 - TEST_SIZE), random_state=seed
    )
    tr_idx, val_idx = next(inner.split(train_val, groups=train_val[group_col]))
    return train_val.iloc[tr_idx], train_val.iloc[val_idx], test


def source_split(df: pd.DataFrame, test_sources, source_col: str = "source"):
    """Test cross-source: trening na pozostałych źródłach, test na `test_sources`."""
    mask = df[source_col].isin(test_sources)
    return df[~mask], df[mask]


def describe(**splits: pd.DataFrame) -> None:
    for name, part in splits.items():
        n_pos = int(part[LABEL_COL].sum())
        print(f"{name:>8}: {len(part):>8} rows, phishing={n_pos} ({n_pos / len(part):.1%})")

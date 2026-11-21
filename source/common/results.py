import pandas as pd

from .config import ROOT

RESULTS_DIR = ROOT / "results"


def save_results(method: str, rows: list[dict]) -> None:
    """Zapisuje wyniki metody (np. `mails_basic`, `url_advance`) do results/<method>.csv (nadpisuje)."""
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / f"{method}.csv", index=False)

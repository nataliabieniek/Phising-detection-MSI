from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "Dane"
URLS_PATH = DATA_DIR / "urls.csv.gz"
EMAILS_PATH = DATA_DIR / "emails.csv.gz"

SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

LABEL_COL = "label"  # 1 = phishing, 0 = legit

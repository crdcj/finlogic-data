from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

# Assets that travel through GitHub Releases.
RELEASE_STAGING_DIR = REPO_ROOT / "release_staging"
CVM_RELEASE_DIR = RELEASE_STAGING_DIR / "cvm"
PROCESSED_DIR = CVM_RELEASE_DIR / "processed"
FINANCIALS_PARQUET = RELEASE_STAGING_DIR / "financials.parquet"
TRADES_PARQUET = RELEASE_STAGING_DIR / "trades.parquet"
TRADED_COMPANIES_JSON = RELEASE_STAGING_DIR / "traded_companies.json"

# Local-only cache for raw CVM downloads. Kept outside git and releases.
BRONZE_DIR = REPO_ROOT / "cache" / "bronze"
CVM_RAW_DIR = BRONZE_DIR / "cvm" / "raw"


def ensure_directories() -> None:
    CVM_RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RELEASE_STAGING_DIR.mkdir(parents=True, exist_ok=True)

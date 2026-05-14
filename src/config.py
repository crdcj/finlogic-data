from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

# Assets that travel through GitHub Releases.
RELEASE_STAGING_DIR = REPO_ROOT / "release_staging"
FINANCIALS_PARQUET = RELEASE_STAGING_DIR / "financials.parquet"
TRADES_PARQUET = RELEASE_STAGING_DIR / "trades.parquet"
TRADED_COMPANIES_JSON = RELEASE_STAGING_DIR / "traded_companies.json"

# Local-only pipeline caches. Kept outside git and releases.
CACHE_DIR = REPO_ROOT / "cache"
BRONZE_DIR = CACHE_DIR / "bronze"
CVM_RAW_DIR = BRONZE_DIR / "cvm" / "raw"
SILVER_DIR = CACHE_DIR / "silver"
PROCESSED_DIR = SILVER_DIR / "cvm" / "processed"


def ensure_directories() -> None:
    CVM_RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RELEASE_STAGING_DIR.mkdir(parents=True, exist_ok=True)

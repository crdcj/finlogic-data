import base64
import json
import tempfile
import zipfile as zf
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import polars as pl
import requests
import urllib3

from src.config import FINANCIALS_PARQUET, RELEASE_STAGING_DIR, TRADED_COMPANIES_JSON, TRADES_PARQUET, ensure_directories

ensure_directories()
DATA_PATH = RELEASE_STAGING_DIR
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def process_df(df: pl.DataFrame) -> pl.DataFrame:
    """Processar DataFrame e retorná-lo pronto para análise"""
    df = (
        df.with_columns(
            [
                pl.col("tpmerc").cast(pl.Int64),
                pl.col("codbdi").cast(pl.Int64),
                pl.col("close_price").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
                pl.col("fatcot").cast(pl.Int64),
                pl.col("most_traded_stock").str.slice(0, 4).alias("b3_id"),
                pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d"),
            ]
        )
        .with_columns(pl.col("close_price").truediv(100))
        .with_columns(pl.col("volume").truediv(100))
        .with_columns((pl.col("close_price") / (pl.col("fatcot"))).alias("close_price"))
        .with_columns(pl.col("codbdi").is_in([7, 8]).alias("is_restructuring"))
        .filter(pl.col("tpmerc") == 10)
        .filter(pl.col("codbdi").is_in([2, 7, 8]))
        .filter(pl.col("fatcot") != 0)
        .drop("full_str", "tpmerc", "fatcot", "codbdi")
        .sort("volume")
        .unique(subset=["b3_id"], keep="last", maintain_order=True)
        .sort("b3_id")
    )
    reorder_cols = [
        "b3_id",
        "is_restructuring",
        "most_traded_stock",
        "stock_type",
        "close_price",
        "volume",
        "trade_date",
    ]

    return df.select(reorder_cols)


def read_zip_file(zip_filepath: Path) -> Path:
    with zf.ZipFile(zip_filepath, "r") as myzip:
        first_filename = myzip.namelist()[0]
        temp_dir = Path(tempfile.mkdtemp(prefix="finlogic-b3-"))
        unzipped_filepath = myzip.extract(first_filename, temp_dir)
    return Path(unzipped_filepath)


def read_raw_file(filepath) -> pl.DataFrame:
    """Ler arquivo bruto e retornar um DataFrame

    Campos de leitura: ver PDF SeriesHistoricas_Layout.pdf
    """
    column_ranges = {
        "trade_date": (2, 8),
        "codbdi": (10, 2),
        "most_traded_stock": (12, 12),
        "tpmerc": (24, 3),
        "stock_type": (39, 12),
        "close_price": (108, 13),
        "volume": (170, 17),
        "fatcot": (210, 7),
    }
    column_names = list(column_ranges.keys())
    slice_tuples = list(column_ranges.values())
    unzipped_filepath = read_zip_file(filepath)
    df = pl.read_csv(
        unzipped_filepath,
        has_header=False,
        skip_rows=1,
        new_columns=["full_str"],
    )
    df = df.head(-1)  # drop last row

    df = df.with_columns(
        [
            pl.col("full_str")
            .str.slice(slice_tuple[0], slice_tuple[1])
            .str.strip_chars()
            .alias(col)
            for slice_tuple, col in zip(slice_tuples, column_names)
        ]
    )
    return df


def save_trades_file(session_date: date) -> Optional[Path]:
    # Filename example: COTAHIST_D27122021.ZIP
    # File URL example: https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_D14072023.ZIP
    formated_session_date = session_date.strftime("%d%m%Y")
    filename = f"COTAHIST_D{formated_session_date}.ZIP"
    file_url = f"https://bvmf.bmfbovespa.com.br/InstDados/SerHist/{filename}"

    print("Nome do arquivo:", filename)
    print("URL: " + file_url)

    r_trades = requests.get(file_url, stream=True, allow_redirects=True, verify=False)
    if r_trades.status_code != 200:
        return None

    filepath = Path("/tmp") / filename
    filepath.write_bytes(r_trades.content)
    return filepath


def string_to_base64(string):
    return base64.b64encode(string.encode("utf-8")).decode("utf-8")


def get_listed_df():
    URL_BASE = "https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetInitialCompanies/"

    # {"language":"pt-br","pageNumber":1,"pageSize":120}
    code_b64 = "eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjEyMH0="
    url_listed = URL_BASE + code_b64

    r = requests.get(url_listed, verify=False)
    if r.status_code != 200:
        raise SystemExit("Não há dados disponíveis")

    df = pl.DataFrame()
    json_data = r.json()
    total_pages = int(json_data["page"]["totalPages"])
    page_numbers = list(range(1, total_pages + 1))
    for page_number in page_numbers:
        code_b64 = string_to_base64(
            '{"language":"pt-br","pageNumber":%d,"pageSize":120}' % page_number
        )
        url_listed = URL_BASE + code_b64
        r = requests.get(url_listed, verify=False)
        listed_dict = (r.json())["results"]

        # Não precisa ler a coluna 'status' -> são todos 'A'
        cols = ["issuingCompany", "codeCVM", "segmentEng", "type"]
        rename_cols = {
            "codeCVM": "cvm_id",
            "issuingCompany": "b3_id",
            "segmentEng": "segment",
        }
        df_page = (
            pl.DataFrame(listed_dict)
            .select(cols)
            .rename(rename_cols)
            .with_columns(
                pl.col("cvm_id").cast(pl.Int64),
                pl.col("type").cast(pl.Int64),
                pl.col("segment").str.to_lowercase(),
            )
            .filter(pl.col("type") == 1)
            .drop("type")
        )
        df = pl.concat([df, df_page])

    return df


def build_trades_df() -> Optional[pl.DataFrame]:
    today = date.today()
    # The job runs at night, so we need to get the trades from the previous day.
    session_date = today - timedelta(days=1)
    # session_date = date(2023, 7, 21)  # Test date
    trades_filepath = save_trades_file(session_date)
    if trades_filepath is None:
        return None
    raw_df = read_raw_file(trades_filepath)
    trades_df = process_df(raw_df)
    listed_df = get_listed_df()
    trades_df = trades_df.join(listed_df, on="b3_id", how="inner").sort(by=["b3_id"])
    # .drop(["close_price", "trade_date"])
    return trades_df


def run():
    trades_df = build_trades_df()
    if trades_df is None:
        print("Não há arquivo de sessão nessa data: 'trades.parquet' não foi atualizado")
        if TRADED_COMPANIES_JSON.exists():
            traded_companies = json.loads(TRADED_COMPANIES_JSON.read_text(encoding="utf-8"))
        else:
            traded_companies = []
    else:
        trades_df.write_parquet(TRADES_PARQUET, compression="zstd")
        traded_companies = (
            trades_df.select(pl.col("cvm_id").unique().sort()).to_series().to_list()
        )
        TRADED_COMPANIES_JSON.write_text(
            json.dumps(traded_companies),
            encoding="utf-8",
        )

    if not FINANCIALS_PARQUET.exists():
        print("financials.parquet not found in release staging.")
        return

    print(
        "Updated release assets: "
        f"{TRADES_PARQUET.name if TRADES_PARQUET.exists() else 'trades.parquet skipped'}, "
        f"{FINANCIALS_PARQUET.name}"
    )

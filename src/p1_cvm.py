import io
import re
import zipfile as zf
from typing import List

import polars as pl
import polars.selectors as cs
import requests
from requests.adapters import HTTPAdapter
from src.config import CVM_RAW_DIR, FINANCIALS_PARQUET, PROCESSED_DIR, ensure_directories
from urllib3.util.retry import Retry

pl.enable_string_cache()

ensure_directories()

session = requests.Session()
retry_strategy = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "HEAD"],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

PAGE_TIMEOUT = (10, 60)
FILE_SIZE_TIMEOUT = (10, 60)
DOWNLOAD_TIMEOUT = (10, 300)


def get_file_urls_in_page(cvm_url) -> List[str]:
    """Return a list of available CVM files.

    File example:
    http://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/dfp_cia_aberta_2020.zip
    http://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/ITR/DADOS/itr_cia_aberta_2020.zip
    """
    available_files = []
    response = session.get(cvm_url, timeout=PAGE_TIMEOUT)
    if response.status_code != 200:
        return available_files
    # Use a regular expression to match and extract all the file links
    matches = re.findall(r'href="(.+\.zip)"', response.text)
    # Add matches to the file list
    available_files.extend(matches)
    available_files.sort()
    # Add the base url to the filename
    available_file_urls = [cvm_url + filename for filename in available_files]
    return available_file_urls


def get_file_urls() -> List[str]:
    """Return a list of all available CVM files."""
    URL_DFP = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/"
    URL_ITR = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/ITR/DADOS/"
    urls_dfp = get_file_urls_in_page(URL_DFP)
    urls_itr = get_file_urls_in_page(URL_ITR)
    urls = urls_dfp + urls_itr
    return urls


def get_url_file_size(url: str) -> int:
    response = session.get(url, stream=True, timeout=FILE_SIZE_TIMEOUT)
    response.raise_for_status()
    file_size = int(response.headers["content-length"])
    response.close()
    return file_size


def build_changed_urls(file_urls: List[str]) -> List[str]:
    changed_urls = []
    for file_url in file_urls:
        filename = file_url.split("/")[-1]
        local_zip = CVM_RAW_DIR / filename
        try:
            cvm_size = get_url_file_size(file_url)
        except requests.RequestException:
            if local_zip.exists():
                print(f"Could not refresh {filename}; using cached bronze zip.")
                continue
            raise

        local_size = local_zip.stat().st_size if local_zip.exists() else 0
        if cvm_size != local_size:
            changed_urls.append(file_url)
    return changed_urls


def update_raw_files(changed_urls: List[str]):
    for url in changed_urls:
        filename = url.split("/")[-1]
        response = session.get(url, timeout=DOWNLOAD_TIMEOUT)
        response.raise_for_status()
        filepath = CVM_RAW_DIR / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_bytes(response.content)
        print(f"{filename} updated in bronze cache")


def build_filenames_to_process(changed_urls: List[str]) -> List[str]:
    """Return a list of files to be processed."""
    changed_filenames = [url.split("/")[-1] for url in changed_urls]
    changed_base_names = [filename.split(".")[0] for filename in changed_filenames]
    changed_set = set(changed_base_names)

    raw_filenames = sorted(path.name for path in CVM_RAW_DIR.glob("*.zip"))
    raw_base_names = [filename.split(".")[0] for filename in raw_filenames]
    raw_set = set(raw_base_names)

    processed_filenames = sorted(path.name for path in PROCESSED_DIR.glob("*.parquet"))
    processed_base_names = [filename.split(".")[0] for filename in processed_filenames]
    processed_set = set(processed_base_names)

    not_processed_set = raw_set - processed_set

    filenames_to_process = sorted(changed_set | not_processed_set)
    filenames_to_process = [filename + ".zip" for filename in filenames_to_process]
    return filenames_to_process


def read_zip_file(fileobj: io.BytesIO) -> pl.DataFrame:
    """Read all CSV files in zip file and return a single DataFrame."""
    cvm_zipfile = zf.ZipFile(fileobj)

    filenames = cvm_zipfile.namelist()
    valid_filenames = []
    valid_reports = ["BPA", "BPP", "DFC_MI", "DRE"]
    for filename in filenames:
        for valid_report in valid_reports:
            if valid_report in filename:
                valid_filenames.append(filename)
                break

    df_list = []
    for filename in valid_filenames:
        file_data = cvm_zipfile.open(filename).read()
        f = io.BytesIO(file_data)
        raw_df = pl.read_csv(
            f,
            separator=";",
            encoding="iso-8859-1",
            try_parse_dates=True,
            quote_char=None,
        )
        cleaned_df = process_df(raw_df, filename)
        df_list.append(cleaned_df)
    df = pl.concat(df_list)
    return df


def process_df(df: pl.DataFrame, filename: str) -> pl.DataFrame:
    """Format a cvm dataframe."""
    columns_translation = {
        "DENOM_CIA": "name_id",
        "CD_CVM": "cvm_id",
        "CNPJ_CIA": "tax_id",
        "VERSAO": "report_version",
        "DT_REFER": "period_reference",
        "DT_FIM_EXERC": "period_end",
        "ORDEM_EXERC": "period_order",
        "CD_CONTA": "acc_code",
        "DS_CONTA": "acc_name",
        "ST_CONTA_FIXA": "is_acc_fixed",
        "VL_CONTA": "acc_value",
        "GRUPO_DFP": "report_group",
        "MOEDA": "currency",
        "ESCALA_MOEDA": "currency_unit",
    }
    for old_name, new_name in columns_translation.items():
        df = df.rename({old_name: new_name})

    # currency_unit values are ['MIL', 'UNIDADE']
    map_dic = {"UNIDADE": 1, "MIL": 1000}
    # Sometimes acc_value can be read as string.
    df = (
        df.with_columns(pl.col("acc_value").cast(pl.Float64))
        .filter(pl.col("acc_value") != 0)
        .filter(pl.col("acc_value").is_not_null())
        .with_columns(
            pl.col("currency_unit").replace_strict(map_dic, return_dtype=pl.Int32)
        )
        .with_columns(
            pl.when(pl.col("acc_code").str.starts_with("3.99"))
            .then(pl.col("acc_value"))
            .otherwise(pl.col("acc_value") * pl.col("currency_unit"))
            .alias("acc_value")
        )
        .with_columns(pl.col("acc_value").round(2))
    )

    df = df.with_columns(
        [
            pl.col("period_reference").cast(pl.Date),
            pl.col("period_end").cast(pl.Date),
            pl.col("cvm_id").cast(pl.UInt32),  # cvm_id max. value is 600_000
            # Remove any extra spaces (line breaks, tabs, etc.)
            pl.col("name_id").str.replace_all(r"\s+", " ").str.strip_chars(),
            pl.col("acc_name").str.replace_all(r"\s+", " ").str.strip_chars(),
        ]
    )
    df = df.with_columns(pl.col("name_id").str.replace("BCO ", "BANCO "))

    # There are two types of CVM files: DFP (ANNUAL) and ITR (QUARTERLY).
    if filename.startswith("dfp"):
        df = df.with_columns(pl.lit(True).alias("is_annual"))
    else:
        df = df.with_columns(pl.lit(False).alias("is_annual"))

    """
    df['report_group'].unique() result:
        'DF Consolidado - Balanço Patrimonial Ativo',
        ...
        'DF Individual - Demonstração do Resultado',
    """
    df = df.with_columns(
        pl.when(pl.col("report_group").str.starts_with("DF Consolidado"))
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
        .alias("is_consolidated")
    )

    select_cols = [
        "name_id",
        "cvm_id",
        "tax_id",
        "is_annual",
        "is_consolidated",
        "period_reference",
        "period_begin",
        "period_end",
        "acc_code",
        "acc_name",
        "acc_value",
    ]
    if "DT_INI_EXERC" in df.columns:
        df = df.with_columns(pl.col("DT_INI_EXERC").cast(pl.Date)).rename(
            {"DT_INI_EXERC": "period_begin"}
        )
    else:
        df = df.with_columns(pl.lit(None).cast(pl.Date).alias("period_begin"))

    # In "itr_cia_aberta_2022.zip", as an example, 2742 rows are duplicated.
    # Few of them have different values in "acc_value". Only one them will be kept.
    # REMOVE ALL VALUES OR MARK THESE ROWS AS ERRORS?
    check_duplicates_cols = select_cols.copy()
    check_duplicates_cols.remove("acc_value")
    df = (
        df.select(select_cols)
        .unique(subset=check_duplicates_cols, keep="last", maintain_order=True)
        .with_columns(cs.string().cast(pl.Categorical))
    )
    return df


def process_file(raw_filename: str):
    fileobj = io.BytesIO((CVM_RAW_DIR / raw_filename).read_bytes())
    df = read_zip_file(fileobj)
    processed_filename = raw_filename.replace("zip", "parquet")
    processed_filepath = PROCESSED_DIR / processed_filename
    df.write_parquet(processed_filepath, compression="zstd")
    print(f"{processed_filename} updated in release staging")


def process_files(filenames: List[str]):
    for filename in filenames:
        print(f"Processing file {filename}...")
        process_file(filename)


def run() -> bool:
    try:
        file_urls = get_file_urls()
    except requests.RequestException as exc:
        existing_processed = list(PROCESSED_DIR.glob("*.parquet"))
        if existing_processed:
            print(
                "Could not reach CVM source; reusing processed parquet files from silver cache. "
                f"Reason: {exc}"
            )
            return True
        if FINANCIALS_PARQUET.exists():
            print(
                "Could not reach CVM source and no silver cache is available; "
                "reusing financials.parquet from release staging. "
                f"Reason: {exc}"
            )
            return False
        raise

    changed_urls = build_changed_urls(file_urls)
    if changed_urls:
        update_raw_files(changed_urls)
    filenames_to_process = build_filenames_to_process(changed_urls)
    if not filenames_to_process:
        print("No CVM files to process.")
        return True
    process_files(filenames_to_process)
    return True

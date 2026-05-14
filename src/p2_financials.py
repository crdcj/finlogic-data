from typing import Any

import polars as pl
from src.config import FINANCIALS_PARQUET, PROCESSED_DIR, ensure_directories

pl.enable_string_cache()

ensure_directories()
PROCESSED_GLOB_DFP = str(PROCESSED_DIR / "dfp*.parquet")
PROCESSED_GLOB_ITR = str(PROCESSED_DIR / "itr*.parquet")

sort_format = {
    "cvm_id": False,
    "is_annual": False,
    "is_consolidated": False,
    "acc_code": False,
    "period_reference": True,
    "period_begin": False,
    "period_end": False,
}
SORT_COLS = list(sort_format.keys())
DESC_ORDER = list(sort_format.values())
# There is a partial quarter and a full quarter after first quarter reports
# Keep only the full quarter. Also keeps only LTM instead of a partial quarter
# if both are available.
DUPL_COLS = ["cvm_id", "is_annual", "is_consolidated", "acc_code", "period_end"]


def read_dfp() -> pl.DataFrame:
    q = (
        pl.scan_parquet(PROCESSED_GLOB_DFP)
        .sort(by=SORT_COLS, descending=DESC_ORDER)
        .unique(subset=DUPL_COLS, keep="first")
        .with_columns(pl.col("period_end").max().over("cvm_id").alias("last_annual"))
        .sort(by=SORT_COLS, descending=DESC_ORDER)
    )
    return q.collect()


def get_itr_dict() -> dict[Any, Any]:
    q = (
        pl.scan_parquet(PROCESSED_GLOB_ITR)
        .select("cvm_id", "period_end")
        .unique()
        .sort("cvm_id", "period_end")
        .group_by("cvm_id")
        .agg(pl.col("period_end").tail(5))
        .with_columns(
            (pl.col("period_end").list.gather([0, -1])).alias("target_quarters")
        )
        .select("cvm_id", "target_quarters")
    )
    periods_to_read = q.collect()
    d = periods_to_read.to_dict(as_series=False)
    d = {
        cvm_id: target_quarters
        for cvm_id, target_quarters in zip(d["cvm_id"], d["target_quarters"])
    }
    return d


def read_itr() -> pl.DataFrame:
    d = get_itr_dict()
    q = (
        pl.scan_parquet(PROCESSED_GLOB_ITR)
        .filter(
            (
                pl.col("period_end")
                == (pl.col("cvm_id").replace_strict(d, default=[])).list.first()
            )
            | (
                pl.col("period_end")
                == (pl.col("cvm_id").replace_strict(d, default=[])).list.last()
            )
        )
        .with_columns(pl.col("period_end").max().over("cvm_id").alias("last_quarter"))
        .sort(by=SORT_COLS, descending=DESC_ORDER)
        .unique(subset=DUPL_COLS, keep="first")
    )
    return q.collect()


def build_ltm_df(df: pl.DataFrame) -> pl.DataFrame:
    """Adjust income and cash flow statements to LTM (Last Twelve Months).
    To get the LTM values, we need to sum the current accumulated quarter with
    the difference between the last annual value and the previous accumulated
    quarter. Provided that quarterly values are always accumulated, we can use
    the following formula:
    LTM = current quarter + (last annual - previous quarter)
    Example for 1Q23: LTM = 1Q23 + (A22 - 1Q22)
    Example for 3Q23: LTM = 3Q23 + (A23 - 3Q23)
    """
    ltm_data = (
        df.with_columns(
            pl.col("acc_code").cast(pl.Utf8).str.slice(0, 1).alias("ltm_type")
        )
        .filter(pl.col("ltm_type").str.contains("3|6"))
        .filter(pl.col("last_quarter") > pl.col("last_annual"))
        .filter(
            (
                ~pl.col("is_annual")
                | (
                    pl.col("is_annual")
                    & (pl.col("period_end") == pl.col("last_annual"))
                )
            )
        )
        .drop("ltm_type")
    )
    annuals = (
        ltm_data.filter(pl.col("is_annual"))
        .select(["cvm_id", "is_consolidated", "acc_code", "acc_value"])
        .rename({"acc_value": "acc_value_annual"})
    )
    quarters = ltm_data.filter(~pl.col("is_annual")).with_columns(
        pl.col("period_end").min().over(["cvm_id"]).alias("first_quarter")
    )
    quarter_m4 = (
        quarters.filter(pl.col("period_end") == pl.col("first_quarter"))
        .select(["cvm_id", "is_consolidated", "acc_code", "acc_value"])
        .rename({"acc_value": "acc_value_m4"})
    )
    quarter_m0 = (
        quarters.filter(pl.col("period_end") == pl.col("last_quarter"))
        .select(["cvm_id", "is_consolidated", "acc_code", "acc_value"])
        .rename({"acc_value": "acc_value_m0"})
    )
    ltm_adjusted = (
        quarter_m0.join(
            quarter_m4, on=["cvm_id", "is_consolidated", "acc_code"], how="inner"
        )
        .join(annuals, on=["cvm_id", "is_consolidated", "acc_code"], how="inner")
        .with_columns(
            pl.col("acc_value_m0")
            .add(pl.col("acc_value_annual"))
            .sub(pl.col("acc_value_m4"))
            .alias("acc_value")
        )
        .select(["cvm_id", "is_consolidated", "acc_code", "acc_value"])
    )
    # Construct again quarter_m0 with all columns except acc_value
    quarter_m0 = (
        quarters.filter(pl.col("period_end") == pl.col("last_quarter"))
        # Adjust period_begin to match period_end
        .with_columns(pl.col("first_quarter").alias("period_begin"))
        # .with_columns(pl.lit(True).alias("is_annual"))
        .drop(["acc_value", "first_quarter"])
    )
    # Insert original quarter_m0 columns in ltm_adjusted
    ltm_adjusted = ltm_adjusted.join(
        quarter_m0, on=["cvm_id", "is_consolidated", "acc_code"], how="inner"
    ).select(
        [
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
            "last_quarter",
            "last_annual",
        ]
    )
    return ltm_adjusted


def build_financials_df():
    itr = read_itr()
    dfp = read_dfp()
    df = (
        pl.concat([itr, dfp], how="diagonal")
        .sort(by=SORT_COLS, descending=DESC_ORDER)
        .unique(subset=DUPL_COLS, keep="first")
        .with_columns(
            pl.col("last_annual").max().over("cvm_id").alias("last_annual"),
            pl.col("last_quarter").max().over("cvm_id").alias("last_quarter"),
        )
        .filter(pl.col("last_annual").is_not_null())
        .filter(
            pl.col("is_annual")
            | (~pl.col("is_annual") & (pl.col("last_quarter") > pl.col("last_annual")))
        )
    )
    ltm_adjusted = build_ltm_df(df)
    df = (
        pl.concat([df, ltm_adjusted])
        .sort(by=SORT_COLS, descending=DESC_ORDER)
        .unique(subset=DUPL_COLS, keep="first")
        .sort(by=SORT_COLS, descending=DESC_ORDER)
    )
    return df


def run():
    df = build_financials_df()
    df.write_parquet(FINANCIALS_PARQUET, compression="zstd")

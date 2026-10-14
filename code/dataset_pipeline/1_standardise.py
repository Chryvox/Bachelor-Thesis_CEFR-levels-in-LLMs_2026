"""
CEFR Dataset Standardizer
=========================
Loads CEFR datasets (local, CSV or JSON/JSONL) and brings them into a
unified standard schema (UniversalCEFR-compatible).
Output is always a CSV file.

Usage: see __main__ below.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Standard schema (UniversalCEFR-compatible)
# ---------------------------------------------------------------------------

STANDARD_COLUMNS = [
    "title",        # Title / unique ID of the text
    "lang",         # Language, e.g. "en", "de"
    "source_name",  # Dataset name / source
    "format",       # Text format, e.g. "essay", "dialogue", "news"
    "category",     # Category, e.g. "grammar", "vocabulary"
    "cefr_level",   # A1 – C2
    "license",      # License
    "text",         # The actual text
]

VALID_CEFR_LEVELS = {"A1", "A2", "B1", "B2", "C1", "C2"}


@dataclass
class DatasetSpec:
    """Describes how a specific dataset is mapped onto the standard schema.

    For lang/license/source_name/format/category/title: either a
    column name (if the info varies per row) OR a fixed value
    (if, e.g., the whole dataset has one language). The *_is_column flags
    explicitly determine which of the two cases applies.

    CEFR mapping strategies (priority descending):
        1. cefr_level_thresholds  – Threshold-based for numeric values.
           Each entry is (upper_bound_exclusive, CEFR_level); the list
           is sorted automatically. The last entry catches all remaining
           values.

           Example: value 1.67 → "A2", anything below → "A1"
               cefr_level_thresholds=[(1.67, "A1"), (float("inf"), "A2")]

        2. cefr_numeric_round  – Rounds the raw value numerically and maps
           the result via cefr_level_map. Rounding mode controllable via
           cefr_round_mode ("round" | "floor" | "ceil").

           Example: 1.2 → round → 1 → "A1"
               cefr_numeric_round=True,
               cefr_level_map={"1": "A1", "2": "A2", ...}

        3. cefr_level_map  – Direct string-to-string mapping (as before).
               cefr_level_map={"low": "A1", "mid": "B1", ...}

        4. No mapping  – Raw value is only uppercase/strip-cleaned.

    If cefr_level_thresholds and cefr_numeric_round are set at the same
    time, cefr_level_thresholds takes precedence (+ warning).
    """

    text_col: str
    cefr_col: str

    lang: str                              # was: language
    license: str
    source_name: str                       # was: source

    lang_is_column: bool = False           # was: language_is_column
    license_is_column: bool = False
    source_name_is_column: bool = False    # was: source_is_column

    # title: column name or fixed value; None → auto-numbered
    title: Optional[str] = None            # was: id_col
    title_is_column: bool = False

    # format: e.g. "essay", "dialogue", "news" – column or fixed value
    format: Optional[str] = None
    format_is_column: bool = False

    # category: e.g. "grammar", "reading" – column or fixed value
    category: Optional[str] = None
    category_is_column: bool = False

    # Strategy 3 ─ direct string mapping  {"raw": "A1", ...}
    cefr_level_map: Optional[dict] = None

    # Strategy 1 ─ threshold mapping for numeric raw values
    # [(upper_bound_exclusive, CEFR_level), ...]
    cefr_level_thresholds: Optional[list[tuple[float, str]]] = None

    # Strategy 2 ─ numeric rounding, then via cefr_level_map
    cefr_numeric_round: bool = False
    cefr_round_mode: str = "round"         # "round" | "floor" | "ceil"

    def __post_init__(self) -> None:
        valid_modes = {"round", "floor", "ceil"}
        if self.cefr_round_mode not in valid_modes:
            raise ValueError(
                f"cefr_round_mode must be one of {valid_modes}, "
                f"not '{self.cefr_round_mode}'."
            )
        if self.cefr_level_thresholds is not None and self.cefr_numeric_round:
            warnings.warn(
                "Both cefr_level_thresholds and cefr_numeric_round are set – "
                "cefr_level_thresholds takes precedence.",
                stacklevel=2,
            )


# ---------------------------------------------------------------------------
# Internal helper functions for CEFR mapping
# ---------------------------------------------------------------------------

def _map_by_threshold(val: object, thresholds: list[tuple[float, str]]) -> str:
    """Maps a numeric value based on sorted thresholds.

    Logic: first (bound, level) whose bound > val → returns level.
    If no bound is large enough, the last level is used as a fallback.

    Args:
        val:        Raw value (converted to float).
        thresholds: Ascending sorted list of (upper_bound_exclusive, level).

    Returns:
        CEFR level string; on conversion error, the cleaned raw value.
    """
    try:
        v = float(val)
        for bound, level in thresholds:
            if v < bound:
                return level
        return thresholds[-1][1]           # Fallback: everything ≥ last threshold
    except (ValueError, TypeError):
        return str(val).upper().strip()


def _map_with_rounding(val: object, spec: DatasetSpec) -> str:
    """Rounds a numeric value and maps it via cefr_level_map.

    Args:
        val:  Raw value (converted to float).
        spec: DatasetSpec with cefr_round_mode and optional cefr_level_map.

    Returns:
        CEFR level string (or the rounded integer as a string, if
        no cefr_level_map is set).
    """
    try:
        v = float(val)
        if spec.cefr_round_mode == "floor":
            rounded = int(math.floor(v))
        elif spec.cefr_round_mode == "ceil":
            rounded = int(math.ceil(v))
        else:                              # "round"
            rounded = round(v)
        key = str(rounded)
        return spec.cefr_level_map.get(key, key) if spec.cefr_level_map else key
    except (ValueError, TypeError):
        raw = str(val).upper().strip()
        return spec.cefr_level_map.get(raw, raw) if spec.cefr_level_map else raw


def _resolve_cefr_levels(series: pd.Series, spec: DatasetSpec) -> pd.Series:
    """Selects the configured mapping strategy and returns a CEFR series.

    Priority: cefr_level_thresholds > cefr_numeric_round > cefr_level_map > raw value.
    """
    if spec.cefr_level_thresholds is not None:
        thresholds_sorted = sorted(spec.cefr_level_thresholds, key=lambda x: x[0])
        return series.map(lambda v: _map_by_threshold(v, thresholds_sorted))

    if spec.cefr_numeric_round:
        return series.map(lambda v: _map_with_rounding(v, spec))

    raw = series.astype(str).str.upper().str.strip()
    return raw.map(spec.cefr_level_map) if spec.cefr_level_map else raw


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Loads a local CSV or JSON/JSONL file as a DataFrame.

    The file type is detected by the extension:
        .csv             -> comma-separated
        .tsv             -> tab-separated
        .json            -> regular JSON (list of objects or records-orient)
        .jsonl / .ndjson -> one JSON object per line

    Args:
        path: Path to the input file.

    Returns:
        Raw DataFrame (not yet standardized).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    suffix = p.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(p)
    elif suffix == ".tsv":
        return pd.read_csv(p, sep="\t")
    elif suffix == ".json":
        return pd.read_json(p)
    elif suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(p, lines=True)
    else:
        raise ValueError(
            f"Unsupported file format '{suffix}'. "
            "Allowed are: .csv, .tsv, .json, .jsonl, .ndjson"
        )


# ---------------------------------------------------------------------------
# Standardizing
# ---------------------------------------------------------------------------

def _resolve_field(
    df: pd.DataFrame,
    value: Optional[str],
    is_column: bool,
    fallback: object = None,
) -> pd.Series:
    """Returns a series – either from a column or as a fixed value.

    Args:
        df:        Raw DataFrame.
        value:     Column name or fixed value (or None).
        is_column: True → value is a column name; False → fixed value.
        fallback:  Used if value is None and is_column is False.

    Returns:
        pd.Series with the resolved value.
    """
    if is_column and value is not None:
        return df[value]
    if value is not None:
        return pd.Series([value] * len(df), index=df.index)
    return pd.Series([fallback] * len(df), index=df.index)


def standardize(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """Brings a raw DataFrame into the standard schema (STANDARD_COLUMNS)."""

    out = pd.DataFrame(index=df.index)

    # title: column, fixed value, or auto-numbered
    out["title"] = _resolve_field(
        df, spec.title, spec.title_is_column, fallback=range(len(df))
    )
    # Special case: fallback is range → assign directly
    if not spec.title_is_column and spec.title is None:
        out["title"] = range(len(df))

    out["lang"]        = _resolve_field(df, spec.lang,        spec.lang_is_column,        fallback="na")
    out["source_name"] = _resolve_field(df, spec.source_name, spec.source_name_is_column, fallback="na")
    out["format"]      = _resolve_field(df, spec.format,      spec.format_is_column,      fallback="na")
    out["category"]    = _resolve_field(df, spec.category,    spec.category_is_column,    fallback="na")

    out["cefr_level"]  = _resolve_cefr_levels(df[spec.cefr_col], spec)

    invalid = ~out["cefr_level"].isin(VALID_CEFR_LEVELS)
    if invalid.any():
        bad_values = out.loc[invalid, "cefr_level"].unique()[:5]
        print(f"[Warning] {invalid.sum()} rows with invalid CEFR level, e.g. {bad_values}")

    out["license"] = _resolve_field(df, spec.license, spec.license_is_column, fallback="na")
    out["text"]    = df[spec.text_col]

    return out[STANDARD_COLUMNS]


# ---------------------------------------------------------------------------
# Pipeline: load + standardize + save as CSV
# ---------------------------------------------------------------------------

def process_dataset(
    input_path: str,
    spec: DatasetSpec,
    output_path: str,
) -> pd.DataFrame:
    """Loads, standardizes, and writes the result as a CSV.

    Args:
        input_path:  Path to the source file (.csv, .tsv, .json, .jsonl).
        spec:        Mapping description for this dataset.
        output_path: Target path for the standardized CSV.

    Returns:
        The standardized DataFrame.
    """
    raw = load_data(input_path)
    df_std = standardize(raw, spec)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_std.to_csv(out, index=False)
    print(f"[OK] {len(df_std)} rows saved -> {out}")

    return df_std


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    # --- Example 1: Fixed values for all metadata -----------------------
    spec_csv = DatasetSpec(
        text_col="text",
        cefr_col="cefrLevel",
        
        lang="de",
        license="na",
        source_name="Ahlers",
        title="na",          title_is_column=False,   # "title" column in the original
        format="paragraph-level",                                  # fixed value
        category="na",                                   # → "na"
    )
    df = process_dataset(
        input_path="/data/dataset_input.json",
        spec=spec_csv,
        output_path="/data/dataset_input-std.csv",
    )
    """
    # --- UniversalCEFR: all fields directly from the columns ---------------
    # Schema matches 1:1 → everything with _is_column=True
    spec_universal = DatasetSpec(
        title="title",                  title_is_column=True,
        lang="lang",                    lang_is_column=True,
        source_name="UniversalCEFR-elg_cefr_de",      source_name_is_column=False,        #"UniversalCEFR-"

        format="format",                format_is_column=True,
        category="category",            category_is_column=True,

        cefr_col="cefr_level",
        license="license",              license_is_column=True,
        text_col="text",
        )
    df = process_dataset(
        input_path="/data/dataset_input.json",
        spec=spec_universal,
        output_path="/data/dataset_input-std.csv",
    )
    
    # --- Example 2: Multilingual JSONL, language from column ---------------
    # spec_jsonl = DatasetSpec(
    #     text_col="text",
    #     cefr_col="label",
    #     cefr_level_map={"0": "A1", "1": "A2", "2": "B1", "3": "B2", "4": "C1", "5": "C2"},
    #     lang="lang",            lang_is_column=True,
    #     license="CC-BY-SA-4.0",
    #     source_name="universal_cefr",
    #     format="format",        format_is_column=True,
    #     category="category",    category_is_column=True,
    # )
    # df = process_dataset(
    #     input_path="daten/universal_cefr.jsonl",
    #     spec=spec_jsonl,
    #     output_path="standardized/universal_cefr_std.csv",
    # )

    # --- Example 3: Threshold mapping --------------------------------
    # spec_threshold = DatasetSpec(
    #     text_col="text",
    #     cefr_col="score",
    #     cefr_level_thresholds=[
    #         (1.5, "A1"), (2.5, "A2"), (3.5, "B1"),
    #         (4.5, "B2"), (5.5, "C1"), (float("inf"), "C2"),
    #     ],
    #     lang="de",
    #     license="CC-BY-4.0",
    #     source_name="numeric_scores_dataset",
    #     format="news",
    # )

    # --- Example 4: Numeric rounding ----------------------------------
    # spec_round = DatasetSpec(
    #     text_col="text",
    #     cefr_col="level_num",
    #     cefr_numeric_round=True,
    #     cefr_round_mode="round",
    #     cefr_level_map={"1": "A1", "2": "A2", "3": "B1",
    #                     "4": "B2", "5": "C1", "6": "C2"},
    #     lang="en",
    #     license="MIT",
    #     source_name="numeric_cefr_dataset",
    # )

    print(df.head())
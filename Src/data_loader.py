"""
data_loader.py
--------------
Ingestion module for the Data Pipeline & EDA Engine.
Supports: CSV, Excel (.xlsx/.xls), JSON, SQLite (.db)

Usage:
    loader = DataLoader("data/raw/ipl_matches.csv")
    df = loader.load()
"""

import os
import json
import sqlite3
import logging

import pandas as pd
import numpy as np
from pathlib import Path
from collections import deque
from typing import Optional

# ── Logger setup ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Custom Exceptions ─────────────────────────────────────────────────────────

class DataLoaderError(Exception):
    """Base exception for all DataLoader errors."""
    pass

class UnsupportedFormatError(DataLoaderError):
    """Raised when the file extension is not supported."""
    pass

class FileNotFoundError(DataLoaderError):
    """Raised when the input file does not exist."""
    pass

class EmptyDatasetError(DataLoaderError):
    """Raised when the loaded file has no rows."""
    pass

class SchemaValidationError(DataLoaderError):
    """Raised when required columns are missing from the dataset."""
    pass


# ── Schema Registry ───────────────────────────────────────────────────────────
# Maps dataset name → expected columns + preferred dtypes
# Acts as a dict-based schema registry (DSA: hash map)

SCHEMA_REGISTRY = {
    "ipl_matches": {
        "required_columns": ["id", "season", "city", "date", "team1", "team2", "winner"],
        "dtype_hints": {
            "id":     "int64",
            "season": "int64",
            "date":   "datetime64",
        },
    },
    "ipl_deliveries": {
        "required_columns": ["match_id", "inning", "batting_team", "bowling_team",
                             "over", "ball", "batsman", "bowler", "total_runs"],
        "dtype_hints": {
            "match_id":   "int64",
            "inning":     "int64",
            "over":       "int64",
            "total_runs": "int64",
        },
    },
    "generic": {
        "required_columns": [],   # no validation for unknown datasets
        "dtype_hints": {},
    },
}


# ── DataLoader Class ──────────────────────────────────────────────────────────

class DataLoader:
    """
    Loads a dataset from CSV / Excel / JSON / SQLite into a pandas DataFrame.

    Parameters
    ----------
    file_path : str | Path
        Path to the input file.
    schema_name : str, optional
        Key in SCHEMA_REGISTRY to validate columns. Defaults to 'generic'.
    sqlite_table : str, optional
        Table name to read when file is a SQLite .db file.
    encoding : str, optional
        File encoding. Defaults to 'utf-8'.

    Example
    -------
    >>> loader = DataLoader("data/raw/ipl_matches.csv", schema_name="ipl_matches")
    >>> df = loader.load()
    >>> print(loader.summary())
    """

    SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".db", ".sqlite"}

    def __init__(
        self,
        file_path: str,
        schema_name: str = "generic",
        sqlite_table: Optional[str] = None,
        encoding: str = "utf-8",
    ):
        self.file_path    = Path(file_path)
        self.schema_name  = schema_name
        self.sqlite_table = sqlite_table
        self.encoding     = encoding
        self.df: Optional[pd.DataFrame] = None

        # Pipeline step log — deque used as an ordered action history (DSA)
        self._action_log: deque = deque(maxlen=50)

    # ── Public Methods ────────────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """
        Main entry point. Validates the file, reads it, and returns a DataFrame.

        Returns
        -------
        pd.DataFrame
            Loaded dataset.

        Raises
        ------
        FileNotFoundError, UnsupportedFormatError, EmptyDatasetError,
        SchemaValidationError
        """
        self._validate_file_exists()
        self._validate_extension()

        ext = self.file_path.suffix.lower()
        logger.info(f"Loading file: {self.file_path.name}  (format: {ext})")

        if ext == ".csv":
            self.df = self._load_csv()
        elif ext in {".xlsx", ".xls"}:
            self.df = self._load_excel()
        elif ext == ".json":
            self.df = self._load_json()
        elif ext in {".db", ".sqlite"}:
            self.df = self._load_sqlite()

        self._validate_not_empty()
        self._apply_dtype_hints()
        self._validate_schema()

        logger.info(
            f"Loaded successfully — {self.df.shape[0]:,} rows × {self.df.shape[1]} columns"
        )
        self._action_log.append("load_complete")
        return self.df

    def summary(self) -> str:
        """Returns a human-readable summary of the loaded dataset."""
        if self.df is None:
            return "No dataset loaded yet. Call .load() first."

        missing = self.df.isnull().sum().sum()
        missing_pct = (missing / self.df.size) * 100

        lines = [
            "=" * 52,
            f"  Dataset : {self.file_path.name}",
            f"  Shape   : {self.df.shape[0]:,} rows × {self.df.shape[1]} columns",
            f"  Missing : {missing:,} values ({missing_pct:.1f}%)",
            f"  Memory  : {self.df.memory_usage(deep=True).sum() / 1024:.1f} KB",
            "-" * 52,
            "  Column dtypes:",
        ]
        for col, dtype in self.df.dtypes.items():
            lines.append(f"    {col:<30} {str(dtype)}")
        lines.append("=" * 52)
        return "\n".join(lines)

    def get_action_log(self) -> list:
        """Returns the ordered list of actions taken during loading."""
        return list(self._action_log)

    # ── Private: Validators ───────────────────────────────────────────────────

    def _validate_file_exists(self):
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"File not found: '{self.file_path}'\n"
                f"Check the path and try again."
            )
        self._action_log.append("file_exists_ok")

    def _validate_extension(self):
        ext = self.file_path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"Unsupported format: '{ext}'\n"
                f"Supported: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )
        self._action_log.append(f"extension_ok:{ext}")

    def _validate_not_empty(self):
        if self.df is None or self.df.empty:
            raise EmptyDatasetError(
                f"The file '{self.file_path.name}' loaded 0 rows. "
                f"Check if the file has data."
            )
        self._action_log.append("not_empty_ok")

    def _validate_schema(self):
        schema = SCHEMA_REGISTRY.get(self.schema_name, SCHEMA_REGISTRY["generic"])
        required = schema["required_columns"]
        if not required:
            return  # generic schema — skip validation

        missing_cols = [c for c in required if c not in self.df.columns]
        if missing_cols:
            raise SchemaValidationError(
                f"Schema '{self.schema_name}' validation failed.\n"
                f"Missing columns: {missing_cols}\n"
                f"Found columns  : {list(self.df.columns)}"
            )
        logger.info(f"Schema validation passed for '{self.schema_name}'")
        self._action_log.append("schema_ok")

    # ── Private: Readers ──────────────────────────────────────────────────────

    def _load_csv(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.file_path, encoding=self.encoding, low_memory=False)
            self._action_log.append("read_csv")
            return df
        except UnicodeDecodeError:
            logger.warning("UTF-8 failed, retrying with latin-1 encoding...")
            df = pd.read_csv(self.file_path, encoding="latin-1", low_memory=False)
            self._action_log.append("read_csv_latin1_fallback")
            return df
        except Exception as e:
            raise DataLoaderError(f"Failed to read CSV: {e}") from e

    def _load_excel(self) -> pd.DataFrame:
        try:
            df = pd.read_excel(self.file_path, engine="openpyxl")
            self._action_log.append("read_excel")
            return df
        except Exception as e:
            raise DataLoaderError(f"Failed to read Excel file: {e}") from e

    def _load_json(self) -> pd.DataFrame:
        try:
            with open(self.file_path, "r", encoding=self.encoding) as f:
                raw = json.load(f)

            # Handle both list-of-dicts and nested dict formats
            if isinstance(raw, list):
                df = pd.DataFrame(raw)
            elif isinstance(raw, dict):
                df = pd.DataFrame.from_dict(raw, orient="index")
            else:
                raise DataLoaderError("JSON must be a list or a dict at the top level.")

            self._action_log.append("read_json")
            return df
        except json.JSONDecodeError as e:
            raise DataLoaderError(f"Invalid JSON file: {e}") from e
        except Exception as e:
            raise DataLoaderError(f"Failed to read JSON: {e}") from e

    def _load_sqlite(self) -> pd.DataFrame:
        try:
            conn = sqlite3.connect(self.file_path)

            # If no table specified, pick the first one found
            if self.sqlite_table is None:
                tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
                if tables.empty:
                    raise DataLoaderError("No tables found in the SQLite database.")
                self.sqlite_table = tables["name"].iloc[0]
                logger.info(f"Auto-selected SQLite table: '{self.sqlite_table}'")

            df = pd.read_sql(f"SELECT * FROM {self.sqlite_table}", conn)
            conn.close()
            self._action_log.append(f"read_sqlite:{self.sqlite_table}")
            return df
        except Exception as e:
            raise DataLoaderError(f"Failed to read SQLite: {e}") from e

    # ── Private: Type coercion ────────────────────────────────────────────────

    def _apply_dtype_hints(self):
        schema = SCHEMA_REGISTRY.get(self.schema_name, SCHEMA_REGISTRY["generic"])
        hints  = schema.get("dtype_hints", {})

        for col, dtype in hints.items():
            if col not in self.df.columns:
                continue
            try:
                if dtype == "datetime64":
                    self.df[col] = pd.to_datetime(self.df[col], infer_datetime_format=True)
                else:
                    self.df[col] = self.df[col].astype(dtype)
                logger.debug(f"  dtype coercion: '{col}' → {dtype}")
            except (ValueError, TypeError) as e:
                logger.warning(f"  Could not cast '{col}' to {dtype}: {e}")

        self._action_log.append("dtype_hints_applied")
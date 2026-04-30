import io
import tempfile
import subprocess
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import textwrap
import sys

# ✅ NEW: Safe dtype helpers
from pandas.api.types import (
    is_numeric_dtype,
    is_datetime64_any_dtype
)

# Optional duckdb import
try:
    import duckdb
except Exception:
    duckdb = None


# ----------------- Load data -----------------
def _looks_like_csv(raw_bytes: bytes) -> bool:
    try:
        sample = raw_bytes[:1024].decode(errors="ignore")
    except Exception:
        return False
    return "," in sample and "\n" in sample


def load_data(file_or_path) -> pd.DataFrame:
    if isinstance(file_or_path, (str, Path)):
        p = Path(file_or_path)
        s = p.suffix.lower()
        if s == ".csv":
            return pd.read_csv(p)
        if s in {".xls", ".xlsx"}:
            return pd.read_excel(p)
        if s == ".json":
            return pd.read_json(p)
        return pd.read_csv(p)

    name = getattr(file_or_path, "name", None)
    suffix = Path(name).suffix.lower() if name else None
    raw = file_or_path.read()
    if isinstance(raw, str):
        raw = raw.encode("utf-8")

    bio = io.BytesIO(raw)

    if suffix == ".csv" or (suffix is None and _looks_like_csv(raw)):
        bio.seek(0)
        return pd.read_csv(bio)

    if suffix in {".xls", ".xlsx"}:
        bio.seek(0)
        return pd.read_excel(bio)

    if suffix == ".json":
        bio.seek(0)
        return pd.read_json(bio)

    bio.seek(0)
    try:
        return pd.read_csv(bio)
    except Exception:
        bio.seek(0)
        return pd.read_json(bio)


# ----------------- FIXED COLUMN DETECTION -----------------
def _detect_column_types(df: pd.DataFrame):
    numeric = []
    datetime = []
    categorical = []

    for c in df.columns:
        col = df[c]

        # ✅ Safe numeric detection
        if is_numeric_dtype(col):
            numeric.append(c)

        # ✅ Safe datetime detection
        elif is_datetime64_any_dtype(col):
            datetime.append(c)

        else:
            # Try parsing datetime from strings
            try:
                parsed = pd.to_datetime(col, errors='coerce')
                if parsed.notna().sum() > len(col) * 0.6:
                    datetime.append(c)
                else:
                    categorical.append(c)
            except Exception:
                categorical.append(c)

    return {
        "numeric": numeric,
        "datetime": datetime,
        "categorical": categorical
    }


# ----------------- Suggestions -----------------
def suggest_prompts(df: pd.DataFrame, max_suggestions: int = 8):
    types = _detect_column_types(df)
    numeric = types["numeric"]
    datetime = types["datetime"]
    categorical = types["categorical"]

    suggestions = []
    suggestions.append("Summarize the dataset in 5 bullet points.")

    if categorical:
        suggestions.append(f"Show top 10 values for '{categorical[0]}'")

    if numeric:
        suggestions.append("Show summary statistics for numeric columns.")
        suggestions.append(f"Create histogram for '{numeric[0]}'")

    if len(numeric) >= 2:
        suggestions.append(f"Scatter plot '{numeric[0]}' vs '{numeric[1]}'")

    if datetime:
        suggestions.append(f"Monthly trend using '{datetime[0]}'")

    suggestions.append("Detect anomalies using z-score")

    return suggestions[:max_suggestions]


# ----------------- Prompt to code -----------------
def prompt_to_code(prompt: str, df: pd.DataFrame):
    import re
    p = prompt.lower()

    if "summary" in p:
        return "result = df.describe(include='all').T"

    if "top 10" in p:
        col = re.findall(r"'([^']+)'", prompt)
        if col:
            return f"result = df['{col[0]}'].value_counts().head(10)"

    if "histogram" in p:
        col = re.findall(r"'([^']+)'", prompt)
        if col:
            return f"""
plt.figure()
df['{col[0]}'].dropna().hist()
result_img_path=None
"""

    if "scatter" in p:
        col = re.findall(r"'([^']+)'", prompt)
        if len(col) >= 2:
            return f"""
plt.figure()
df.plot.scatter(x='{col[0]}', y='{col[1]}')
result_img_path=None
"""

    return None


# ----------------- Run code -----------------
def run_code(df: pd.DataFrame, code: str):
    local_ns = {"pd": pd, "np": np, "df": df, "plt": plt}

    try:
        exec(code, {}, local_ns)

        figs = plt.get_fignums()
        if figs:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
                plt.savefig(f.name)
                plt.close("all")
                return {"type": "image", "path": f.name}

        if "result" in local_ns:
            res = local_ns["result"]
            if isinstance(res, pd.DataFrame):
                return {"type": "dataframe", "df": res}
            return {"type": "text", "output": str(res)}

        return {"type": "text", "output": "No output"}

    except Exception as e:
        return {"type": "text", "output": str(e)}


# ----------------- LLM -----------------
def ask_llm(prompt: str, model: str = "llama3.1"):
    try:
        proc = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return proc.stdout.decode()
    except Exception:
        return "[LLM unavailable]"
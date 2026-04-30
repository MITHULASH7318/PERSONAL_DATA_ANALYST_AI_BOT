import io
import tempfile
import subprocess
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import textwrap
import sys
import os

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
        bio.seek(0); return pd.read_csv(bio)
    if suffix in {".xls", ".xlsx"}:
        bio.seek(0); return pd.read_excel(bio)
    if suffix == ".json":
        bio.seek(0); return pd.read_json(bio)
    bio.seek(0)
    try:
        return pd.read_csv(bio)
    except Exception:
        bio.seek(0); return pd.read_json(bio)

# ----------------- Column detection & suggestions -----------------
def _detect_column_types(df: pd.DataFrame):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime = []
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            datetime.append(c)
        else:
            try:
                sample = df[c].dropna().astype(str).iloc[:20]
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().sum() >= max(1, min(5, len(sample)//2)):
                    datetime.append(c)
            except Exception:
                pass
    categorical = [c for c in df.columns if c not in numeric + datetime and df[c].nunique(dropna=True) <= 50]
    return {"numeric": numeric, "datetime": datetime, "categorical": categorical}

def suggest_prompts(df: pd.DataFrame, max_suggestions: int = 8):
    types = _detect_column_types(df)
    numeric = types["numeric"]
    datetime = types["datetime"]
    categorical = types["categorical"]

    suggestions = []
    suggestions.append("Summarize the dataset in 5 bullet points (rows, columns, missing values, numeric columns, top categorical).")
    if categorical:
        col = categorical[0]
        suggestions.append(f"Show the top 10 counts for the categorical column '{col}'.")
    if numeric:
        suggestions.append(f"Show summary statistics (count, mean, std, min, 25%, 50%, 75%, max) for numeric columns.")
        col = numeric[0]
        suggestions.append(f"Create a histogram of the numeric column '{col}'.")
        if len(numeric) >= 2:
            suggestions.append(f"Create a scatter plot comparing '{numeric[0]}' (x) vs '{numeric[1]}' (y).")
        suggestions.append(f"Show the top 10 rows sorted by '{col}' descending.")
    if datetime:
        dcol = datetime[0]
        ag = numeric[0] if numeric else None
        if ag:
            suggestions.append(f"Create a time series of monthly sum of '{ag}' using the datetime column '{dcol}'.")
        else:
            suggestions.append(f"Show counts per month using the datetime column '{dcol}'.")
    if len(numeric) >= 2:
        suggestions.append("Show the correlation matrix heatmap for numeric columns.")
    suggestions.append("Find rows that look like anomalies using z-score > 3 on numeric columns and show top 20.")
    return suggestions[:max_suggestions]

# ----------------- Prompt to code -----------------
def prompt_to_code(prompt: str, df: pd.DataFrame):
    import re
    p = prompt.strip().lower()
    # Summary
    if p.startswith("summarize the dataset"):
        code = textwrap.dedent("""
            info = []
            info.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")
            info.append("Column types: " + ", ".join([f"{c}:{str(df[c].dtype)[:10]}" for c in df.columns[:10]]))
            miss = df.isnull().sum().sort_values(ascending=False).head(10)
            info.append("Top missing: " + ", ".join([f"{idx}:{val}" for idx,val in miss.items() if val>0]))
            numeric = df.select_dtypes(include=['number']).columns.tolist()
            info.append(f"Numeric columns count: {len(numeric)}")
            result = "\\n".join(["- "+i for i in info])
        """)
        return code
    # Top 10 counts categorical
    if "top 10 counts for the categorical column" in p and "'" in p:
        m = re.search(r"'([^']+)'", prompt)
        col = m.group(1) if m else None
        if col:
            code = textwrap.dedent(f"""
                result = df['{col}'].value_counts(dropna=False).head(10).reset_index()
                result.columns = ['value','count']
            """)
            return code
    # Summary statistics numeric
    if "summary statistics" in p or "describe" in p:
        return "result = df.select_dtypes(include=['number']).describe().T"
    # Histogram
    if "histogram of the numeric column" in p:
        m = re.search(r"'([^']+)'", prompt)
        col = m.group(1) if m else None
        if col:
            code = textwrap.dedent(f"""
                plt.figure(figsize=(6,4))
                df['{col}'].dropna().astype(float).hist(bins=30)
                plt.title('Histogram of {col}')
                plt.xlabel('{col}')
                plt.ylabel('count')
                result_img_path = None
            """)
            return code
    # Scatter plot
    if "scatter plot comparing" in p and "vs" in p:
        m = re.search(r"'([^']+)' \(x\) vs '([^']+)' \(y\)", prompt)
        if m:
            xcol, ycol = m.group(1), m.group(2)
            code = textwrap.dedent(f"""
                plt.figure(figsize=(6,4))
                df.plot.scatter(x='{xcol}', y='{ycol}')
                plt.title('{ycol} vs {xcol}')
                result_img_path = None
            """)
            return code
    # Top N rows sorted
    if p.startswith("show the top 10 rows sorted by"):
        m = re.search(r"by '([^']+)'", prompt)
        if m:
            col = m.group(1)
            return f"result = df.sort_values('{col}', ascending=False).head(10).reset_index(drop=True)"
    # Time series / counts per month / correlation / anomalies
    if "monthly sum" in p:
        m = re.search(r"sum of '([^']+)' using the datetime column '([^']+)'", prompt)
        if m:
            ag, dcol = m.group(1), m.group(2)
            code = textwrap.dedent(f"""
                tmp = df.copy()
                tmp['{dcol}'] = pd.to_datetime(tmp['{dcol}'], errors='coerce')
                res = tmp.dropna(subset=['{dcol}'])
                res = res.set_index('{dcol}').resample('M')['{ag}'].sum().reset_index()
                result = res
            """)
            return code
    if "counts per month using the datetime column" in p:
        m = re.search(r"datetime column '([^']+)'", prompt)
        dcol = m.group(1) if m else None
        if dcol:
            code = textwrap.dedent(f"""
                tmp = df.copy()
                tmp['{dcol}'] = pd.to_datetime(tmp['{dcol}'], errors='coerce')
                res = tmp.dropna(subset=['{dcol}']).set_index('{dcol}').resample('M').size().reset_index(name='count')
                result = res
            """)
            return code
    if "correlation matrix heatmap" in p:
        code = textwrap.dedent("""
            corr = df.select_dtypes(include=['number']).corr()
            plt.figure(figsize=(6,5))
            plt.imshow(corr, cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.xticks(range(len(corr)), corr.columns, rotation=90)
            plt.yticks(range(len(corr)), corr.columns)
            plt.title('Correlation matrix')
            result_img_path = None
        """)
        return code
    if "anomalies" in p and "z-score" in p:
        code = textwrap.dedent("""
            from scipy import stats
            num = df.select_dtypes(include=['number']).dropna()
            if num.shape[1]==0:
                result = pd.DataFrame()
            else:
                z = np.abs(stats.zscore(num.select_dtypes(include=['number'])))
                mask = (z > 3).any(axis=1)
                result = df.loc[mask].head(20).reset_index(drop=True)
        """)
        return code
    return None

# ----------------- Run code -----------------
def run_code(df: pd.DataFrame, code: str):
    local_ns = {"pd": pd, "np": np, "df": df, "plt": plt}
    old_stdout = sys.stdout
    stdout_buf = io.StringIO()
    sys.stdout = stdout_buf
    try:
        exec(code, {}, local_ns)
        if "result_img_path" in local_ns and local_ns["result_img_path"]:
            path = local_ns["result_img_path"]
            return {"type": "image", "path": path}
        figs = plt.get_fignums()
        if figs:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
                plt.savefig(f.name, bbox_inches="tight", dpi=150)
                plt.close("all")
                return {"type": "image", "path": f.name}
        if "result" in local_ns:
            res = local_ns["result"]
            if isinstance(res, pd.DataFrame):
                return {"type": "dataframe", "df": res}
            else:
                return {"type": "text", "output": str(res)}
        out = stdout_buf.getvalue().strip()
        if out:
            return {"type": "text", "output": out}
        return {"type": "text", "output": "Execution finished. No result produced."}
    except Exception as e:
        return {"type": "text", "output": f"Execution error: {e}"}
    finally:
        sys.stdout = old_stdout

# ----------------- LLM connector -----------------
def ask_llm(prompt: str, model: str = "llama3.1", timeout: int = 60) -> str:
    try:
        proc = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout
        )
        out = proc.stdout.decode("utf-8", errors="replace")
        err = proc.stderr.decode("utf-8", errors="replace")
        if not out and err:
            return f"[LLM-error] {err}"
        return out
    except FileNotFoundError:
        return "[LLM-missing] ollama not found on PATH."
    except Exception as e:
        return f"[LLM-failed] {e}"

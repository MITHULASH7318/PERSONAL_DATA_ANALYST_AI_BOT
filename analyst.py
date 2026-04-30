import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile, requests, os
from dotenv import load_dotenv

load_dotenv()

# ── LOAD DATA ─────────────────────────────
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith((".xls", ".xlsx")):
        return pd.read_excel(file)
    elif file.name.endswith(".json"):
        return pd.read_json(file)
    else:
        raise ValueError("Unsupported file")

# ── PROMPT SUGGESTIONS (BUSINESS FRIENDLY) ─────────────────
def suggest_prompts(df):
    return [
        "Show first 10 rows",
        "Show last 10 rows",
        "Show summary statistics",
        "Show mean of numeric columns",
        "Show median of numeric columns",
        "Show mode of dataset",
        "Summarize dataset in 5 business points",
        "Show correlation between columns",
        "Find missing values",
        "Top 5 highest values",
        "Top 5 lowest values"
    ]

# ── RULE ENGINE ───────────────────────────
def prompt_to_code(prompt, df):
    p = prompt.lower()

    if "first 10" in p:
        return "result = df.head(10)"

    if "last 10" in p:
        return "result = df.tail(10)"

    if "summary statistics" in p:
        return "result = df.describe()"

    if "mean" in p:
        return "result = df.select_dtypes(include=['number']).mean().to_frame(name='Mean')"

    if "median" in p:
        return "result = df.select_dtypes(include=['number']).median().to_frame(name='Median')"

    if "mode" in p:
        return "result = df.mode().head(1)"

    if "missing" in p:
        return "result = df.isnull().sum().to_frame(name='Missing Values')"

    if "correlation" in p:
        return "result = df.select_dtypes(include=['number']).corr()"

    if "top 5 highest" in p:
        col = df.select_dtypes(include=['number']).columns[0]
        return f"result = df.sort_values('{col}', ascending=False).head(5)"

    if "top 5 lowest" in p:
        col = df.select_dtypes(include=['number']).columns[0]
        return f"result = df.sort_values('{col}', ascending=True).head(5)"

    if "summarize" in p:
        return """
result = pd.DataFrame({
"Metric": ["Rows","Columns","Missing Values","Numeric Columns"],
"Value": [
len(df),
len(df.columns),
df.isnull().sum().sum(),
len(df.select_dtypes(include=['number']).columns)
]})
"""

    return None

# ── RUN CODE ──────────────────────────────
def run_code(df, code):
    local = {"df": df, "pd": pd, "np": np, "plt": plt}

    try:
        exec(code, {}, local)

        if "result" in local:
            return {"type": "dataframe", "df": local["result"]}

        figs = plt.get_fignums()
        if figs:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
                plt.savefig(f.name)
                plt.close()
                return {"type": "image", "path": f.name}

        return {"type": "text", "output": "No result"}

    except Exception as e:
        return {"type": "text", "output": str(e)}

# ── FIXED OPENROUTER ─────────────────────
def ask_llm(prompt):
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        return "[LLM unavailable] Missing API key"

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}]
            },
        )

        data = response.json()

        if "choices" not in data:
            return f"[LLM error] {data}"

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"[LLM error] {e}"
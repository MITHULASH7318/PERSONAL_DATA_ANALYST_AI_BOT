import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile, os, requests
from dotenv import load_dotenv

# Load local env
load_dotenv()

# ── LOAD DATA ─────────────────────────
def load_data(file):
    name = file.name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    elif name.endswith(".json"):
        return pd.read_json(file)
    else:
        return pd.read_csv(file)


# ── SMART PROMPT ENGINE ──────────────
def prompt_to_code(prompt, df):
    p = prompt.lower()

    # 🔹 BASIC ROWS
    if "first 5" in p:
        return "result = df.head(5)"

    if "first 10" in p:
        return "result = df.head(10)"

    if "first 20" in p:
        return "result = df.head(20)"

    # 🔹 SUMMARY
    if "summary" in p:
        return "result = df.describe()"

    if "mean" in p:
        return "result = df.mean(numeric_only=True)"

    if "median" in p:
        return "result = df.median(numeric_only=True)"

    if "mode" in p:
        return "result = df.mode().head(5)"

    # 🔹 HISTOGRAM
    if "histogram" in p:
        col = df.select_dtypes(include=np.number).columns[0]
        return f"""
plt.figure()
df['{col}'].dropna().hist()
plt.title('Histogram of {col}')
"""

    # 🔹 BAR CHART
    if "bar" in p:
        col = df.columns[0]
        return f"""
plt.figure()
df['{col}'].value_counts().head(10).plot(kind='bar')
plt.title('Bar Chart of {col}')
"""

    # 🔹 PIE CHART
    if "pie" in p:
        col = df.columns[0]
        return f"""
plt.figure()
df['{col}'].value_counts().head(5).plot(kind='pie', autopct='%1.1f%%')
plt.title('Pie Chart of {col}')
"""

    # 🔹 SCATTER
    if "scatter" in p and len(df.select_dtypes(include=np.number).columns) >= 2:
        cols = df.select_dtypes(include=np.number).columns
        return f"""
plt.figure()
df.plot.scatter(x='{cols[0]}', y='{cols[1]}')
plt.title('Scatter Plot')
"""

    return None


# ── RUN CODE ──────────────────────────
def run_code(df, code):
    local = {"df": df, "pd": pd, "np": np, "plt": plt}

    try:
        exec(code, {}, local)

        # 📊 IMAGE
        if plt.get_fignums():
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
                plt.savefig(f.name, bbox_inches="tight")
                plt.close()
                return {"type": "image", "path": f.name}

        # 📄 TABLE / TEXT
        if "result" in local:
            res = local["result"]
            if isinstance(res, pd.DataFrame):
                return {"type": "dataframe", "df": res}
            else:
                return {"type": "text", "output": str(res)}

        return {"type": "text", "output": "Done"}

    except Exception as e:
        return {"type": "text", "output": f"Error: {e}"}


# ── LLM (OPENROUTER FIXED) ────────────
def ask_llm(prompt):
    import streamlit as st

    # 🔥 FIX: support BOTH local + deployment
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets["OPENROUTER_API_KEY"]
        except:
            return "[LLM unavailable] Missing API key"

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [
            {"role": "system", "content": "Return ONLY Python code using pandas and matplotlib."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        res = requests.post(url, headers=headers, json=data)
        res_json = res.json()

        if "choices" not in res_json:
            return f"[LLM error] {res_json}"

        return res_json["choices"][0]["message"]["content"]

    except Exception as e:
        return f"[LLM error] {e}"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, tempfile, sys, os, requests
from dotenv import load_dotenv

load_dotenv()

# ── LOAD DATA ─────────────────────────
def load_data(file):
    name = file.name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file)
    elif name.endswith(".json"):
        return pd.read_json(file)
    else:
        return pd.read_csv(file)

# ── SIMPLE PROMPT ENGINE ──────────────
def prompt_to_code(prompt, df):
    p = prompt.lower()

    if "first 5" in p:
        return "result = df.head(5)"

    if "first 10" in p:
        return "result = df.head(10)"

    if "mean" in p:
        return "result = df.mean(numeric_only=True)"

    if "median" in p:
        return "result = df.median(numeric_only=True)"

    return None

# ── RUN CODE ──────────────────────────
def run_code(df, code):
    local = {"df": df, "pd": pd, "np": np, "plt": plt}

    try:
        exec(code, {}, local)

        if "result" in local:
            if isinstance(local["result"], pd.DataFrame):
                return {"type": "dataframe", "df": local["result"]}
            else:
                return {"type": "text", "output": str(local["result"])}

        figs = plt.get_fignums()
        if figs:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
                plt.savefig(f.name)
                plt.close()
                return {"type": "image", "path": f.name}

        return {"type": "text", "output": "Done"}

    except Exception as e:
        return {"type": "text", "output": str(e)}

# ── LLM (OPENROUTER) ──────────────────
def ask_llm(prompt):
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        return "[LLM unavailable] Missing API key"

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [
            {"role": "system", "content": "Return only Python code using pandas."},
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
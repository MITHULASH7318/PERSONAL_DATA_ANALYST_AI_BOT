import streamlit as st
from analyst import load_data, suggest_prompts, prompt_to_code, run_code, ask_llm

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PERSONAL CHATBOT BUDA",
    page_icon="◈",
    layout="wide",
)

# ── PREMIUM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400&display=swap');

:root {
    --bg: #0b0b0d;
    --surface: #111114;
    --surface2: #18181c;
    --border: #23232a;
    --text: #e6e6e9;
    --text-dim: #8c8c98;
    --accent: #d4af37;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem !important; }

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

.stButton > button {
    background: var(--accent) !important;
    color: black !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: none !important;
}

textarea, input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}

.stat-box {
    background: var(--surface2);
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid var(--border);
    text-align: center;
}

.title-main {
    font-size: 2.4rem;
    font-weight: 700;
}

.subtitle {
    color: var(--text-dim);
    font-size: 0.9rem;
}

.ai-badge {
    border: 1px solid var(--border);
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 0.7rem;
    color: var(--accent);
}
</style>
""", unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;
margin-bottom:25px;border-bottom:1px solid #23232a;padding-bottom:15px;">

<div>
    <div class="title-main">PERSONAL CHATBOT <span style="color:#d4af37;">BUDA</span></div>
    <div class="subtitle">Business Data Analyst ChatBot</div>
</div>

<div class="ai-badge">AI AGENT • ACTIVE</div>
</div>
""", unsafe_allow_html=True)

# ── AI ICON ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:right;margin-top:-10px;margin-bottom:20px;">
    <img src="https://cdn-icons-png.flaticon.com/512/4712/4712027.png" width="40">
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR (SIMPLIFIED FOR USERS) ───────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ System")
    st.info("AI Powered Analysis Enabled")

# ── FILE UPLOAD ──────────────────────────────────────────────────────────────
st.subheader("Upload Dataset")
uploaded = st.file_uploader("Upload CSV / Excel / JSON", type=["csv","xlsx","xls","json"])

if uploaded is None:
    st.info("Upload a dataset to begin analysis.")
    st.stop()

try:
    df = load_data(uploaded)
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

# ── STATS ────────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.markdown(f'<div class="stat-box"><b>Rows</b><br>{len(df)}</div>', unsafe_allow_html=True)
c2.markdown(f'<div class="stat-box"><b>Columns</b><br>{len(df.columns)}</div>', unsafe_allow_html=True)
c3.markdown(f'<div class="stat-box"><b>File</b><br>{uploaded.name}</div>', unsafe_allow_html=True)

# ── PREVIEW ──────────────────────────────────────────────────────────────────
with st.expander("Preview Data"):
    st.dataframe(df.head(100), use_container_width=True)

# ── PROMPT SECTION ───────────────────────────────────────────────────────────
st.subheader("Choose Analysis")

suggestions = suggest_prompts(df)

selected = st.selectbox("Suggested", suggestions)
custom = st.text_area("Custom Prompt")

final_prompt = custom.strip() if custom.strip() else selected

st.code(final_prompt, language="text")

# ── RUN ANALYSIS ─────────────────────────────────────────────────────────────
if st.button("Run Analysis"):

    with st.spinner("Analyzing..."):

        # First try rule-based system
        code = prompt_to_code(final_prompt, df)

        if code:
            res = run_code(df, code)

        else:
            # 🔥 OpenRouter LLM
            system = """
You are a professional business data analyst.

STRICT RULES:
- Return ONLY Python code
- Wrap code inside ```python ... ```
- DataFrame name is df
- Use pandas and matplotlib only
- Do NOT explain anything
"""

            llm_out = ask_llm(system + "\nUser request: " + final_prompt)

            if llm_out.startswith("[LLM"):
                st.error(llm_out)
                st.stop()

            if "```python" in llm_out:
                code = llm_out.split("```python")[1].split("```")[0]
                res = run_code(df, code)
            else:
                st.error("LLM did not return valid Python code")
                st.code(llm_out)
                st.stop()

    # ── OUTPUT ───────────────────────────────────────────────────────────────
    if res["type"] == "text":
        st.code(res["output"])

    elif res["type"] == "dataframe":
        st.dataframe(res["df"])
        csv = res["df"].to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="result.csv")

    elif res["type"] == "image":
        st.image(res["path"])

    else:
        st.write(res)
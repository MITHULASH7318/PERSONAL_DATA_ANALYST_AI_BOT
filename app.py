import streamlit as st
from analyst import load_data, suggest_prompts, prompt_to_code, run_code, ask_llm
import pandas as pd

st.set_page_config(
    page_title="PERSONAL CHATBOT BUDA",
    layout="wide"
)

# ── MODERN SaaS UI ─────────────────────────────────────────
st.markdown("""
<style>
body {background-color:#0d0f14; color:#e6e6e6;}
.block-container {padding:2rem 3rem;}
h1 {font-size:2.2rem;}
.card {
    background:#151821;
    padding:1rem;
    border-radius:12px;
    border:1px solid #222;
}
button {
    background:#d4af37 !important;
    color:black !important;
    border-radius:8px !important;
}
</style>
""", unsafe_allow_html=True)

# ── HEADER ─────────────────────────────────────────
st.markdown("""
# 💼 PERSONAL CHATBOT BUDA  
### Business Data Analyst Dashboard
""")

# ── FILE UPLOAD ───────────────────────────────────
st.subheader("📂 Upload Dataset")
file = st.file_uploader("Upload CSV / Excel / JSON", type=["csv","xlsx","json"])

if not file:
    st.stop()

df = load_data(file)

# ── METRICS ───────────────────────────────────────
c1,c2,c3 = st.columns(3)
c1.metric("Rows", len(df))
c2.metric("Columns", len(df.columns))
c3.metric("File", file.name)

# ── PREVIEW ───────────────────────────────────────
with st.expander("Preview Data"):
    st.dataframe(df.head(100), use_container_width=True)

# ── ANALYSIS OPTIONS ──────────────────────────────
st.subheader("📊 Choose Analysis")

analysis_type = st.selectbox("Select Analysis Type", [
    "Summary (5 points)",
    "Mean / Median / Mode",
    "Bar Chart",
    "Pie Chart",
    "Histogram",
    "Scatter Plot",
    "Correlation Heatmap",
    "Custom Prompt (AI)"
])

# COLUMN SELECTION
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(exclude="number").columns.tolist()

col1, col2 = st.columns(2)

with col1:
    x_col = st.selectbox("Select Column X", df.columns)

with col2:
    y_col = st.selectbox("Select Column Y (optional)", df.columns)

custom_prompt = st.text_area("Custom Prompt (optional)")

# ── RUN ───────────────────────────────────────────
if st.button("🚀 Run Analysis"):

    # ── SUMMARY ─────────────────
    if analysis_type == "Summary (5 points)":
        st.write(f"""
        • Rows: {len(df)}  
        • Columns: {len(df.columns)}  
        • Missing values: {df.isnull().sum().sum()}  
        • Numeric columns: {len(num_cols)}  
        • Categorical columns: {len(cat_cols)}
        """)

    # ── MEAN MEDIAN MODE ───────
    elif analysis_type == "Mean / Median / Mode":
        col = x_col
        st.write("Mean:", df[col].mean())
        st.write("Median:", df[col].median())
        st.write("Mode:", df[col].mode().values)

    # ── BAR CHART ─────────────
    elif analysis_type == "Bar Chart":
        st.bar_chart(df[x_col].value_counts().head(10))

    # ── PIE CHART ─────────────
    elif analysis_type == "Pie Chart":
        st.pyplot(df[x_col].value_counts().head(10).plot.pie(autopct='%1.1f%%').figure)

    # ── HISTOGRAM ─────────────
    elif analysis_type == "Histogram":
        st.bar_chart(df[x_col])

    # ── SCATTER ───────────────
    elif analysis_type == "Scatter Plot":
        if y_col:
            st.scatter_chart(df[[x_col, y_col]])

    # ── HEATMAP ───────────────
    elif analysis_type == "Correlation Heatmap":
        import matplotlib.pyplot as plt
        import numpy as np

        corr = df[num_cols].corr()
        fig, ax = plt.subplots()
        cax = ax.matshow(corr)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        fig.colorbar(cax)
        st.pyplot(fig)

    # ── AI MODE ───────────────
    elif analysis_type == "Custom Prompt (AI)":
        if not custom_prompt:
            st.error("Enter a prompt")
        else:
            code = prompt_to_code(custom_prompt, df)

            if code:
                res = run_code(df, code)
            else:
                llm_out = ask_llm(custom_prompt)

                if "```python" in llm_out:
                    code = llm_out.split("```python")[1].split("```")[0]
                    res = run_code(df, code)
                else:
                    st.error("LLM failed")
                    st.code(llm_out)
                    st.stop()

            if res["type"] == "dataframe":
                st.dataframe(res["df"], use_container_width=True)
            elif res["type"] == "image":
                st.image(res["path"])
            else:
                st.write(res["output"])
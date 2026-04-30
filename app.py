import streamlit as st
from analyst import load_data, ask_llm, run_code, prompt_to_code

st.set_page_config(page_title="PERSONAL CHATBOT BUDA", layout="wide")

# ── SaaS UI ─────────────────────────────────────────
st.markdown("""
<style>
body {background:#0b0d12; color:#e8e8e8;}
.block-container {padding:2rem 3rem;}
.title {font-size:2.2rem; font-weight:700;}
.sub {color:#9aa0aa; font-size:0.9rem;}
.card {
    background:#141821;
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

# ── HEADER ─────────────────────────
st.markdown("""
<div class="title">💼 PERSONAL CHATBOT <span style="color:#d4af37;">BUDA</span></div>
<div class="sub">Business Data Analyst Dashboard</div>
<hr>
""", unsafe_allow_html=True)

# ── FILE ───────────────────────────
file = st.file_uploader("Upload Dataset", type=["csv","xlsx","json"])

if not file:
    st.stop()

df = load_data(file)

# ── METRICS ───────────────────────
c1,c2,c3 = st.columns(3)
c1.metric("Rows", len(df))
c2.metric("Columns", len(df.columns))
c3.metric("File", file.name)

# ── PREVIEW ───────────────────────
with st.expander("Preview Data"):
    st.dataframe(df.head(100), use_container_width=True)

# ── ANALYSIS ──────────────────────
st.subheader("📊 Analysis")

option = st.selectbox("Choose Analysis", [
    "Show Rows",
    "Summary",
    "Statistics (Mean/Median/Mode)",
    "Value Counts",
    "Bar Chart",
    "Pie Chart",
    "Histogram",
    "Scatter Plot",
    "Correlation Heatmap",
    "Custom AI Prompt"
])

col1, col2 = st.columns(2)

col_x = col1.selectbox("Column X", df.columns)
col_y = col2.selectbox("Column Y", df.columns)

# ── EXTRA OPTIONS (SMART UI) ───────

limit = 10
if option == "Show Rows":
    limit = st.selectbox("How many rows?", [5, 10, 20, 50])

top_n = 10
if option in ["Bar Chart", "Pie Chart", "Value Counts"]:
    top_n = st.selectbox("Top values to show", [5, 10, 15, 20])

bins = 20
if option == "Histogram":
    bins = st.selectbox("Bins", [10, 20, 30, 50])

prompt = ""
if option == "Custom AI Prompt":
    prompt = st.text_area("Enter your prompt")

# ── RUN ───────────────────────────
if st.button("🚀 Run Analysis"):

    # ── SHOW ROWS ────────────────
    if option == "Show Rows":
        st.dataframe(df.head(limit), use_container_width=True)

    # ── SUMMARY ──────────────────
    elif option == "Summary":
        st.markdown(f"""
        • Rows: {len(df)}  
        • Columns: {len(df.columns)}  
        • Missing Values: {df.isnull().sum().sum()}  
        • Numeric Columns: {len(df.select_dtypes(include='number').columns)}  
        • Categorical Columns: {len(df.select_dtypes(exclude='number').columns)}
        """)

    # ── STATS ────────────────────
    elif option == "Statistics (Mean/Median/Mode)":
        st.write("Mean:", df[col_x].mean())
        st.write("Median:", df[col_x].median())
        st.write("Mode:", df[col_x].mode().values)

    # ── VALUE COUNTS ─────────────
    elif option == "Value Counts":
        st.dataframe(df[col_x].value_counts().head(top_n).reset_index())

    # ── BAR ──────────────────────
    elif option == "Bar Chart":
        st.bar_chart(df[col_x].value_counts().head(top_n))

    # ── PIE ──────────────────────
    elif option == "Pie Chart":
        st.pyplot(df[col_x].value_counts().head(top_n).plot.pie(autopct='%1.1f%%').figure)

    # ── HISTOGRAM ────────────────
    elif option == "Histogram":
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        df[col_x].dropna().hist(bins=bins, ax=ax)
        st.pyplot(fig)

    # ── SCATTER ──────────────────
    elif option == "Scatter Plot":
        st.scatter_chart(df[[col_x, col_y]])

    # ── HEATMAP ──────────────────
    elif option == "Correlation Heatmap":
        import matplotlib.pyplot as plt
        corr = df.select_dtypes(include='number').corr()
        fig, ax = plt.subplots()
        cax = ax.matshow(corr)
        fig.colorbar(cax)
        st.pyplot(fig)

    # ── AI ───────────────────────
    elif option == "Custom AI Prompt":

        if not prompt:
            st.error("Enter prompt")
            st.stop()

        code = prompt_to_code(prompt, df)

        if code:
            res = run_code(df, code)

        else:
            llm_out = ask_llm(prompt)

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
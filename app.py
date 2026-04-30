import streamlit as st
from analyst import load_data, ask_llm, run_code, prompt_to_code

st.set_page_config(page_title="PERSONAL CHATBOT BUDA", layout="wide")

# ── PREMIUM SaaS CSS ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0b0c10;
    color: #e6e6e6;
}

.block-container {
    padding: 2rem 3rem;
}

h1, h2, h3 {
    font-weight: 600;
}

.metric-card {
    background: #151821;
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid #222;
    text-align: center;
}

button {
    background: linear-gradient(135deg, #d4af37, #b8962e) !important;
    color: black !important;
    border-radius: 8px !important;
    border: none !important;
    font-weight: 600 !important;
}

textarea, input {
    background: #151821 !important;
    border: 1px solid #222 !important;
    color: #e6e6e6 !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #222;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ── HEADER ─────────────────────────────────────────
st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;
border-bottom:1px solid #222;padding-bottom:15px;margin-bottom:20px;">

<div>
<h1>💼 PERSONAL CHATBOT <span style="color:#d4af37;">BUDA</span></h1>
<p style="color:#888;">Business Data Analyst Dashboard</p>
</div>

<div style="color:#d4af37;font-size:0.8rem;">
AI AGENT ● ACTIVE
</div>

</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("Professional Data Analysis Tool")

# ── FILE UPLOAD ─────────────────────────────────────
st.subheader("📂 Upload Dataset")
file = st.file_uploader("", type=["csv","xlsx","json"])

if not file:
    st.info("Upload a dataset to begin analysis")
    st.stop()

df = load_data(file)

# ── METRICS ─────────────────────────────────────────
c1, c2, c3 = st.columns(3)

c1.markdown(f'<div class="metric-card"><b>Rows</b><br>{len(df)}</div>', unsafe_allow_html=True)
c2.markdown(f'<div class="metric-card"><b>Columns</b><br>{len(df.columns)}</div>', unsafe_allow_html=True)
c3.markdown(f'<div class="metric-card"><b>File</b><br>{file.name}</div>', unsafe_allow_html=True)

# ── PREVIEW ─────────────────────────────────────────
with st.expander("Preview Data"):
    st.dataframe(df.head(100), use_container_width=True)

# ── ANALYSIS SECTION ─────────────────────────────────
st.markdown("## 📊 Analysis")

option = st.selectbox("Choose Action", [
    "Show First 5 Rows",
    "Show First 10 Rows",
    "Summary (5 Points)",
    "Mean / Median / Mode",
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

prompt = st.text_area("Custom Prompt")

# ── RUN ─────────────────────────────────────────────
if st.button("🚀 Run Analysis"):

    if option == "Show First 5 Rows":
        st.dataframe(df.head(), use_container_width=True)

    elif option == "Show First 10 Rows":
        st.dataframe(df.head(10), use_container_width=True)

    elif option == "Summary (5 Points)":
        st.markdown(f"""
        • Rows: **{len(df)}**  
        • Columns: **{len(df.columns)}**  
        • Missing Values: **{df.isnull().sum().sum()}**  
        • Numeric Columns: **{len(df.select_dtypes(include='number').columns)}**  
        • Categorical Columns: **{len(df.select_dtypes(exclude='number').columns)}**
        """)

    elif option == "Mean / Median / Mode":
        st.write("Mean:", df[col_x].mean())
        st.write("Median:", df[col_x].median())
        st.write("Mode:", df[col_x].mode().values)

    elif option == "Value Counts":
        st.dataframe(df[col_x].value_counts().reset_index(), use_container_width=True)

    elif option == "Bar Chart":
        st.bar_chart(df[col_x].value_counts().head(10))

    elif option == "Pie Chart":
        st.pyplot(df[col_x].value_counts().head(10).plot.pie(autopct='%1.1f%%').figure)

    elif option == "Histogram":
        st.bar_chart(df[col_x])

    elif option == "Scatter Plot":
        st.scatter_chart(df[[col_x, col_y]])

    elif option == "Correlation Heatmap":
        import matplotlib.pyplot as plt
        corr = df.select_dtypes(include='number').corr()
        fig, ax = plt.subplots()
        cax = ax.matshow(corr)
        fig.colorbar(cax)
        st.pyplot(fig)

    elif option == "Custom AI Prompt":

        if not prompt:
            st.error("Enter a prompt")
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
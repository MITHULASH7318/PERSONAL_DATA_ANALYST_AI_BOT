import streamlit as st
from analyst import load_data, ask_llm, run_code, prompt_to_code

st.set_page_config(page_title="PERSONAL CHATBOT BUDA", layout="wide")

# ── HEADER ─────────────────────────
st.markdown("""
# 💼 PERSONAL CHATBOT BUDA  
### Business Data Analyst Dashboard
""")

# ── FILE UPLOAD ───────────────────
file = st.file_uploader("Upload Dataset", type=["csv","xlsx","json"])

if not file:
    st.info("Upload a dataset to start")
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

# ── ANALYSIS OPTIONS ──────────────
st.subheader("📊 Analysis")

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

# ── RUN ───────────────────────────
if st.button("Run Analysis"):

    if option == "Show First 5 Rows":
        st.dataframe(df.head(), use_container_width=True)

    elif option == "Show First 10 Rows":
        st.dataframe(df.head(10), use_container_width=True)

    elif option == "Summary (5 Points)":
        st.write(f"""
        • Rows: {len(df)}  
        • Columns: {len(df.columns)}  
        • Missing Values: {df.isnull().sum().sum()}  
        • Numeric Columns: {len(df.select_dtypes(include='number').columns)}  
        • Categorical Columns: {len(df.select_dtypes(exclude='number').columns)}
        """)

    elif option == "Mean / Median / Mode":
        st.write("Mean:", df[col_x].mean())
        st.write("Median:", df[col_x].median())
        st.write("Mode:", df[col_x].mode().values)

    elif option == "Value Counts":
        st.dataframe(df[col_x].value_counts().reset_index())

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
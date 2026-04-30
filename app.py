import streamlit as st
from analyst import *

st.set_page_config(page_title="PERSONAL CHATBOT BUDA", layout="wide")

st.markdown("## 💼 PERSONAL CHATBOT BUDA")
st.caption("Business Data Analyst ChatBot")

# Upload
file = st.file_uploader("Upload your business dataset")

if not file:
    st.stop()

df = load_data(file)

# Stats
c1,c2,c3 = st.columns(3)
c1.metric("Rows", len(df))
c2.metric("Columns", len(df.columns))
c3.metric("File", file.name)

st.dataframe(df.head(20), use_container_width=True)

# Prompt UI
st.markdown("### 📊 Analysis Options")

suggestions = suggest_prompts(df)
selected = st.selectbox("Choose Analysis", suggestions)

custom = st.text_input("Or ask your own question")

prompt = custom if custom else selected

# Run
if st.button("Run Analysis"):

    code = prompt_to_code(prompt, df)

    if code:
        res = run_code(df, code)

    else:
        llm_out = ask_llm(prompt)

        if "```python" in llm_out:
            code = llm_out.split("```python")[1].split("```")[0]
            res = run_code(df, code)
        else:
            st.error(llm_out)
            st.stop()

    # Output
    if res["type"] == "dataframe":
        st.dataframe(res["df"], use_container_width=True)

    elif res["type"] == "image":
        st.image(res["path"])

    else:
        st.write(res["output"])
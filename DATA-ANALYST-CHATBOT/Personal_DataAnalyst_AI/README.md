# 🧠 Personal AI Data Analyst

An **interactive AI-powered data analyst dashboard** built with Python and Streamlit. This tool allows users to **upload CSV, Excel, or JSON datasets** and get automated, reliable data analysis including **summary statistics, visualizations, anomaly detection, and more**. It combines the **linguistic capabilities of AI** with **Python’s computational precision**, ensuring accurate results without hallucinations.

---

## **Project Overview**

Traditional AI chatbots often fail at precise computations. Our solution avoids this by **generating Python code for analysis** instead of asking the AI to calculate directly. This project provides:

- **Automated deterministic analysis** for common tasks.
- **Custom AI-powered analysis** through local LLMs (e.g., Llama 3 via Ollama).
- **Interactive, visual interface** built with Streamlit.
- **Robust data handling** for CSV, Excel, and JSON files.
- **Safe execution** of generated Python code in a sandbox environment.

---

## **Key Features**

- **File Upload**: Supports CSV, Excel (`.xls/.xlsx`), and JSON.
- **Automatic Column Detection**: Numeric, categorical, and datetime columns detected automatically.
- **Suggested Prompts**: Ready-to-use analysis suggestions like:
  - Summarize dataset
  - Top counts for categorical columns
  - Histograms, scatter plots, correlation heatmaps
  - Time series aggregations
  - Anomaly detection using z-score
- **Custom Prompts with AI**: Generate Python code for unique analysis tasks using a local LLM (optional).
- **Visualization & Download**: Charts displayed in the app, results downloadable as CSV.

---

## **Technologies & Packages Used**

| Package/Tool | Purpose |
|--------------|---------|
| `streamlit` | Frontend dashboard for uploading files, displaying results, and interaction. |
| `pandas` | Data manipulation and statistics. |
| `numpy` | Numerical computations. |
| `matplotlib` | Visualizations: histograms, scatter plots, heatmaps. |
| `scipy` | Statistical calculations (z-score for anomaly detection). |
| `openpyxl` | Reading Excel files. |
| `duckdb` (optional) | Fast query engine for large datasets. |
| `ollama` (optional) | Local LLM to generate Python code from custom natural language prompts. |

---

## **How It Works (Step-by-Step)**

1. **Upload Dataset**  
   Users upload CSV, Excel, or JSON files via the Streamlit interface. `load_data()` automatically reads the file into a pandas DataFrame.

2. **Column Type Detection**  
   The app detects numeric, categorical, and datetime columns using `_detect_column_types()`. This helps suggest meaningful analyses automatically.

3. **Suggested Prompts**  
   `suggest_prompts()` generates ready-made analysis prompts based on dataset structure.

4. **Prompt-to-Code Translation**  
   Deterministic prompts are converted into **Python code** using `prompt_to_code()`. This ensures accurate calculations and visualizations.

5. **Execution Engine**  
   `run_code()` executes Python code safely:
   - Returns a DataFrame or text output
   - Generates plots as temporary images
   - Handles errors gracefully

6. **Custom LLM Analysis (Optional)**  
   If a prompt is not recognized, the app can send it to a **local LLM (Llama 3)** via `ask_llm()`, which returns Python code for execution.

7. **Display Results**  
   Results are displayed as:
   - **Tables** (DataFrames)
   - **Charts** (saved images)
   - **Text summaries**  
   Users can **download results** as CSV.

---

## **Installation & Setup**

1. Clone the repository:

```bash
git clone https://github.com/your-username/DATA-ANALYST-CHATBOT.git
cd DATA-ANALYST-CHATBOT/Personal_DataAnalyst_AI



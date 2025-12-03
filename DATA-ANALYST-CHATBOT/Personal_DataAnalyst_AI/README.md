# Personal AI Data Analyst

## Setup

1. create & activate a virtual environment (recommended)
   python3 -m venv venv
   source venv/bin/activate

2. install dependencies
   pip install -r requirements.txt

3. (Optional) to support custom prompts via local LLM:
   - install Ollama and pull a model, e.g.:
     ollama pull llama3.1

## Run

streamlit run app.py

Upload a CSV / XLSX / JSON file.  
Pick a suggested analysis or write your own prompt.  
Click “Run analysis” to get a result (table, text, or chart).  

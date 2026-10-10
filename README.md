# Thesis Explorer

Scrape BINUS graduation data, extract thesis methods/domains with a local LLM (Ollama) or Gemini,
and explore everything in a Streamlit dashboard.

## Setup (once)
```
python -m venv venv
venv\Scripts\activate            # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt -r requirements-pipeline.txt
```
Install Ollama from https://ollama.com/download (the code starts it and downloads the model for you).
Optional model override: `set LOCAL_MODEL=qwen2.5:3b`.

For Gemini, create an API key at https://aistudio.google.com and set it:
`set GEMINI_API_KEY=your_key` (PowerShell: `$env:GEMINI_API_KEY="your_key"`, Linux/Mac: `export ...`).
Override the model with the `GEMINI_MODEL` env var if needed.

## Run order
```
python scraper.py 70 72 73        # 1. scrape -> data/*.csv (resumable)
python build_data.py              # 2. clean + embeddings + 2D map
python extract.py ollama          # 3. extract method/domain (or: gemini). Resumable
python normalize.py               # 4. clean up labels with keyword rules
streamlit run app.py              # 5. dashboard
```
Steps 3-4 are optional; without them only the "Method × Domain" tab stays empty.
Names are dropped in step 2; do not commit the raw `data/*.csv` files (they contain names).

## Deploy (Streamlit Community Cloud)
1. Commit `app.py`, `build_data.py`, `llm.py`, `requirements.txt`, `.gitignore` and the data files
   `data/clean.csv`, `data/emb.npy`, `data/extracted_norm.csv`. Raw `data/*Graduates.csv` (names) is git-ignored.
2. On https://share.streamlit.io create an app from the repo, main file `app.py`.
3. In Advanced settings -> Secrets paste `GEMINI_API_KEY = "..."` (see `.streamlit/secrets.toml.example`).
Online only the Gemini backend is offered (Ollama is not available on the host).

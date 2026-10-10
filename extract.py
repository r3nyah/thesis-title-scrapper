"""Extract method/domain/type from each thesis title. Resumable if interrupted.
Usage: python extract.py ollama   |   python extract.py gemini"""
import sys, time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from llm import ask, pull_local_model

PROMPT = """From the thesis title below, reply with JSON ONLY using these keys:
"method": the main technique, algorithm or technology, short and in English (e.g. CNN, LSTM, YOLO, BERT, Random Forest, Fuzzy, Laravel, Flutter). Use "Other" if none is stated.
"domain": the application field in 1-2 words (e.g. Health, Education, Finance, Agriculture, Transportation, Gaming).
"type": one of "application", "ml_research", "data_analysis".
Title: {title}"""
OUT = Path("data/extracted.csv")
COLS = ["title", "method", "domain", "type"]
OLD_COLS = {"metode": "method", "tipe": "type"}  # files created by the earlier Indonesian version still work
MAX_FAILS = 5  # stop after this many consecutive failures (usually a setup problem, not the title)


def save(rows):
    pd.DataFrame(rows, columns=COLS).to_csv(OUT, index=False)


def main(backend):
    titles = pd.read_csv("data/clean.csv")["title"].tolist()
    try:
        rows = pd.read_csv(OUT).rename(columns=OLD_COLS).to_dict("records") if OUT.exists() else []
    except pd.errors.EmptyDataError:
        rows = []
    seen = {r["title"] for r in rows}
    if backend == "ollama":
        pull_local_model()

    fails = 0
    for i, t in enumerate(tqdm([t for t in titles if t not in seen])):
        try:
            r = ask(PROMPT.format(title=t), backend, as_json=True)
            rows.append({"title": t, "method": str(r.get("method", "Other")).strip().title(),
                         "domain": str(r.get("domain", "Other")).strip().title(),
                         "type": str(r.get("type", "")).strip().lower()})
            fails = 0
        except Exception as e:
            fails += 1
            print(f"\nfailed ({fails}/{MAX_FAILS}): {t[:50]} -> {e}")
            if fails >= MAX_FAILS:
                save(rows)
                sys.exit(f"Stopped after {MAX_FAILS} consecutive failures. Fix the problem, then re-run "
                         f"(progress on {len(rows)} titles is saved).")
        if backend == "gemini":
            time.sleep(4)  # stay under the free-tier rate limit
        if i % 20 == 0:
            save(rows)
    save(rows)
    print(f"Done: {len(rows)} titles -> {OUT}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ollama")

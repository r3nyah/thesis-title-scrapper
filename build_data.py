"""Merge CSVs -> clean -> anonymize -> embeddings -> 2D coordinates.
Usage: python build_data.py"""
import glob, os
import numpy as np, pandas as pd

EMB_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # small (~120MB), understands Indonesian + English
# Indonesian stopwords are kept on purpose: some thesis titles are written in Indonesian.
STOP = set("""yang dan di ke dari untuk pada dengan dalam atau ini itu sebagai oleh serta berbasis menggunakan
penggunaan metode antara terhadap studi kasus analisis perancangan pembuatan implementasi sistem aplikasi
the of and for in on to a an with using based by from at is are its case study design analysis implementation system
application development""".split())


def load_raw():
    dfs = []
    for p in glob.glob("data/*-*Graduates.csv"):
        name = os.path.basename(p)
        d = pd.read_csv(p)
        d["edition"] = name.split("-")[0]
        d["kind"] = "Outstanding" if "Outstanding" in name else "Regular"
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def main():
    from sentence_transformers import SentenceTransformer
    from sklearn.manifold import TSNE

    df = load_raw().rename(columns={"Thesis Title": "title", "Program": "program", "IPK": "gpa", "GPA": "gpa"})
    df = df.dropna(subset=["title"])
    df["title"] = df["title"].str.replace(r"\s+", " ", regex=True).str.strip()
    df["gpa"] = pd.to_numeric(df["gpa"].astype(str).str.replace(",", "."), errors="coerce")
    # If a title appears in both lists, keep the Outstanding row ("Outstanding" < "Regular" alphabetically).
    df = df.sort_values("kind").drop_duplicates("title").reset_index(drop=True)
    df = df.drop(columns=["Name"], errors="ignore")  # anonymization: names are dropped

    print(f"{len(df)} titles. Building embeddings...")
    emb = SentenceTransformer(EMB_MODEL).encode(df["title"].tolist(), normalize_embeddings=True, show_progress_bar=True)
    xy = TSNE(2, perplexity=min(30, len(df) - 1), random_state=42, init="pca").fit_transform(emb)
    df["x"], df["y"] = xy[:, 0], xy[:, 1]

    df.to_csv("data/clean.csv", index=False)
    np.save("data/emb.npy", emb)
    print("Done -> data/clean.csv, data/emb.npy")


if __name__ == "__main__":
    main()

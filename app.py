"""Run: streamlit run app.py"""
import os, re, shutil
from collections import Counter
import numpy as np, pandas as pd, plotly.express as px, streamlit as st
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from build_data import STOP, EMB_MODEL
from llm import ask

st.set_page_config(page_title="Thesis Explorer", layout="wide")

MAX_QUESTIONS = 10  # new AI-chat questions allowed per browser session (protects your API quota online)


def _secret(name):
    """Read a value from Streamlit secrets (online) or environment variables (local)."""
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name)


if _secret("GEMINI_API_KEY"):  # llm.py reads the key from the environment
    os.environ["GEMINI_API_KEY"] = _secret("GEMINI_API_KEY")


@st.cache_data(show_spinner=False)
def cached_ask(prompt, backend):  # identical question + context = no new API call on Streamlit reruns
    return ask(prompt, backend)



@st.cache_data
def load():
    # old column names (ipk, metode, tipe) are renamed so files from the earlier version still work
    df = pd.read_csv("data/clean.csv").rename(columns={"ipk": "gpa"})
    emb = np.load("data/emb.npy")
    ex = next((p for p in ["data/extracted_norm.csv", "data/extracted.csv"] if os.path.exists(p)), None)
    if ex:
        ext = pd.read_csv(ex).rename(columns={"metode": "method", "tipe": "type"}).drop_duplicates("title")
        df = df.merge(ext, on="title", how="left")
    return df, emb


@st.cache_resource
def encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMB_MODEL)


@st.cache_data
def cluster(k, _emb, titles):
    labels = KMeans(k, n_init=10, random_state=42).fit_predict(_emb)
    docs = [" ".join(t for t, l in zip(titles, labels) if l == c) for c in range(k)]
    tf = TfidfVectorizer(stop_words=list(STOP), ngram_range=(1, 2), max_df=0.8)
    m = tf.fit_transform(docs)
    terms = tf.get_feature_names_out()
    names = {c: ", ".join(terms[m[c].toarray()[0].argsort()[::-1][:3]]) for c in range(k)}
    return labels, names


df, emb = load()
st.title("🎓 Thesis Explorer - BINUS Computer Science (S1)")

# --- sidebar ---
k = st.sidebar.slider("Number of topics (clusters)", 2, 15, 6)
eds = sorted(df["edition"].astype(str).unique())
ed = st.sidebar.multiselect("Graduation edition", eds, default=eds)
kind = st.sidebar.multiselect("Graduate type", sorted(df["kind"].unique()), default=sorted(df["kind"].unique()))
prog = st.sidebar.multiselect("Program", sorted(df["program"].dropna().unique()))
backend = st.sidebar.radio("AI backend (chat)", (["ollama"] if shutil.which("ollama") else []) + ["gemini"])

labels, names = cluster(k, emb, tuple(df["title"]))
df["topic"] = [f"{c}: {names[c]}" for c in labels]
df["edition"] = df["edition"].astype(str)
mask = df["edition"].isin(ed) & df["kind"].isin(kind)
if prog:
    mask &= df["program"].isin(prog)
f = df[mask]
st.sidebar.metric("Filtered titles", len(f))

t1, t2, t3, t4, t5, t6, t7 = st.tabs(
    ["Overview", "Topic Map", "Trends", "Method × Domain", "GPA", "Similar Titles", "Ask the Data"])

with t1:
    c1, c2 = st.columns(2)
    words = Counter(w for t in f["title"] for w in re.findall(r"[a-zA-Z]{3,}", t.lower()) if w not in STOP)
    top = pd.DataFrame(words.most_common(25), columns=["word", "count"])
    c1.plotly_chart(px.bar(top[::-1], x="count", y="word", orientation="h", title="Top words"), width="stretch")
    c2.plotly_chart(px.histogram(f, x="edition", color="kind", title="Graduates per edition"), width="stretch")

with t2:
    st.plotly_chart(px.scatter(f, x="x", y="y", color="topic", hover_name="title",
                               hover_data={"x": False, "y": False, "gpa": True, "edition": True},
                               height=650, title="Thesis title map (closer = more similar meaning)"),
                    width="stretch")
    pick = st.selectbox("Browse a topic", sorted(f["topic"].unique()))
    st.dataframe(f[f["topic"] == pick][["title", "edition", "kind", "gpa"]], width="stretch")

with t3:
    tr = pd.crosstab(f["edition"], f["topic"], normalize="index").reset_index().melt("edition", var_name="topic", value_name="share")
    st.plotly_chart(px.bar(tr, x="edition", y="share", color="topic", title="Topic mix per edition"), width="stretch")

with t4:
    if "method" in f and f["method"].notna().any():
        s = f.dropna(subset=["method", "domain"])
        s = s[(s["method"] != "Other") & (s["domain"] != "Other")]
        s = s[s["method"].isin(s["method"].value_counts().head(12).index) & s["domain"].isin(s["domain"].value_counts().head(12).index)]
        st.plotly_chart(px.imshow(pd.crosstab(s["method"], s["domain"]), text_auto=True, aspect="auto",
                                  title="Empty / low cells = research gaps"), width="stretch")
    else:
        st.info("Run `python extract.py ollama` (or gemini), then `python normalize.py`, first.")

with t5:
    g = f.dropna(subset=["gpa"])
    st.plotly_chart(px.box(g, x="topic", y="gpa", color="kind", title="GPA by topic"), width="stretch")
    st.caption("Correlation is not causation; some topics have very few data points.")

with t6:
    q = st.text_input("Describe your thesis idea")
    if q:
        qv = encoder().encode([q], normalize_embeddings=True)[0]
        sims = emb[f.index] @ qv
        idx = sims.argsort()[::-1][:10]
        st.dataframe(pd.DataFrame({"Title": f["title"].values[idx], "Score": sims[idx].round(3),
                                   "Edition": f["edition"].values[idx]}), width="stretch")

with t7:
    pw = _secret("APP_PASSWORD")
    if pw and st.text_input("Password for the AI chat", type="password") != pw:
        st.info("Enter the password to use the AI chat.")
    else:
        q = st.text_input("Ask about this data (e.g. which topic dominates and what are the gaps?)")
        asked = st.session_state.setdefault("asked", set())
        if q and (q, backend) not in asked and len(asked) >= MAX_QUESTIONS:
            st.warning(f"Limit of {MAX_QUESTIONS} new questions per session reached. Reload the page to reset.")
        elif q:
            ctx = f"Total {len(f)} titles.\nPer edition: {f['edition'].value_counts().to_dict()}\nTopics: {f['topic'].value_counts().to_dict()}\n"
            ctx += f"Average GPA per topic: {f.groupby('topic')['gpa'].mean().round(2).to_dict()}\n"
            if "method" in f:
                ctx += f"Top methods: {f['method'].value_counts().head(10).to_dict()}\nTop domains: {f['domain'].value_counts().head(10).to_dict()}\n"
            with st.spinner("Thinking..."):
                try:
                    # retrieval: add the titles most similar to the question so the model can "search" the data
                    qv = encoder().encode([q], normalize_embeddings=True)[0]
                    near = f.assign(score=emb[f.index] @ qv).nlargest(15, "score")
                    cols = [c for c in ["edition", "method", "domain"] if c in f]
                    ctx += "\nThesis titles most relevant to the question (title | " + " | ".join(cols) + "):\n"
                    ctx += "\n".join(f"- {r['title']} | " + " | ".join(str(r[c]) for c in cols) for _, r in near.iterrows())
                    prompt = ("You are a data analyst for BINUS thesis titles. Answer concisely in English, using only the data below. "
                              "If the question asks to search for or compare with a topic, base the answer on the relevant titles list "
                              "and mention a few of those titles.\n" + ctx + f"\n\nQuestion: {q}")
                    st.write(cached_ask(prompt, backend))
                    asked.add((q, backend))
                    with st.expander("Titles sent to the AI as context"):
                        st.dataframe(near[["title", "edition", "score"]], width="stretch")
                except Exception as e:
                    st.error(f"Failed to call {backend}: {e}")

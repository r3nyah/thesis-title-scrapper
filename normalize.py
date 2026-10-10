"""Clean up the LLM extraction: keyword rules first, the LLM label only as a fallback.
Usage: python normalize.py  -> data/extracted_norm.csv (data/extracted.csv is left untouched)"""
import re
import pandas as pd

# (label, regex). Specific methods win; generic ones are used only when no specific method matches.
SPECIFIC = [
    ("CNN", r"\bcnn\b|convolutional|efficientnet|resnet|vgg|mobilenet"), ("LSTM", r"\blstm\b|long short"),
    ("RNN/GRU", r"\brnn\b|\bgru\b|recurrent"), ("YOLO", r"yolo"), ("BERT", r"bert\b|indobert"),
    ("Transformer", r"transformer"),
    ("LLM/RAG", r"\bllm|large language model|\bgpt|chatgpt|gemini|\brag\b|retrieval.augmented|generative ai"),
    ("Random Forest", r"random forest"), ("XGBoost", r"xgboost|gradient boost|lightgbm"),
    ("SVM", r"\bsvm\b|support vector"), ("Naive Bayes", r"na[iï]ve bayes"),
    ("KNN", r"\bknn\b|nearest neighbo"), ("Decision Tree", r"decision tree|\bc4\.5\b|\bid3\b"),
    ("Neural Network", r"neural network|\bann\b|\bmlp\b"), ("K-Means", r"k-?means|clustering"),
    ("Fuzzy", r"fuzzy"), ("Genetic Algorithm", r"genetic algorithm"),
    ("Ensemble Learning", r"ensemble|stacking"), ("Blockchain", r"blockchain|smart contract|ethereum"),
    ("Augmented Reality", r"augmented reality|\bar\b"), ("Virtual Reality", r"virtual reality|\bvr\b"),
    ("IoT", r"\biot\b|internet of things"), ("Laravel", r"laravel"), ("Flutter", r"flutter"),
    ("React", r"react"), ("Unity", r"\bunity\b"), ("Odoo", r"odoo"), ("Data Mining", r"data mining"),
    ("Recommender System", r"recommend"), ("Sentiment Analysis", r"sentiment"),
]
GENERIC = [
    ("Deep Learning", r"deep learning"), ("Machine Learning", r"machine learning|\bml\b"),
    ("Mobile Development", r"android|\bios\b|mobile"), ("Web Development", r"\bweb|website"),
    ("Game Development", r"game"), ("Gamification", r"gamif"),
    ("Cyber/Network Security", r"cyber|threat|attack|flood|wazuh|intrusion|firewall|honeypot|vulnerab|secur|\bids\b|\bsiem\b|network"),
    ("UI/UX Design", r"ui/ux|\bui\b|user experience|design system|human-cent"),
    ("Information System", r"point of sale|\bpos\b|inventory|\berp\b|information system|management system|booking|reservation"),
    ("Application Development", r"applicat|platform|system|digital"),  # last-resort bucket
]
DOMAINS = [  # order = priority; a few Indonesian keywords are included because some titles are in Indonesian
    ("Health", r"health|medical|medic|disease|cancer|diabet|pneumonia|patient|hospital|stunting|mental|depress|covid|x-ray|ecg|cardio|clinic|posyandu|diet|nutrition"),
    ("Security", r"secur|malware|phishing|attack|intrusion|vulnerab|honeypot|ransomware|spoof|leak|injection|cyber"),
    ("Finance", r"stock|financ|bank|loan|credit|payment|tax|accounting|ledger|budget|crypto|treasury|invest"),
    ("Agriculture", r"agricultur|plant|farm|crop|aquaponic|spinach|seed|irrigation"),
    ("Education", r"educat|e-learning|\blearning (app|platform|media|application|game)|school|student|teach|course|tutor|literacy|curriculum"),
    ("Gaming", r"game|gaming|esport"),
    ("Social Media", r"social media|twitter|tweet|reddit|instagram|tiktok|youtube|\bx tweets"),
    ("Transportation", r"transport|traffic|vehicle|\bcar\b|flight|logistic|ride"),
    ("E-Commerce", r"e-?commerce|marketplace|shop|retail|product|\bpos\b|point of sale|inventory|restaurant|hotel|canteen"),
]
BAD_DOMAIN = {"", "other", "aplikasi", "application", "technology", "nan"}
OLD_COLS = {"metode": "method", "tipe": "type"}  # files from the earlier Indonesian version still work


def first(rules, text):
    """Label of the rule that matches earliest in the text, or None."""
    best = None
    for label, pat in rules:
        m = re.search(pat, text)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), label)
    return best[1] if best else None


def method(title, llm_label):
    t = title.lower()
    r = first(SPECIFIC, t) or first(GENERIC, t)
    if r:
        return r
    lab = str(llm_label).strip()
    if first(SPECIFIC + GENERIC, lab.lower()):  # known method named by the LLM but absent from the title -> likely hallucinated
        return "Other"
    words = [w for w in re.findall(r"[a-z0-9]+", lab.lower()) if len(w) >= 4]
    return lab.title() if words and any(w in t for w in words) else "Other"


def domain(title, llm_label):
    t = title.lower()
    for label, pat in DOMAINS:
        if re.search(pat, t):
            return label
    lab = str(llm_label).strip()
    canon = next((l for l, p in DOMAINS if re.search(p, lab.lower())), None)
    if canon:
        return canon
    return "Other" if lab.lower() in BAD_DOMAIN else lab.title()


def main():
    df = pd.read_csv("data/extracted.csv").rename(columns=OLD_COLS)
    df["method"] = [method(t, m) for t, m in zip(df["title"], df["method"])]
    df["domain"] = [domain(t, d) for t, d in zip(df["title"], df["domain"])]
    df.to_csv("data/extracted_norm.csv", index=False)
    print(f"Other method: {(df.method == 'Other').mean():.0%} | Other domain: {(df.domain == 'Other').mean():.0%}")
    print(df.method.value_counts().head(20), "\n", df.domain.value_counts().head(15))
    print("Saved -> data/extracted_norm.csv")


if __name__ == "__main__":
    main()

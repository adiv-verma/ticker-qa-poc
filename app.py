import os, re, json, requests, pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Ticker Q&A (POC)", page_icon="📄", layout="centered")
st.title("📄 Ticker Q&A (POC) — Free SEC Data")

# ---- SEC helpers ----
SEC_BASE = "https://data.sec.gov"
HEADERS = {"User-Agent": "nirvaanventuresllc@gmail.com"}  # replace with your email

def get_cik_and_filings(symbol: str):
    mapping = requests.get(f"{SEC_BASE}/files/company_tickers.json", headers=HEADERS, timeout=30).json()
    sym = symbol.upper()
    for row in mapping:
        if row["ticker"].upper() == sym:
            cik = str(row["cik_str"]).zfill(10)
            sub = requests.get(f"{SEC_BASE}/submissions/CIK{cik}.json", headers=HEADERS, timeout=30).json()
            filings = sub.get("filings", {}).get("recent", {})
            df = pd.DataFrame(filings)
            return cik, df
    raise ValueError("Ticker not found in SEC mapping.")

def latest_filing_url(cik: str, df_recent: pd.DataFrame, form="10-K"):
    rows = df_recent[df_recent["form"] == form].head(1)
    if rows.empty: return None
    acc_no = rows["accessionNumber"].iloc[0].replace("-", "")
    primary_doc = rows["primaryDocument"].iloc[0]
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no}/{primary_doc}"

def fetch_text(url: str):
    r = requests.get(url, headers=HEADERS, timeout=60)
    soup = BeautifulSoup(r.text, "lxml")
    for tag in soup(["script","style","noscript"]): tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text

def chunk_text(text: str, size=900, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunk = text[start:end]
        last_break = chunk.rfind("\n\n")
        if last_break > 200:
            chunk = chunk[:last_break]
            end = start + last_break
        chunks.append(chunk.strip())
        start = max(end - overlap, end)
    return [c for c in chunks if len(c) > 200]

def build_index(chunks):
    vec = TfidfVectorizer(stop_words="english", max_df=0.85)
    X = vec.fit_transform(chunks)
    return vec, X

def retrieve(question, vec, X, chunks, topk=5):
    qv = vec.transform([question])
    sims = cosine_similarity(qv, X).ravel()
    idx = sims.argsort()[::-1][:topk]
    return [(chunks[i], float(sims[i])) for i in idx]

# ---- UI ----
ticker = st.text_input("Ticker", "AAPL").upper().strip()
question = st.text_input("Your question", "What risks are mentioned?")
if st.button("Search"):
    try:
        cik, recent = get_cik_and_filings(ticker)
        url = latest_filing_url(cik, recent, form="10-K") or latest_filing_url(cik, recent, form="10-Q")
        if not url:
            st.warning("No recent 10-K/10-Q found.")
            st.stop()
        text = fetch_text(url)
        chunks = chunk_text(text)
        vec, X = build_index(chunks)
        hits = retrieve(question, vec, X, chunks, topk=5)
        st.subheader("Top excerpts from SEC filing")
        for chunk, score in hits:
            st.write(f"**Relevance:** {score:.3f}")
            st.markdown(f"> {chunk}")
            st.caption(f"Source: {url}")
            st.divider()
    except Exception as e:
        st.error(str(e))

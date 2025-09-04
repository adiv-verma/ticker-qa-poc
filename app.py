import os, re, time, json, requests, pandas as pd
import streamlit as st
from typing import List, Tuple, Dict
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Ticker Q&A (POC)", page_icon="📄", layout="centered")
st.title("📄 Ticker Q&A (POC) — Free SEC Data")

# ──────────────────────────────────────────────────────────────────────────────
# SEC endpoints + REQUIRED header
# SEC *requires* a User-Agent with contact info. Using your provided email.
# ──────────────────────────────────────────────────────────────────────────────
SEC_BASE = "https://data.sec.gov"
HEADERS = {"User-Agent": "nirvaanventuresllc@gmail.com"}

# ──────────────────────────────────────────────────────────────────────────────
# Small HTTP helper with retries (handles transient rate limits)
# ──────────────────────────────────────────────────────────────────────────────
def http_get_json(url: str, headers: dict, timeout: int = 30, retries: int = 3):
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            # Backoff on 429/5xx
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.5 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(1.2 * (i + 1))
    raise RuntimeError(f"Failed to GET JSON from {url}: {last_err}")

def http_get_text(url: str, headers: dict, timeout: int = 60, retries: int = 3):
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.5 * (i + 1))
                continue
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_err = e
            time.sleep(1.2 * (i + 1))
    raise RuntimeError(f"Failed to GET text from {url}: {last_err}")

# ──────────────────────────────────────────────────────────────────────────────
# Core SEC helpers
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)  # cache 6h
def load_ticker_mapping() -> List[Dict]:
    url = f"{SEC_BASE}/files/company_tickers.json"
    mapping = http_get_json(url, HEADERS, timeout=30)
    # mapping is a list[{"cik_str": 0000320193, "ticker": "AAPL", "title": "Apple Inc."}, ...]
    return mapping

@st.cache_data(show_spinner=False, ttl=60 * 30)  # cache 30m
def get_cik_and_recent(symbol: str) -> Tuple[str, pd.DataFrame]:
    mapping = load_ticker_mapping()
    sym = symbol.upper()
    for row in mapping:
        if row.get("ticker", "").upper() == sym:
            cik = str(row["cik_str"]).zfill(10)
            sub_url = f"{SEC_BASE}/submissions/CIK{cik}.json"
            submissions = http_get_json(sub_url, HEADERS, timeout=30)
            filings = submissions.get("filings", {}).get("recent", {})
            df = pd.DataFrame(filings) if filings else pd.DataFrame()
            return cik, df
    raise ValueError(f"Ticker {sym} not found in SEC mapping.")

def latest_filing_urls(cik: str, df_recent: pd.DataFrame, forms=("10-K", "10-Q")) -> List[Tuple[str, str]]:
    """Return [(FORM, url_to_primary_doc_html)] for the latest available of each requested form."""
    out = []
    if df_recent is None or df_recent.empty:
        return out
    for form in forms:
        rows = df_recent[df_recent["form"] == form]
        if not rows.empty:
            # Take the most recent row for that form
            row = rows.iloc[0]
            acc_no = row["accessionNumber"].replace("-", "")
            primary_doc = row["primaryDocument"]
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no}/{primary_doc}"
            out.append((form, filing_url))
    return out

def clean_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # normalize whitespace
    text = re.sub(r"\u00A0", " ", text)  # non-breaking spaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text).strip()
    return text

def chunk_text_with_source(text: str, source_url: str, size=900, overlap=150) -> List[Dict]:
    """Split text into overlapping chunks and keep per-chunk source URL."""
    chunks = []
    start = 0
    N = len(text)
    while start < N:
        end = min(N, start + size)
        chunk = text[start:end]
        # try to end on paragraph boundary
        lb = chunk.rfind("\n\n")
        if lb > 200:
            chunk = chunk[:lb]
            end = start + lb
        chunk = chunk.strip()
        if len(chunk) > 200:
            chunks.append({"text": chunk, "source": source_url})
        start = max(end - overlap, end)
    return chunks

# ──────────────────────────────────────────────────────────────────────────────
# Retrieval functions
# ──────────────────────────────────────────────────────────────────────────────
def build_index(chunks: List[Dict]):
    texts = [c["text"] for c in chunks]
    vec = TfidfVectorizer(stop_words="english", max_df=0.85)
    X = vec.fit_transform(texts)
    return vec, X

def retrieve(question: str, vec, X, chunks: List[Dict], topk=5):
    qv = vec.transform([question])
    sims = cosine_similarity(qv, X).ravel()
    idx = sims.argsort()[::-1][:topk]
    results = []
    for i in idx:
        results.append({"text": chunks[i]["text"], "score": float(sims[i]), "source": chunks[i]["source"]})
    return results

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**Data source:** Official SEC EDGAR (10-K / 10-Q).")
    st.caption("Tip: Ask for risks, liquidity, competition, supply chain, climate, lawsuits, etc.")

ticker = st.text_input("Ticker", "AAPL").upper().strip()
question = st.text_input("Your question", "What risks are mentioned?")
run = st.button("Search")

if run:
    try:
        if not ticker:
            st.warning("Please enter a ticker (e.g., AAPL, MSFT).")
            st.stop()

        # 1) Find CIK and recent filings
        cik, recent = get_cik_and_recent(ticker)

        if recent is None or recent.empty:
            st.warning("SEC returned no recent filings for this ticker.")
            st.stop()

        # 2) Get latest 10-K (or fallback to 10-Q)
        urls = latest_filing_urls(cik, recent, forms=("10-K", "10-Q"))
        if not urls:
            st.warning("Couldn't find a recent 10-K or 10-Q for this ticker.")
            st.stop()

        # 3) Fetch documents (start with 10-K, then 10-Q) and build a corpus
        all_chunks: List[Dict] = []
        for form, url in urls:
            with st.spinner(f"Fetching {form}…"):
                html = http_get_text(url, HEADERS, timeout=90, retries=3)
            text = clean_html_to_text(html)
            chunks = chunk_text_with_source(text, url, size=900, overlap=150)
            all_chunks.extend(chunks)

        if not all_chunks:
            st.warning("No readable text extracted from the filing.")
            st.stop()

        # 4) Build index & retrieve
        vec, X = build_index(all_chunks)
        hits = retrieve(question, vec, X, all_chunks, topk=5)

        # 5) Show results
        st.subheader("Top excerpts from SEC filing")
        for hit in hits:
            st.write(f"**Relevance:** {hit['score']:.3f}")
            st.markdown(f"> {hit['text']}")
            st.caption(f"Source: {hit['source']}")
            st.divider()

    except Exception as e:
        st.error(f"Error: {e}")
        st.caption(
            "If this persists, wait a minute (rate limits) and try again. "
            "Make sure the User-Agent header contains a valid email."
        )

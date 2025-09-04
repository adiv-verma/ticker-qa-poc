import os, re, time, json, requests, pandas as pd
import streamlit as st
from typing import List, Tuple, Dict
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Ticker Q&A — LLM + SEC Receipts", page_icon="📄", layout="centered")
st.title("📄 Ticker Q&A — LLM Summary + SEC Receipts (Free Data)")

with st.sidebar:
    st.markdown("**Data:** Official SEC EDGAR (10-K / 10-Q)")
    st.caption("Ask about risks, liquidity, competition, supply chain, climate, lawsuits, outlook, etc.")
    st.markdown("---")
    st.caption("No paid market data used. LLM summarizes only the retrieved filing excerpts.")

# ──────────────────────────────────────────────────────────────────────────────
# Secrets / environment
# ──────────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY  # so the OpenAI client can pick it up

# ──────────────────────────────────────────────────────────────────────────────
# SEC endpoints + REQUIRED header
# ──────────────────────────────────────────────────────────────────────────────
EDGAR_API_BASE = "https://data.sec.gov"                    # submissions live here
HEADERS = {"User-Agent": "nirvaanventuresllc@gmail.com"}   # SEC requests a contact email in UA

# ──────────────────────────────────────────────────────────────────────────────
# HTTP helpers with retries
# ──────────────────────────────────────────────────────────────────────────────
def http_get_json(url: str, headers: dict, timeout: int = 30, retries: int = 3):
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
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
# SEC helpers
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def load_ticker_mapping() -> List[Dict]:
    """
    Robust loader: SEC sometimes returns list[...] or dict{"0":{...}} or {"data":[...]}.
    We normalize to list[dict].
    """
    url = "https://www.sec.gov/files/company_tickers.json"  # hosted on www.sec.gov
    raw = http_get_json(url, HEADERS, timeout=30)

    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        if "data" in raw and isinstance(raw["data"], list):
            items = raw["data"]
        else:
            items = list(raw.values())
    else:
        items = []

    out = []
    for obj in items:
        if isinstance(obj, dict) and "ticker" in obj and "cik_str" in obj:
            out.append(obj)
    if not out:
        raise RuntimeError("SEC ticker mapping is empty or malformed. Try again later.")
    return out

@st.cache_data(show_spinner=False, ttl=60 * 30)
def get_cik_and_recent(symbol: str) -> Tuple[str, pd.DataFrame]:
    mapping = load_ticker_mapping()
    sym = symbol.upper()
    match = next((row for row in mapping if str(row.get("ticker", "")).upper() == sym), None)
    if not match:
        raise ValueError(f"Ticker {sym} not found in SEC mapping.")
    cik = str(match["cik_str"]).zfill(10)

    sub_url = f"{EDGAR_API_BASE}/submissions/CIK{cik}.json"
    submissions = http_get_json(sub_url, HEADERS, timeout=30)
    recent = submissions.get("filings", {}).get("recent", {})
    df = pd.DataFrame(recent) if recent else pd.DataFrame()

    if not df.empty:
        for col in ("accessionNumber", "primaryDocument", "form", "filingDate"):
            if col in df.columns:
                df[col] = df[col].astype(str)
        if "filingDate" in df.columns:
            try:
                df = df.sort_values("filingDate", ascending=False)
            except Exception:
                pass

    return cik, df

def latest_filing_urls(cik: str, df_recent: pd.DataFrame, forms=("10-K", "10-Q")) -> List[Tuple[str, str]]:
    out = []
    if df_recent is None or df_recent.empty:
        return out
    for form in forms:
        rows = df_recent[df_recent["form"] == form]
        if not rows.empty:
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
    text = re.sub(r"\u00A0", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()

def chunk_text_with_source(text: str, source_url: str, size=900, overlap=150) -> List[Dict]:
    chunks = []
    start = 0
    N = len(text)
    while start < N:
        end = min(N, start + size)
        chunk = text[start:end]
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
# Retrieval (TF-IDF)
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
    return [{"text": chunks[i]["text"], "score": float(sims[i]), "source": chunks[i]["source"], "i": int(i)} for i in idx]

# ──────────────────────────────────────────────────────────────────────────────
# LLM summary (only uses retrieved passages)
# ──────────────────────────────────────────────────────────────────────────────
def summarize_with_llm(question: str, passages: List[Dict]) -> Dict:
    if not OPENAI_API_KEY:
        return {"answer": "(AI disabled: add OPENAI_API_KEY in Secrets)", "bullets": [], "citations": []}

    try:
        from openai import OpenAI
        client = OpenAI()  # reads OPENAI_API_KEY from environment
    except Exception as e:
        return {"answer": f"(OpenAI init error: {e})", "bullets": [], "citations": []}

    snippets = []
    for p in passages:
        snippets.append({
            "id": p["i"],
            "source": p["source"],
            "excerpt": p["text"][:2000]  # keep prompt compact
        })

    system = (
        "You are an equity research assistant. Answer ONLY using the provided excerpts. "
        "Do not invent facts. Keep the answer concise and neutral. "
        "Return JSON with keys: answer (string), bullets (array of strings, max 5), "
        "citations (array of snippet ids used)."
    )
    user_payload = {"question": question, "snippets": snippets}

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
        )
        data = resp.choices[0].message.content or "{}"
        out = json.loads(data)
        if "answer" not in out:
            out["answer"] = ""
        if "bullets" not in out or not isinstance(out["bullets"], list):
            out["bullets"] = []
        if "citations" not in out or not isinstance(out["citations"], list):
            out["citations"] = []
        return out
    except Exception as e:
        return {"answer": f"(LLM error: {e})", "bullets": [], "citations": []}

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
ticker = st.text_input("Ticker", "AAPL").upper().strip()
question = st.text_input("Your question", "What risks are mentioned?")
run = st.button("Search")

if run:
    try:
        if not ticker:
            st.warning("Please enter a ticker (e.g., AAPL, MSFT).")
            st.stop()

        # 1) Map ticker → CIK & recent filings
        cik, recent = get_cik_and_recent(ticker)
        if recent is None or recent.empty:
            st.warning("SEC returned no recent filings for this ticker.")
            st.stop()

        # 2) Latest 10-K (fallback to 10-Q)
        urls = latest_filing_urls(cik, recent, forms=("10-K", "10-Q"))
        if not urls:
            st.warning("Couldn't find a recent 10-K or 10-Q for this ticker.")
            st.stop()

        # 3) Fetch documents & build corpus
        all_chunks: List[Dict] = []
        for form, url in urls:
            with st.spinner(f"Fetching {form}…"):
                html = http_get_text(url, HEADERS, timeout=90, retries=3)
            text = clean_html_to_text(html)
            all_chunks.extend(chunk_text_with_source(text, url, size=900, overlap=150))

        if not all_chunks:
            st.warning("No readable text extracted from the filing.")
            st.stop()

        # 4) Index & retrieve
        vec, X = build_index(all_chunks)
        hits = retrieve(question, vec, X, all_chunks, topk=5)

        # 5) LLM summary on top passages
        with st.spinner("Generating AI summary…"):
            summary = summarize_with_llm(question, hits)

        # 6) Render — LLM summary first
        st.subheader("AI Summary")
        st.write(summary.get("answer", "").strip() or "(no summary)")
        if summary.get("bullets"):
            st.markdown("**Key points:**")
            for b in summary["bullets"][:5]:
                st.markdown(f"- {b}")

        # Citations mapped back to sources (dedup)
        if summary.get("citations"):
            st.caption("Citations:")
            cited_ids = []
            for x in summary["citations"]:
                try:
                    cited_ids.append(int(x))
                except Exception:
                    pass
            seen = set()
            for p in hits:
                if p["i"] in cited_ids and p["source"] not in seen:
                    st.markdown(f"- [{p['source']}]({p['source']})")
                    seen.add(p["source"])

        st.divider()

        # 7) Receipts — exact filing excerpts
        st.subheader("Supporting excerpts from SEC filing")
        for h in hits:
            st.write(f"**Relevance:** {h['score']:.3f}  •  Snippet ID: {h['i']}")
            st.markdown(f"> {h['text']}")
            st.caption(f"Source: {h['source']}")
            st.divider()

    except Exception as e:
        st.error(f"Error: {e}")
        st.caption("If this persists, wait a minute (rate limits) and try again. "
                   "Ensure the User-Agent header contains a valid email.")

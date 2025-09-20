# app.py
import streamlit as st
import pdfplumber
import pandas as pd
import spacy
from nltk.corpus import stopwords

# Load NLP model
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

st.set_page_config(page_title="Resume Relevance Checker", page_icon="🚀", layout="wide")
st.title("🚀 Resume Relevance Checker")

# --- Helper functions ---
def read_pdf(file):
    """Extracts text from PDF"""
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return " ".join(text.split())

def extract_keywords(text, top_n=20):
    """Extract keywords ignoring stopwords, focusing on nouns/proper nouns"""
    doc = nlp(text)
    keywords = [token.text for token in doc if token.pos_ in ('NOUN','PROPN') and token.text.lower() not in stop_words]
    return list(dict.fromkeys(keywords))[:top_n]

def score_resume(keywords, resume_text):
    """Compute score & verdict"""
    matched = [k for k in keywords if k.lower() in resume_text.lower()]
    missing = [k for k in keywords if k.lower() not in resume_text.lower()]
    score = round(len(matched)/max(1,len(keywords))*100,2)

    if score >= 75: verdict, color = "High", "green"
    elif score >= 50: verdict, color = "Medium", "orange"
    else: verdict, color = "Low", "red"

    return score, verdict, color, matched, missing

# --- Upload JD and Resumes ---
with st.expander("Upload Job Description (JD)"):
    jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"], key="jd")

with st.expander("Upload Resumes"):
    resumes = st.file_uploader("Upload Resume PDFs", type=["pdf"], accept_multiple_files=True, key="resumes")

if st.button("Evaluate ▶️"):
    if not jd_file:
        st.error("Please upload JD first!")
        st.stop()
    if not resumes:
        st.error("Please upload at least one resume!")
        st.stop()

    # --- Extract JD text & keywords ---
    jd_text = read_pdf(jd_file)
    keywords = extract_keywords(jd_text, top_n=20)
    st.subheader("🔑 JD Keywords for Evaluation")
    st.write(", ".join(keywords))

    # --- JD Preview ---
    with st.expander("Preview Job Description"):
        st.write(jd_text[:1000] + "..." if len(jd_text) > 1000 else jd_text)

    results = []
    for r in resumes:
        resume_text = read_pdf(r)
        score, verdict, color, matched, missing = score_resume(keywords, resume_text)

        results.append({
            "Resume": r.name,
            "Score (%)": score,
            "Verdict": verdict,
            "Matched Skills": ", ".join(matched),
            "Missing Skills": ", ".join(missing)
        })

        # Show individual resume analysis
        st.markdown(f"### 📄 {r.name}")
        st.markdown(f"**Score:** {score}%  |  **Verdict:** <span style='color:{color}'>{verdict}</span>", unsafe_allow_html=True)
        st.markdown(f"**Matched Skills:** {', '.join(matched) if matched else 'None'}")
        st.markdown(f"**Missing Skills:** {', '.join(missing) if missing else 'None'}")

    # --- Summary Table ---
    st.subheader("📊 Summary Table")
    df = pd.DataFrame(results)
    st.dataframe(df)

    # Download button
    st.download_button("💾 Download CSV", df.to_csv(index=False).encode(), "resume_scores.csv", "text/csv")


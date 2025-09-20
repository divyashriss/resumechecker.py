import streamlit as st
import pdfplumber
import pandas as pd
import spacy
from nltk.corpus import stopwords
import re

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

st.set_page_config(page_title="Resume Relevance Checker", page_icon="🚀", layout="wide")
st.title("🚀 Resume Relevance Checker")

# --- Helper functions ---
def read_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return " ".join(text.split())

def extract_keywords_phrases(text, top_n=30):
    doc = nlp(text)
    phrases = set()
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip()
        # remove stopwords and short words
        if phrase.lower() not in stop_words and len(phrase.split()) <= 5:
            phrases.add(phrase)
    return list(phrases)[:top_n]

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def score_resume(keywords, resume_text):
    resume_text_clean = clean_text(resume_text)
    matched = [k for k in keywords if clean_text(k) in resume_text_clean]
    missing = [k for k in keywords if clean_text(k) not in resume_text_clean]
    score = round(len(matched)/max(1,len(keywords))*100,2)

    if score >= 75: verdict, color = "High", "green"
    elif score >= 50: verdict, color = "Medium", "orange"
    else: verdict, color = "Low", "red"

    return score, verdict, color, matched, missing

# --- Upload JD and Resumes ---
jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"], key="jd")
resumes = st.file_uploader("Upload Resume PDFs", type=["pdf"], accept_multiple_files=True, key="resumes")

if st.button("Evaluate ▶️"):
    if not jd_file or not resumes:
        st.error("Upload both JD and resumes to continue!")
        st.stop()

    jd_text = read_pdf(jd_file)
    keywords = extract_keywords_phrases(jd_text, top_n=30)
    st.subheader("🔑 Extracted JD Keywords & Phrases")
    st.write(", ".join(keywords))

    # JD preview
    with st.expander("Preview Job Description"):
        st.write(jd_text[:1500] + "..." if len(jd_text) > 1500 else jd_text)

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

        # Resume analysis panel
        with st.expander(f"📄 {r.name} Analysis"):
            st.markdown(f"**Score:** {score}%  |  **Verdict:** <span style='color:{color}'>{verdict}</span>", unsafe_allow_html=True)
            st.markdown("**Matched Skills:**")
            st.markdown(", ".join(matched) if matched else "None")
            st.markdown("**Missing Skills:**")
            # highlight missing skills in red
            missing_html = ", ".join([f"<span style='color:red'>{m}</span>" for m in missing]) if missing else "None"
            st.markdown(missing_html, unsafe_allow_html=True)

    # Summary Table
    st.subheader("📊 Summary Table")
    df = pd.DataFrame(results)
    st.dataframe(df)

    # Download CSV
    st.download_button("💾 Download CSV", df.to_csv(index=False).encode(), "resume_scores.csv", "text/csv")


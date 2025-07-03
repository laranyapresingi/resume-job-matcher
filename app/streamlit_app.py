import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from parser import extract_text_from_pdf
from utils.helper import preprocess_text
from matcher import compute_similarity

st.set_page_config(page_title="Resume vs JD Matcher", layout="centered")

st.title("🤖 Resume vs Job Description Matcher")
st.markdown("Upload your **resume** and paste a **job description** to see the match score.")

# --- Upload Resume ---
uploaded_resume = st.file_uploader("📄 Upload your Resume (PDF)", type=["pdf"])

# --- Paste Job Description ---
job_description = st.text_area("📋 Paste Job Description here")

# --- Run Matching ---
if uploaded_resume and job_description and st.button("🔍 Match"):
    with open("temp_resume.pdf", "wb") as f:
        f.write(uploaded_resume.read())
    
    # Parse
    resume_raw = extract_text_from_pdf("temp_resume.pdf")
    jd_raw = job_description

    # Preprocess
    resume_clean = preprocess_text(resume_raw)
    jd_clean = preprocess_text(jd_raw)

    # Match
    score = compute_similarity(resume_clean, jd_clean)

    st.success(f"✅ Match Score: {score * 100:.2f}%")

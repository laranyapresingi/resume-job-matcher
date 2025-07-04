import streamlit as st
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

from app.parser import extract_text_from_pdf
from utils.helper import preprocess_text
from app.matcher import compute_similarity

st.title("🔍 Match Results")

if "resume_path" not in st.session_state or "jd_text" not in st.session_state:
    st.warning("No input found. Go to the homepage first.")
    st.stop()

# Extract and preprocess
resume_text = extract_text_from_pdf(st.session_state.resume_path)
jd_text = st.session_state.jd_text

resume_clean = preprocess_text(resume_text)
jd_clean = preprocess_text(jd_text)

# Compute similarity
score = compute_similarity(resume_clean, jd_clean)

if score < 0.50:
    st.error(f"Your score is too low :( {score * 100:.2f}%")
else:
    st.success(f"✅ Match Score: {score * 100:.2f}%")

# Optional: Show overlap
keywords = set(resume_clean.split()) & set(jd_clean.split())
st.info("🔑 Top Overlapping Keywords: " + ", ".join(list(keywords)[:15]))

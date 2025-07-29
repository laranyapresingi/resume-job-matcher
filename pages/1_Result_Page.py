import streamlit as st
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

from app.parser import extract_text_from_pdf
from utils.helper import preprocess_text,clean_input_text
from app.matcher import compute_similarity,test_models,compute_enhanced_similarity,section_aware_similarity,extract_skills_section

st.title("🔍 Match Results")

if "resume_path" not in st.session_state or "jd_text" not in st.session_state:
    st.warning("No input found. Go to the homepage first.")
    st.stop()

# Extract and preprocess
resume_text = extract_text_from_pdf(st.session_state.resume_path)
jd_text = st.session_state.jd_text
resume_text2 = clean_input_text(resume_text)
jd_text2 = clean_input_text(jd_text)
st.write(jd_text2)
resume_extract = extract_skills_section(resume_text2)
job_extract = extract_skills_section(jd_text2)
resume_clean =  preprocess_text(resume_extract)
jd_clean = preprocess_text(job_extract)
st.write(jd_clean)
# a = extract_skills_section(jd_clean)



# # Compute similarity

score = compute_enhanced_similarity(resume_clean, jd_clean)

# # model_results = test_models(resume_clean, jd_clean)
# # st.success(model_results)

if score['combined_score']< 0.50:
    st.error(f"Your score is too low :( {score['combined_score']* 100:.2f}%")
else:
    st.success(f"✅ Match Score: {score['combined_score']* 100:.2f}%")
    # st.success(model_results)

# Optional: Show overlap
keywords = set(resume_clean.split()) & set(jd_clean.split())
st.info("🔑 Top Overlapping Keywords: " + ", ".join(list(keywords)[:15]))

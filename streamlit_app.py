import streamlit as st
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))
from app.job_roles import get_roles, get_job_descriptions

st.set_page_config(page_title="Resume Matcher", layout="centered")
st.title("🎯 Resume Matcher")

# Upload resume
resume_file = st.file_uploader("📄 Upload your Resume (PDF)", type=["pdf"])

# Step 1: Choose Role Category (SDE or Data Scientist)
role_type = st.selectbox("Select the Role You're Applying For", get_roles())

# Step 2: Paste or select a specific JD
use_custom = st.radio("How do you want to provide the Job Description?", ["Paste my own", "Choose from presets"])

job_descriptions = get_job_descriptions()

if use_custom == "Paste my own":
    job_description = st.text_area("📋 Paste Job Description here")
else:
    specific_role = st.selectbox(f"Select a {role_type} role", list(job_descriptions[role_type].keys()))
    job_description = job_descriptions[role_type][specific_role]
    st.code(job_description)

# Proceed
if st.button("➡️ Match Resume"):
    if resume_file and job_description:
        with open("temp_resume.pdf", "wb") as f:
            f.write(resume_file.read())
        st.session_state.resume_path = "temp_resume.pdf"
        st.session_state.jd_text = job_description
        st.switch_page("pages/1_Result_Page.py")
    else:
        st.warning("Please upload a resume and provide a job description.")

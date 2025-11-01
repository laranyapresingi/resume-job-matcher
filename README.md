# Resume Matcher Web App

A modern, interactive resume matcher that leverages advanced NLP and machine learning to assess candidate-job fit. Whether you’re a recruiter or job seeker, this app provides a transparent and actionable breakdown of how well a resume matches a specific job description.

## Features

- **File Uploads:** Upload resumes (PDF) and job descriptions from your device.
- **Role Selection:** Choose from preset roles (SDE, Data Scientist) or enter custom JDs.
- **Semantic Scoring:** Uses `sentence-transformers/all-MiniLM-L6-v2` for deep semantic matching between resumes and job descriptions.
- **Keyword & Skills Analysis:** Applies TF-IDF vectorization to capture keyword relevance, automatically extracts and compares hard/soft skills.
- **Detailed Metrics:** Provides semantic score, keyword score, skills match score and overall compatibility.
- **Skills Gap Analysis:** Identifies matched and missing skills, helping candidates and recruiters focus preparation efforts.

## How It Works

1. **Upload your resume** and select the role/job description.
2. **Paste or select** the relevant job description.
3. **Specify experience and level**—entry, mid, or senior.
4. Review your detailed match score and skill breakdown to identify improvement areas.

## Tech Stack

- Python
- Streamlit
- sentence-transformers (all-MiniLM-L6-v2)
- scikit-learn (TF-IDF)
- Responsive UI/UX

## Getting Started

1. Clone the repo:  
   `git clone https://github.com/<your-username>/resume-matcher.git`
2. Install requirements:  
   `pip install -r requirements.txt`
3. Run Streamlit app:  
   `streamlit run streamlit_app.py`

## Example Screenshots

  <img width="1361" height="767" alt="image" src="https://github.com/user-attachments/assets/ef614402-993e-455d-bb11-557cc0e10314" />

  <img width="820" height="811" alt="image" src="https://github.com/user-attachments/assets/a8d75b28-8c6f-4177-b61e-7d8c6348458c" />


## License

This project is licensed under the MIT License.

---

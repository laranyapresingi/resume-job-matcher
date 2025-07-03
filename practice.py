print("hello")
from utils.helper import preprocess_text

# sample = '''Junior Data Scientist
#  TATA CONSULTANCY SERVICES
#  Oct 2023– Present
#  Hyderabad,IND
#  • Built and deployed a machine learning model to classify email case types within Microsoft Dynamics CRM
#  365, improving operational efficiency.
#  • Migrated the ML model endpoint from Azure Classic to Azure Portal, transitioning from v1 to v2 (Managed
#  Endpoint) for enhanced performance, scalability, and reliability.
#  • Implemented autoscaling to dynamically allocate resources, reducing endpoint load, preventing failures, and
#  optimizing compute resources.
#  • Improved model accuracy from 82% to 88% by implementing data augmentation, feature engineering,
#  and advanced preprocessing techniques, ensuring higher prediction reliability and deployed SVC model as highest
#  performing classifier
#  • Automated monthly data analysis to evaluate model performance, integrating insights into a Power BI
#  dashboard for real-time monitoring.
#  Machine Learning Intern
#  RASHTRIYA ISPAT NIGAM LIMITED (RINL)
#  May 2022– July 2022
#  Visakhapatnam,IND
#  • Developed a machine learning model to predict critical temperatures (skin temperature and top gas
#  temperature) for RINL’s blast furnace
#  • Utilized Pandas and Matplotlib for data cleaning and visualization and performed regression analysis to identify
#  the most optimized predictive model for accurate temperature forecasting.
#  • Deployed the trained model using Streamlit, enabling an interactive and user-friendly interface for real-time
#  predictions.'''
# cleaned = preprocess_text(sample)
# print(cleaned)
# from app.parser import extract_text_from_pdf, extract_text_from_txt

# resume_text = extract_text_from_pdf("sample_data/resume_2.pdf")
# jd_text = extract_text_from_txt("sample_data/jd_sde_amazon.txt")

# # print("RESUME SAMPLE:\n", resume_text[:1000])
# print("\nJD SAMPLE:\n", jd_text)

from app.parser import extract_text_from_pdf, extract_text_from_txt
from utils.helper import preprocess_text
from app.matcher import compute_similarity

# Paths to your sample data
resume_raw = extract_text_from_pdf("sample_data/resume_2.pdf")
jd_raw = extract_text_from_txt("sample_data/jd_data_scientist.txt")

# Preprocess
resume_clean = preprocess_text(resume_raw)
jd_clean = preprocess_text(jd_raw)

# Match
score = compute_similarity(resume_clean, jd_clean)
print("✅ Match Score:", score)

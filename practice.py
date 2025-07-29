print("hello")
# from utils.helper import preprocess_text

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

# from app.parser import extract_text_from_pdf, extract_text_from_txt
# from utils.helper import preprocess_text
# from app.matcher import compute_similarity

# # Paths to your sample data
# resume_raw = extract_text_from_pdf("sample_data/resume_2.pdf")
# jd_raw = extract_text_from_txt("sample_data/jd_data_scientist.txt")

# # Preprocess
# resume_clean = preprocess_text(resume_raw)
# jd_clean = preprocess_text(jd_raw)

# # Match
# score = compute_similarity(resume_clean, jd_clean)
# print("✅ Match Score:", score)
from utils.helper import preprocess_text
text='''Key Responsibilities:

Solution Development: Design, implement, and deploy scalable data solutions utilizing Cognite Data Fusion, focusing on data modeling, UNS, and ontologies to address industry-specific challenges. Data Analysis: Analyze large and complex data sets to identify trends, insights, and opportunities, supporting solution development and business strategies. Collaboration: Collaborate with cross-functional teams to understand data needs and translate them into data science solutions, ensuring seamless integration and operationalization of digital solutions across various domains. Client Engagement: Engage with clients to understand their business objectives, lead discovery workshops, and provide expert guidance on data-driven strategies and potential challenges. Visualization: Develop dashboards and visualizations using tools such as Power BI, Grafana, or web development frameworks like Plotly Dash and Streamlit to effectively communicate data insights. Mentorship: Provide guidance and mentorship to junior team members, promoting best practices in data science and software development.

Qualifications: Educational Background: Master’s or PhD degree in a quantitative field. Experience: Minimum of 2 years of experience in data science, with a strong background in developing analytical solutions within domains such as pharma, oil and gas, manufacturing, or power & utilities. Technical Skills: Proficiency in Python and its data ecosystem (pandas, numpy), machine learning libraries (scikit-learn, keras), and experience with SQL. Visualization Tools: Experience with data visualization tools like Power BI, Grafana, Tableau, or web development frameworks such as Plotly Dash and Streamlit. Software Practices: Strong understanding of software development practices, including version control (e.g., Git), automated testing, and documentation. Cloud Platforms: Experience with cloud services such as GCP, Azure, or AWS is advantageous. Domain Knowledge: Familiarity with industrial data management concepts, including Unified Namespace (UNS), ontologies, and data product identification. Communication Skills: Excellent communication and collaboration skills, with the ability to work with cross-functional teams and stakeholders. Leadership: Demonstrated ability to lead projects and mentor junior team members.'''
def clean_input_text(text):
    import re
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[ \xa0]+', ' ', text)  # Replace weird whitespaces
    return text.strip()
def extract_skills_section(text):
    """Extract skills/technical section from resume/job description."""
    lines = text.split('\n')
    skills_text = ""

    SECTION_HEADERS = [
        'skills', 'technical skills', 'technologies', 'tools',
        'programming languages', 'core competencies',
        'qualifications', 'preferred qualifications','key responsibilities'
    ]

    for i, line in enumerate(lines):
        print(i)
        line_lower = line.lower().strip().rstrip(":")
        print(line_lower)
        if any(line_lower.startswith(header) for header in SECTION_HEADERS):
            # Extract next up to 10 lines (non-empty)
            print("entered 1")
            for j in range(i + 1, min(i + 11, len(lines))):
                print(j)
                if lines[j].strip():  # skip empty lines
                    skills_text += lines[j].strip() + "/n  "
            break

    extracted = skills_text.strip()
    print(f"Extracted skills: {extracted}")
    return extracted

a=clean_input_text(text)
b= preprocess_text(a)
print(extract_skills_section(b))
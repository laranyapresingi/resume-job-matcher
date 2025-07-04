# def get_predefined_roles():
#     return {
#         "Experian Data Scientist": '''Job Description

# Experian Innovation Lab is a research and development unit at Experian formed with the desire to work in collaboration with Experian's teams to enhance relationships with clients and acquire strategic datasets. Using our comprehensive understanding of individuals, markets and economies, we help organizations develop customer relationships to make their businesses more profitable.

# As a Data Scientist at the Experian Innovation Lab, you will help develop analytical solutions, contributing to product prototypes, and helping evaluate data assets. You will bring experience in predictive modeling, machine learning, and deep learning to this position. You will report into the VP of Data Science

# Responsibilities

# Craft advanced machine learning analytical solutions to extract insights from diverse structured and unstructured data sources.
# Unearth data value by selecting and applying the right machine learning, deep learning and processing techniques.
# Refine data manipulation and retrieval through the design of efficient data structures and storage solutions.
# Innovate with tools designed for data processing and information retrieval.
# Dissect and document vast datasets, analyzing them to highlight patterns and insights.
# Solve complex challenges by developing impactful algorithms.
# Ensure model excellence by validating performance scores and analyzing Return on investment and benefits.
# Articulate model processes and outcomes, documenting and presenting findings and performance metrics.

# Qualifications

# Advanced degree in Machine Learning, Data Science, AI, Computer Science, or a related quantitative field.
# 1 or more years of experience in AI, data science, or predictive modeling
# Proficiency in at least one programming language, with coding skills in Python
# Experience with deep learning (CNN, RNN, LSTM, attention models), machine learning methodologies (SVM, GLM, boosting, random forest), graph models, or, reinforcement learning.
# Experience with open-source tools for deep learning and machine learning technology such as pytorch, Keras, tensorflow, scikit-learn, pandas.
# Experience with large data analysis using Spark (pySpark preferred)
# Experience with LLMs and the relevant tools in the Generative AI domain.
# Experience developing advanced language models
# Experience applying Generative AI-based tools
# Experience with Hadoop and NoSQL related technologies such as Map Reduce, Hive, HBase, mongoDB, Cassandra.
# Experience modifying and applying advanced algorithms to address practical problems''',


# "Cognizant Data Scientist": '''Job Summary

# As a Data Scientist with expertise in Azure Analysis Services and Python3 you will play a crucial role in designing and implementing innovative solutions for our Retail Banking projects. With a hybrid work model, you will collaborate with cross-functional teams to enhance our technological capabilities and drive business success. Your contributions will directly impact our company's growth and societal advancement.

# Responsibilities

# Lead the design and development of scalable solutions using Azure Analysis Services to meet business requirements.
# Oversee the implementation of Python3 scripts to automate processes and improve system efficiency.
# Collaborate with cross-functional teams to gather and analyze requirements for Retail Banking projects.
# Provide technical guidance and mentorship to junior developers to foster a culture of continuous learning.
# Ensure the integration of new technologies aligns with existing systems and business goals.
# Conduct code reviews to maintain high-quality standards and ensure best practices are followed.
# Develop and maintain documentation for system architecture and processes to support future development.
# Troubleshoot and resolve complex technical issues to minimize downtime and enhance user experience.
# Participate in agile development processes to ensure timely delivery of project milestones.
# Communicate effectively with stakeholders to provide updates on project progress and challenges.
# Utilize data analytics to identify opportunities for process improvement and innovation.
# Contribute to the strategic planning of technology initiatives to support business growth.
# Stay updated with industry trends and emerging technologies to drive continuous improvement.

# Qualifications

# Demonstrate proficiency in Azure Analysis Services and Python3 with a minimum of 7 years of experience.
# Exhibit strong problem-solving skills and the ability to work collaboratively in a hybrid work environment.
# Possess experience in Retail Banking domain which is considered a plus.
# Show excellent communication skills to effectively interact with team members and stakeholders.
# Display a proactive approach to learning and adapting to new technologies and methodologies.''',
#         "Google Data Scientist": '''About the job
# Minimum qualifications:

# Master's degree in Statistics, Data Science, Mathematics, Physics, Economics, Operations Research, Engineering, or a related quantitative field.
# 3 years of work experience using analytics to solve product or business problems, coding (e.g., Python, R, SQL), querying databases or statistical analysis, or a PhD degree.

# Preferred qualifications:

# 5 years of work experience using analytics to solve product or business problems, coding (e.g., Python, R, SQL), querying databases or statistical analysis, or a PhD degree.''',
#         "Pinterest Data Scientist": '''What You’ll Do

# Develop best practices for instrumentation and experimentation and communicate those to product engineering teams to help us fulfill our mission - to bring everyone the inspiration to create a life they love
# Bring scientific rigor and statistical methods to the challenges of product creation, development and improvement with an appreciation for the behaviors of our Pinners
# Build and prototype analysis pipelines iteratively to provide insights at scale while developing comprehensive knowledge of data structures and metrics, advocating for changes where needed for product development
# Work cross-functionally to build and communicate key insights, and collaborate closely with product managers, engineers, designers, and researchers to help build the next experiences on Pinterest


# What We’re Looking For

# 4+ years of experience analyzing data in a fast-paced, data-driven environment with proven ability to apply scientific methods to solve real-world problems on web-scale data
# Extensive experience solving analytical problems using quantitative approaches including in the fields of Machine Learning, Statistical Modeling, Forecasting, Econometrics or other related fields
# Experience using machine learning and deep learning frameworks, such as PyTorch, TensorFlow or scikit-learn
# A scientifically rigorous approach to analysis and data, and a well-tuned sense of skepticism, attention to detail and commitment to high-quality, results-oriented output
# Ability to manipulate large data sets with high dimensionality and complexity; fluency in SQL (or other database languages) and a scripting language (Python or R)
# Excellent communication skills and ability to explain learnings to both technical and non-technical partners
# A team player who’s able to partner with cross-functional leadership to quickly turn insights into actions
# Bachelor’s degree in Computer Science, a related field or equivalent experience

# ''',
#         "FinTech Data Scientist": "Work on fraud detection, price prediction using LightGBM, deploy via AWS or GCP.",
#         "Amazon SDE": '''Key job responsibilities

#  Collaborate with experienced cross-disciplinary Amazonians to conceive, design, and bring innovative products and services to market.
#  Design and build innovative technologies in a large distributed computing environment, and help lead fundamental changes in the industry.
#  Create solutions to run predictions on distributed systems with exposure to innovative

# technologies at incredible scale and speed.

#  Build distributed storage, index, and query systems that are scalable, fault-tolerant, low cost, and easy to manage/use.
#  Ability to design and code the right solutions starting with broadly defined problems.
#  Work in an agile environment to deliver high-quality software.

# Basic Qualifications

#  Knowledge of computer science fundamentals such as object-oriented design, operating systems, algorithms, data structures, and complexity analysis
#  Knowledge of programming languages such as C/C++, Python, Java or Perl
#  Year of Graduation 2026

# Preferred Qualifications

#  Bachelor's degree in computer science, computer engineering, or related field
# '''
#     }
def get_roles():
    return ["SDE", "Data Scientist"]

def get_job_descriptions():
    return {
        "SDE": {
            "Backend Developer": "Design and build scalable backend systems using Java, Python, and cloud platforms.",
            "Frontend Developer": "Develop responsive UIs with React, TypeScript, and integrate APIs securely.",
            "DevOps Engineer": "Automate CI/CD pipelines, manage deployments using Docker and Kubernetes.",
        },
        "Data Scientist": {
            "NLP Scientist": "Build language models for classification and text generation tasks using BERT/GPT.",
            "ML Engineer": "Design scalable ML pipelines using TensorFlow, deploy on AWS/GCP.",
            "Data Analyst": "Analyze KPIs, create dashboards using SQL and Tableau, and provide data insights."
        }
    }

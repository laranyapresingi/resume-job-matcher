from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-V2')

def get_embedding(text):
    return model.encode(text)

def compute_similarity(resume_text,jd_text):
    resume_vec = get_embedding(resume_text)
    jd_vec= get_embedding(jd_text)

    similarity = cosine_similarity([resume_vec],[jd_vec])
    return round(float(similarity[0][0]),4)
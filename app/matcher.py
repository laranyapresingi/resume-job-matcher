from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils.helper import preprocess_text

model = SentenceTransformer('all-roberta-large-v1')

def get_embedding(text):
    return model.encode(text)

def compute_similarity(resume_text,jd_text):
    # resume_vec = get_embedding(resume_text)
    # jd_vec= get_embedding(jd_text)

    # similarity = cosine_similarity([resume_vec],[jd_vec])
    # return round(float(similarity[0][0]),4)
    resume_clean = preprocess_text(resume_text)
    jd_clean = preprocess_text(jd_text)
    
    # Get embeddings
    resume_vec = get_embedding(resume_clean)
    jd_vec = get_embedding(jd_clean)
    
    # Calculate semantic similarity
    semantic_similarity = cosine_similarity([resume_vec], [jd_vec])[0][0]
    print("hi")
    # Calculate keyword similarity
    keyword_similarity = calculate_keyword_overlap(resume_clean, jd_clean)
    print(keyword_similarity)
    # Combine scores
    final_score = (semantic_similarity * 0.7) + (keyword_similarity * 0.3)
    
    return round(final_score, 4)

def test_models(resume_text, jd_text):
    models_to_test = [
        'all-MiniLM-L6-v2',      # Your current model
        'all-mpnet-base-v2',     # Better semantic understanding
        'all-roberta-large-v1',   # Highest quality
        'paraphrase-MiniLM-L6-v2' # Good for paraphrasing
    ]
    
    results = {}
    for model_name in models_to_test:
        try:
            model = SentenceTransformer(model_name)
            resume_vec = model.encode(resume_text)
            jd_vec = model.encode(jd_text)
            similarity = cosine_similarity([resume_vec], [jd_vec])[0][0]
            results[model_name] = round(similarity, 4)
        except Exception as e:
            print(f"Error with {model_name}: {e}")
    
    return results


def calculate_keyword_overlap(resume_text, jd_text):
    """Calculate keyword overlap score using TF-IDF"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    try:
        # Use TF-IDF for better keyword weighting
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),  # 1-grams and 2-grams
            max_features=1000,
            stop_words='english'
        )
        
        tfidf_matrix = vectorizer.fit_transform([jd_text, resume_text])
        keyword_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return round(keyword_similarity, 4)
    except:
        # Fallback to simple word overlap
        resume_words = set(resume_text.split())
        jd_words = set(jd_text.split())
        
        if len(jd_words) == 0:
            return 0.0
        
        overlap = resume_words.intersection(jd_words)
        return round(len(overlap) / len(jd_words), 4)

def compute_enhanced_similarity(resume_text, jd_text):

    """Compute similarity using multiple methods"""
    print("entering compute enchanced similarity")
    # Preprocess both texts
    # resume_clean = preprocess_text(resume_text)
    # jd_clean = preprocess_text(jd_text)
    
    # 1. Semantic similarity (your current approach)
    resume_vec = get_embedding(resume_text)
    jd_vec = get_embedding(jd_text)
    semantic_similarity = cosine_similarity([resume_vec], [jd_vec])[0][0]
    print("printing semantic similarity")
    # 2. Keyword overlap similarity
    keyword_similarity = calculate_keyword_overlap(resume_text, jd_text)
    print("printing keyword similarity")
    # 3. Combined score (weighted average)
    combined_score = (semantic_similarity * 0.4) + (keyword_similarity * 0.6)
    print(f"Combined score {combined_score*100}")
    return {
        'semantic_score': round(semantic_similarity, 4),
        'keyword_score': round(keyword_similarity, 4),
        'combined_score': round(combined_score, 4)
    }


def extract_skills_section(text):
    """Extract skills/technical section from resume/job description."""
    lines = text.split('\n')
    skills_text = ""

    SECTION_HEADERS = [
        'skills', 'technical skills', 'technologies', 'tools',
        'programming languages', 'core competencies',
        'qualifications', 'preferred qualifications'
    ]

    for i, line in enumerate(lines):
        # print(f" i : {i},{line}")
        line_lower = line.lower().strip().rstrip(":")
        
        if any(line_lower.startswith(preprocess_text(header)) for header in SECTION_HEADERS):
            # Extract next up to 10 lines (non-empty)
            for j in range(i + 1, min(i + 4, len(lines))):
                print(f"j: {j}")
                if lines[j].strip():  # skip empty lines
                    skills_text += lines[j].strip() + " "
            break

    extracted = skills_text.strip()
    print(f"Extracted skills: {extracted}")
    if extracted:
        return extracted
    else:
        return text



def section_aware_similarity(resume_text, jd_text):
    """Calculate similarity with section-specific weighting"""
    print("Calculating section aware similarity")
    # Extract skills section
    skills_section = extract_skills_section(resume_text)
    jd_skills  = extract_skills_section(jd_text)
    print(jd_skills)
    # Calculate similarities
    
    overall_scores = compute_enhanced_similarity(resume_text, jd_text)
    print(f"\n Calculating overall score {overall_scores}")
    skills_scores = compute_enhanced_similarity(skills_section, jd_skills)
    print(f" \n Calculating skill  score {skills_scores}")
    
    # Weight skills section higher
    final_score = (overall_scores['combined_score'] * 0.4) + (skills_scores['combined_score'] * 0.6)
    
    return {
        'overall_similarity': overall_scores,
        'skills_similarity': skills_scores,
        'final_score': round(final_score, 4)
    }
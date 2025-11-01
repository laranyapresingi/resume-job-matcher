# fixed_resume_matcher.py
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import re
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from pathlib import Path
import json
from functools import lru_cache
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MatchingResult:
    """Structured result for resume-job matching"""
    overall_score: float
    semantic_score: float
    keyword_score: float
    skills_match_score: float
    experience_score: float
    education_score: float
    matched_skills: List[str]
    missing_skills: List[str]
    skill_categories: Dict[str, float]
    recommendations: List[str]
    confidence: float
    debug_info: Dict = None  # Added for debugging

class AdvancedResumeJobMatcher:
    """Fixed resume-job matcher with proper debugging"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', debug_mode: bool = True):
        """Initialize with debug mode enabled by default"""
        self.debug_mode = debug_mode
        self.model = self._load_model_with_cache(model_name)
        self.nlp = self._load_spacy_model()
        self.skills_database = self._load_skills_database()
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),  # Reduced from (1,3) to avoid over-fragmentation
            max_features=3000,   # Reduced for better performance
            stop_words='english',
            lowercase=True,
            min_df=1,  # Include all terms
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9+#.\-]*[a-zA-Z0-9+#]\b|\b[a-zA-Z]\b'  # Better token pattern
        )
        
    @lru_cache(maxsize=1)
    def _load_model_with_cache(self, model_name: str) -> SentenceTransformer:
        """Load sentence transformer with caching"""
        try:
            model = SentenceTransformer(model_name)
            if self.debug_mode:
                print(f"DEBUG: Loaded model {model_name}")
            return model
        except Exception as e:
            logger.warning(f"Failed to load {model_name}, falling back to default")
            return SentenceTransformer('all-MiniLM-L6-v2')
    
    def _load_spacy_model(self):
        """Load spaCy model with error handling"""
        try:
            nlp = spacy.load("en_core_web_sm")
            if self.debug_mode:
                print("DEBUG: Loaded spaCy model successfully")
            return nlp
        except OSError:
            logger.warning("spaCy model not found. Using basic preprocessing.")
            return None
    
    def _load_skills_database(self) -> Dict[str, List[str]]:
        """Load comprehensive skills database with normalized terms"""
        """FIXED: More comprehensive skills database with your exact terms"""
        skills_db = {
        'programming_languages': [
            'python', 'java', 'javascript', 'c++', 'c/c++', 'c#', 'c', 'sql',
            'html', 'css', 'html/css', 'php', 'ruby', 'go', 'rust', 'scala',
            'r', 'matlab', 'julia'
        ],
        'ml_frameworks': [
            'tensorflow', 'pytorch', 'scikit-learn', 'scikit learn', 'sklearn',
            'keras', 'xgboost', 'lightgbm', 'catboost', 'hugging face', 
            'transformers', 'machine learning', 'deep learning', 'neural networks',
            'natural language processing', 'nlp', 'computer vision', 'opencv'
        ],
        'data_tools': [
            'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly', 'bokeh',
            'tableau', 'power bi', 'powerbi', 'looker', 'qlik', 'excel',
            'jupyter', 'jupyter notebook', 'git', 'github'
        ],
        'cloud_platforms': [
            'aws', 'amazon web services', 'azure', 'microsoft azure', 'gcp', 
            'google cloud', 'docker', 'kubernetes', 'github actions',
            'azure devops', 'ml studio', 'managed endpoints'
        ],
        'databases': [
            'mysql', 'postgresql', 'mongodb', 'sqlite', 'oracle',
            'sql server', 'redis', 'cassandra', 'elasticsearch',
            'sql4cds', 'database management systems', 'dbms'
        ],
        'data_mining': [
            'data mining', 'data processing', 'data visualization',
            'data analysis', 'statistical analysis', 'business intelligence'
        ]
    }
    
        if self.debug_mode:
            total_skills = sum(len(skills) for skills in skills_db.values())
            print(f"DEBUG: Loaded {total_skills} skills across {len(skills_db)} categories")
        
        return skills_db
    def preprocess_text(self, text: str) -> str:
        if not text:
            return ""
    
        original_text = text
        
        # MUCH gentler preprocessing
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        # DON'T lowercase yet - preserve case for proper matching
        # Only remove truly problematic characters, keep /,+,#,-
        text = re.sub(r'[^\w\s+#.\-/(),:]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        if self.debug_mode:
            print(f"DEBUG: Text preprocessing:")
            print(f"  Original: {original_text[:100]}...")
            print(f"  Processed: {text[:100]}...")
        
        return text
    
    def extract_skills_advanced(self, text: str) -> Dict[str, List[str]]:
            # Keep original text for case-sensitive matching
        text_original = text
        text_lower = text.lower()
        found_skills = {category: [] for category in self.skills_database.keys()}
        
        if self.debug_mode:
            print(f"DEBUG: Starting skills extraction from text length: {len(text)}")
        
        # PHASE 1: Direct exact matching (case-insensitive)
        for category, skills in self.skills_database.items():
            for skill in skills:
                # Try multiple matching strategies
                skill_variations = [
                    skill,
                    skill.replace('-', ' '),  # "scikit-learn" -> "scikit learn"
                    skill.replace(' ', '-'),  # "power bi" -> "power-bi"
                    skill.replace('/', ' '),  # "html/css" -> "html css"
                ]
                
                for skill_var in skill_variations:
                    # Case-insensitive word boundary search
                    if len(skill_var.split()) == 1:
                        # Single word - use word boundaries
                        pattern = r'\b' + re.escape(skill_var) + r'\b'
                    else:
                        # Multi-word - simple contains check
                        pattern = re.escape(skill_var)
                    
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        if skill not in found_skills[category]:
                            found_skills[category].append(skill)
                            if self.debug_mode:
                                print(f"DEBUG: Found skill '{skill}' in category '{category}'")
        
        # PHASE 2: Resume section-specific extraction
        # Look for "Technical Skills" section specifically
        lines = text.split('\n')
        in_skills_section = False
        skills_section_text = ""
        
        for line in lines:
            line_clean = line.strip()
            if re.match(r'^(technical\s+skills?|skills?|programming|technologies)', line_clean, re.IGNORECASE):
                in_skills_section = True
                if self.debug_mode:
                    print(f"DEBUG: Found skills section starting with: {line_clean}")
                continue
            
            # Stop at next major section
            if in_skills_section and re.match(r'^(experience|education|projects|certifications)', line_clean, re.IGNORECASE):
                in_skills_section = False
                break
            
            if in_skills_section and line_clean:
                skills_section_text += " " + line_clean
        
        if skills_section_text:
            if self.debug_mode:
                print(f"DEBUG: Extracted skills section: {skills_section_text[:200]}...")
            
            # Extract from skills section with more patterns
            self._extract_from_skills_section(skills_section_text, found_skills)
        
        # PHASE 3: Pattern-based extraction for context mentions
        skill_patterns = [
            r'(?:experience|skilled|proficient|knowledge|familiar)\s+(?:with|in|using)\s+([^,.;]+)',
            r'(?:technologies|tools|languages)\s*:\s*([^.\n]+)',
            r'(?:including|such as|like)\s+([^,.;]+)',
            r'using\s+([^,.;]+)',
        ]
        
        for pattern in skill_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                skills_text = match.group(1).strip()
                self._process_skills_text(skills_text, found_skills)
        
        # Remove duplicates and sort
        for category in found_skills:
            found_skills[category] = sorted(list(set(found_skills[category])))
        
        if self.debug_mode:
            print(f"DEBUG: Final skills extraction results:")
            total_found = 0
            for category, skills in found_skills.items():
                if skills:
                    print(f"  {category}: {skills}")
                    total_found += len(skills)
            print(f"DEBUG: Total skills found: {total_found}")
        
        return found_skills
    
    def calculate_experience_score(self, resume_text: str, jd_text: str,years_experience :  str) -> float:
        """FIXED: Better experience extraction and scoring"""
        resume_years = self._extract_years_experience(resume_text)
        jd_years = self._extract_years_experience(jd_text)
        years = int(years_experience)
        
        if self.debug_mode:
            print(f"DEBUG: Experience - Resume: {years}, JD: {jd_years}")
        
        if not jd_years:
            return 0.7  # Higher neutral score if no requirement specified
        
        jd_min = min(jd_years)
        if not resume_years:
            return years  # Low score if no experience found in resume
        
        resume_max = max(resume_years)
        
        if years >= jd_min:
            # Give full score if meets requirement, bonus for exceeding
            return min(1.0, 0.9 + (years - jd_min) * 0.05)
        else:
            # Partial credit for less experience
            return max(0.1, (years / jd_min) * 0.8)
    
    def _extract_years_experience(self, text: str) -> List[int]:
        """FIXED: Better patterns for experience extraction"""
        patterns = [
            r'(\d+)[\s\-]*(?:years?|yrs?|y)[\s\-]*(?:of\s+)?(?:experience|exp)',
            r'(?:experience|exp)[\s\w]*(\d+)[\s\-]*(?:years?|yrs?|y)',
            r'(\d+)\+?\s*(?:years?|yrs?|y)[\s\-]*(?:experience|exp)',
            r'(\d+)\+?\s*(?:years?|yrs?)\s+(?:in|of|with)',
        ]
        
        years = []
        text_lower = text.lower()
        
        for pattern in patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                try:
                    year_val = int(match.group(1))
                    if 0 < year_val <= 50:  # Reasonable bounds
                        years.append(year_val)
                except (ValueError, IndexError):
                    continue
        
        return list(set(years))  # Remove duplicates
    
    def calculate_education_score(self, resume_text: str, jd_text: str) -> float:
        """Education level matching with better scoring"""
        education_levels = {
            'phd': 5, 'ph.d': 5, 'doctorate': 5, 'doctoral': 5,
            'master': 4, 'masters': 4, 'msc': 4, 'ms': 4, 'mba': 4, 'ma': 4,
            'bachelor': 3, 'bachelors': 3, 'bsc': 3, 'bs': 3, 'ba': 3, 'be': 3,
            'associate': 2, 'diploma': 2,
            'high school': 1, 'secondary': 1
        }
        
        resume_level = self._get_education_level(resume_text, education_levels)
        jd_level = self._get_education_level(jd_text, education_levels)
        
        if self.debug_mode:
            print(f"DEBUG: Education - Resume level: {resume_level}, JD level: {jd_level}")
        
        if jd_level == 0:
            return 0.7  # Higher neutral score if no requirement
        
        if resume_level >= jd_level:
            return 1.0
        elif resume_level > 0:
            return max(0.3, resume_level / jd_level)
        else:
            return 0.2  # Some credit for undetected education
    
    def _get_education_level(self, text: str, levels: Dict[str, int]) -> int:
        """Extract highest education level from text"""
        text_lower = text.lower()
        max_level = 0
        
        for degree, level in levels.items():
            if re.search(r'\b' + re.escape(degree) + r'\b', text_lower):
                max_level = max(max_level, level)
        
        return max_level
    
    def match_resume_to_job(self, resume_text: str, jd_text: str, years_experience: str) -> MatchingResult:
        """FIXED: Main matching function with proper scoring"""
        start_time = time.time()
        
        if self.debug_mode:
            print(f"\n=== STARTING RESUME MATCHING ===")
            print(f"Resume length: {len(resume_text)} chars")
            print(f"JD length: {len(jd_text)} chars")
        
        # Preprocess texts
        resume_clean = self.preprocess_text(resume_text)
        jd_clean = self.preprocess_text(jd_text)
        
        # Extract skills with debug output
        resume_skills = self.extract_skills_advanced(resume_text)
        jd_skills = self.extract_skills_advanced(jd_text)
        
        # Calculate individual scores
        semantic_score = self._calculate_semantic_similarity(resume_clean, jd_clean)
        keyword_score = self._calculate_keyword_similarity(resume_clean, jd_clean)
        skills_score, matched_skills, missing_skills = self._calculate_skills_score(resume_skills, jd_skills)
        experience_score = self.calculate_experience_score(resume_text, jd_text,years_experience)
        education_score = self.calculate_education_score(resume_text, jd_text)
        
        # Calculate category-wise scores
        skill_categories = self._calculate_category_scores(resume_skills, jd_skills)
        
        # FIXED: Better weighted scoring
        overall_score = (
            semantic_score * 0.10 +    # Reduced semantic weight
            keyword_score * 0.25 +     # Increased keyword weight
            skills_score * 0.35+      # Increased skills weight (most important)
            experience_score * 0.10   # Reduced education weight
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            skills_score, missing_skills, experience_score, education_score
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(resume_text, jd_text, resume_skills, jd_skills)
        
        processing_time = time.time() - start_time
        
        if self.debug_mode:
            print(f"\n=== SCORING BREAKDOWN ===")
            print(f"Semantic Score: {semantic_score:.3f}")
            print(f"Keyword Score: {keyword_score:.3f}")
            print(f"Skills Score: {skills_score:.3f}")
            print(f"Experience Score: {experience_score:.3f}")
            # print(f"Education Score: {education_score:.3f}")
            print(f"Overall Score: {overall_score:.3f}")
            print(f"Matched Skills: {matched_skills}")
            print(f"Missing Skills: {missing_skills[:5]}...")  # Show first 5
            print(f"Processing time: {processing_time:.2f}s")
        
        debug_info = {
            'resume_clean': resume_clean[:200],
            'jd_clean': jd_clean[:200],
            'resume_skills': resume_skills,
            'jd_skills': jd_skills,
            'processing_time': processing_time
        } if self.debug_mode else None
        
        return MatchingResult(
            overall_score=round(overall_score, 3),
            semantic_score=round(semantic_score, 3),
            keyword_score=round(keyword_score, 3),
            skills_match_score=round(skills_score, 3),
            experience_score=round(experience_score, 3),
            education_score=round(education_score, 3),
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            skill_categories=skill_categories,
            recommendations=recommendations,
            confidence=round(confidence, 3),
            debug_info=debug_info
        )
    
    def _calculate_semantic_similarity(self, resume_text: str, jd_text: str) -> float:
        """Calculate semantic similarity using sentence transformers"""
        try:
            if not resume_text.strip() or not jd_text.strip():
                return 0.0
                
            resume_embedding = self.model.encode([resume_text])
            jd_embedding = self.model.encode([jd_text])
            similarity = cosine_similarity(resume_embedding, jd_embedding)[0][0]
            
            # Ensure reasonable bounds
            similarity = max(0.0, min(1.0, similarity))
            
            if self.debug_mode:
                print(f"DEBUG: Semantic similarity raw: {similarity:.3f}")
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error in semantic similarity calculation: {e}")
            return 0.0
    
    def _calculate_keyword_similarity(self, resume_text: str, jd_text: str) -> float:
        """FIXED: Calculate keyword similarity using TF-IDF"""
        try:
            if not resume_text.strip() or not jd_text.strip():
                return 0.0
            
            # Fit on both documents
            docs = [jd_text, resume_text]
            tfidf_matrix = self.vectorizer.fit_transform(docs)
            
            # Calculate similarity between JD (0) and resume (1)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            similarity = max(0.0, min(1.0, similarity))
            
            if self.debug_mode:
                print(f"DEBUG: Keyword similarity: {similarity:.3f}")
                # Show top keywords
                feature_names = self.vectorizer.get_feature_names_out()
                jd_scores = tfidf_matrix[0].toarray()[0]
                top_keywords = sorted(zip(feature_names, jd_scores), key=lambda x: x[1], reverse=True)[:10]
                print(f"DEBUG: Top JD keywords: {[kw for kw, score in top_keywords if score > 0]}")
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error in keyword similarity calculation: {e}")
            return 0.0
    
    def _calculate_skills_score(self, resume_skills: Dict, jd_skills: Dict) -> Tuple[float, List[str], List[str]]:
        """FIXED: Calculate skills matching score without duplicates"""
        all_resume_skills = []
        all_jd_skills = []
        
        for category_skills in resume_skills.values():
            all_resume_skills.extend(category_skills)
        
        for category_skills in jd_skills.values():
            all_jd_skills.extend(category_skills)
        
        # CRITICAL FIX: Remove duplicates before calculation
        all_resume_skills = list(set(all_resume_skills))
        all_jd_skills = list(set(all_jd_skills))
        
        if not all_jd_skills:
            return 0.5, [], []
        
        matched = set(all_resume_skills).intersection(set(all_jd_skills))
        missing = set(all_jd_skills) - set(all_resume_skills)
        
        # Calculate the actual score
        score = len(matched) / len(all_jd_skills)
        
        if self.debug_mode:
            print(f"DEBUG: Skills calculation FIXED:")
            print(f"  Resume skills (unique): {len(all_resume_skills)} - {all_resume_skills}")
            print(f"  JD skills (unique): {len(all_jd_skills)} - {all_jd_skills}")
            print(f"  Matched: {len(matched)} - {list(matched)}")
            print(f"  Missing: {len(missing)} - {list(missing)}")
            print(f"  Score: {len(matched)}/{len(all_jd_skills)} = {score:.3f}")
        
        return score, list(matched), list(missing)
    
    def _calculate_category_scores(self, resume_skills: Dict, jd_skills: Dict) -> Dict[str, float]:
        """Calculate category-wise skill scores"""
        category_scores = {}
        
        for category in self.skills_database.keys():
            resume_cat_skills = set(resume_skills.get(category, []))
            jd_cat_skills = set(jd_skills.get(category, []))
            
            if jd_cat_skills:
                matched = resume_cat_skills.intersection(jd_cat_skills)
                score = len(matched) / len(jd_cat_skills)
                category_scores[category] = round(score, 3)
            else:
                category_scores[category] = 0.0
        
        return category_scores
    
    def _generate_recommendations(self, skills_score: float, missing_skills: List[str], 
                                experience_score: float, education_score: float) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if skills_score < 0.7 and missing_skills:
            # Prioritize most important missing skills
            top_missing = missing_skills[:3]
            recommendations.append(f"Learn these key skills: {', '.join(top_missing)}")
        
        if skills_score < 0.5:
            recommendations.append("Consider taking online courses or certifications in the required technologies")
        
        if experience_score < 0.6:
            recommendations.append("Highlight relevant projects, internships, and practical work to demonstrate experience")
        
        if education_score < 0.7:
            recommendations.append("Consider relevant certifications or additional education to meet requirements")
        
        if skills_score > 0.8:
            recommendations.append("Strong skill match! Focus on demonstrating impact and results in your experience")
        
        if not recommendations:
            recommendations.append("Good overall match! Tailor your resume language to closely match the job description")
        
        return recommendations
    
    def _calculate_confidence(self, resume_text: str, jd_text: str, 
                            resume_skills: Dict, jd_skills: Dict) -> float:
        """Calculate confidence in the matching result"""
        confidence_factors = []
        
        # Text length factor
        resume_length = len(resume_text.split())
        jd_length = len(jd_text.split())
        length_factor = min(1.0, (resume_length * jd_length) / (100 * 100))
        confidence_factors.append(length_factor)
        
        # Skills detection factor
        resume_skills_count = sum(len(skills) for skills in resume_skills.values())
        jd_skills_count = sum(len(skills) for skills in jd_skills.values())
        skills_factor = min(1.0, (resume_skills_count + jd_skills_count) / 10)
        confidence_factors.append(skills_factor)
        
        # Text quality factor
        structure_keywords = ['experience', 'education', 'skills', 'projects', 'work']
        structure_count = sum(1 for keyword in structure_keywords if keyword in resume_text.lower())
        structure_factor = min(1.0, structure_count / 3)  # Need at least 3 for good confidence
        confidence_factors.append(structure_factor)
        
        confidence = np.mean(confidence_factors)
        
        if self.debug_mode:
            print(f"DEBUG: Confidence factors - Length: {length_factor:.2f}, Skills: {skills_factor:.2f}, Structure: {structure_factor:.2f}")
        
        return confidence

    def _process_skills_text(self, skills_text: str, found_skills: Dict):
        """Process a text snippet for skills"""
        
        # Split and clean
        candidates = re.split(r'[,;&]|\sand\s|\sor\s', skills_text)
        
        for candidate in candidates:
            candidate = candidate.strip().lower()
            if len(candidate) < 2:
                continue
            
            # Match against database
            for category, skills in self.skills_database.items():
                for skill in skills:
                    if skill.lower() in candidate or candidate in skill.lower():
                        if skill not in found_skills[category]:
                            found_skills[category].append(skill)

    def _extract_from_skills_section(self, skills_text: str, found_skills: Dict):
        """Extract skills from the dedicated skills section"""
        
        # Split by common delimiters
        skill_candidates = re.split(r'[,;:|•\n]|\sand\s|\sor\s', skills_text)
        
        for candidate in skill_candidates:
            candidate = candidate.strip()
            if len(candidate) < 2:
                continue
            
            # Clean up the candidate
            candidate = re.sub(r'^\W+|\W+$', '', candidate)  # Remove leading/trailing punctuation
            candidate = re.sub(r'^\d+\.\s*', '', candidate)  # Remove numbering
            
            if len(candidate) < 2:
                continue
            
            # Check against all known skills
            candidate_lower = candidate.lower()
            
            for category, skills in self.skills_database.items():
                for skill in skills:
                    if (skill.lower() == candidate_lower or 
                        skill.lower() in candidate_lower or 
                        candidate_lower in skill.lower()):
                        if skill not in found_skills[category]:
                            found_skills[category].append(skill)
                            if self.debug_mode:
                                print(f"DEBUG: Matched '{candidate}' -> '{skill}' in {category}")

# Enhanced testing function
def test_matcher_with_debug(extracted_jd,extracted_resume,years_experience):
    """Test the matcher with debug output"""
    print("Testing Resume Matcher with Debug Output")
    print("=" * 50)
    
    matcher = AdvancedResumeJobMatcher(debug_mode=True)
    
    
    result = matcher.match_resume_to_job(extracted_resume,extracted_jd,years_experience)
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Overall Match Score: {result.overall_score:.1%}")
    print(f"Matched Skills: {result.matched_skills}")
    print(f"Missing Skills: {result.missing_skills}")
    print(f"Recommendations: {result.recommendations}")
    print(f"Confidence: {result.confidence:.1%}")

    # print("=== SKILLS EXTRACTION DEBUG ===")
    # resume_skills = matcher.extract_skills_advanced(extracted_resume)
    # jd_skills = matcher.extract_skills_advanced(extracted_jd)

    # print(f"Resume Skills Found: {resume_skills}")
    # print(f"JD Skills Found: {jd_skills}")

    # # Flatten for comparison
    # resume_flat = [skill for skills_list in resume_skills.values() for skill in skills_list]
    # jd_flat = [skill for skills_list in jd_skills.values() for skill in skills_list]

    # print(f"Resume Skills Flat: {resume_flat}")
    # print(f"JD Skills Flat: {jd_flat}")

    # matched = set(resume_flat).intersection(set(jd_flat))
    # missing = set(jd_flat) - set(resume_flat)

    # print(f"Actually Matched: {list(matched)}")
    # print(f"Actually Missing: {list(missing)}")
    
    print(result)
    print(type(result))
    return result

# if __name__ == "__main__":
#     test_result = test_matcher_with_debug()
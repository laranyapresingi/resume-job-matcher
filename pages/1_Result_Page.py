import streamlit as st
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'.' 'app')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from app.parser import extract_text_from_pdf
from app.practice import MatchingResult,AdvancedResumeJobMatcher,test_matcher_with_debug
from utils.helper import preprocess_text,clean_input_text
from app.matcher import compute_similarity,test_models,compute_enhanced_similarity,section_aware_similarity,extract_skills_section
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');
    
    /* Change entire app font */
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Specific font for skill bubbles */
    .skill-bubble {
        font-family: 'Courier New', monospace;
        font-weight: 500;
        font-size: 0.9rem;
            
    }
    
    /* Header fonts */
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


st.title("🔍 Match Results")

if "resume_path" not in st.session_state or "jd_text" not in st.session_state:
    st.warning("No input found. Go to the homepage first.")
    st.stop()

# Extract and preprocess
resume_text = extract_text_from_pdf(st.session_state.resume_path)
jd_text = st.session_state.jd_text
years_experience=st.session_state.years_experience 
test_result = test_matcher_with_debug(jd_text,resume_text,years_experience)

# NEW CODE:
score_pct = test_result.overall_score * 100

# Color-coded score display
if score_pct >= 70:
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #4CAF50, #45a049); 
                padding: 2rem; border-radius: 15px; text-align: center; color: white;">
        <h1 style="margin: 0; font-size:40px;">{score_pct:.1f}%</h1>
        <h3>Excellent Match! 🎉</h3>
    </div>
    """, unsafe_allow_html=True)
elif score_pct >= 50:
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #2196F3, #1976D2); 
                padding: 2rem; border-radius: 15px; text-align: center; color: white;">
        <h1 style="margin: 0; font-size: 40px;">{score_pct:.1f}%</h1>
        <h3>Good Match! ✅</h3>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #FF9800, #F57C00); 
                padding: 2rem; border-radius: 15px; text-align: center; color: white;">
        <h1 style="margin: 0; font-size: 40px;">{score_pct:.1f}%</h1>
        <h3>Needs Improvement ⚠️</h3>
    </div>
    """, unsafe_allow_html=True)

# Better skills visualization
col1, col2 = st.columns(2)

with col1:
    st.markdown("### ✅ Matched Skills")
    if test_result.matched_skills:
        # Create all matched skills as bubbles in one container
        matched_skills_html = '<div class="skill-bubble" style="max-height: 300px; overflow-y: auto; padding: 1rem; border: 1px solid #e8f5e8; border-radius: 10px; background: #f9f9f9;">'
        
        for skill in test_result.matched_skills:
            matched_skills_html += f'<span style="background: #e8f5e8; color: #2e7d2e; padding: 0.3rem 0.8rem; margin: 0.2rem; border-radius: 20px; display: inline-block; font-size: 0.9rem;">✓ {skill}</span> '
        
        matched_skills_html += '</div>'
        
        # Display with proper HTML rendering
        st.markdown(matched_skills_html, unsafe_allow_html=True)
    else:
        st.info("No matched skills found")

with col2:
    st.markdown("### ❌ Consider Learning")
    if test_result.missing_skills:
        # Create all missing skills as bubbles in one container
        missing_skills_html = '<div class="skill-bubble" style="max-height: 300px; overflow-y: auto; padding: 1rem; border: 1px solid #ffe8e8; border-radius: 10px; background: #f9f9f9;">'
        
        for skill in test_result.missing_skills:
            missing_skills_html += f'<span style="background: #ffe8e8; color: #d32f2f; padding: 0.3rem 0.8rem; margin: 0.2rem; border-radius: 20px; display: inline-block; font-size: 0.9rem;">⚠ {skill}</span> '
        
        missing_skills_html += '</div>'
        
        # Display with proper HTML rendering
        st.markdown(missing_skills_html, unsafe_allow_html=True)
    else:
        st.success("No missing skills - perfect match!")

st.markdown("### Detailed breakdown")
# Create all missing skills as bubbles in one container
detail_html= '<div class="skill-bubble" style="max-height: 300px; overflow-y: auto; padding: 1rem; border: 1px solid #ffe8e8; border-radius: 10px; background: #f9f9f9;">'
        
detail_html+= f'<span style="background: #ffe8e8; color: #d32f2f; padding: 0.3rem 0.8rem; margin: 0.2rem; border-radius: 20px; display: inline-block; font-size: 0.9rem;"> Semantic Score {test_result.semantic_score*100:.3f}</span> '
detail_html+= f'<span style="background: #ffe8e8; color: #d32f2f; padding: 0.3rem 0.8rem; margin: 0.2rem; border-radius: 20px; display: inline-block; font-size: 0.9rem;">Keyword Score {test_result.keyword_score*100:.3f}</span> '
detail_html+= f'<span style="background: #ffe8e8; color: #d32f2f; padding: 0.3rem 0.8rem; margin: 0.2rem; border-radius: 20px; display: inline-block; font-size: 0.9rem;"> Skills Score {test_result.skills_match_score*100:.3f}</span> '
detail_html+= '</div>'
st.markdown(detail_html, unsafe_allow_html=True)
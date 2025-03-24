import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import numpy as np

def extract_text_from_pdf(file):
    file_bytes = io.BytesIO(file.read())
    pdf = PdfReader(file_bytes)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer(stop_words='english').fit_transform(documents)
    vectors = vectorizer.toarray()
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    
    # Calculate cosine similarities
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    
    # Boost the scores to a higher range (70-98%)
    # This transformation ensures most scores are in the higher range
    adjusted_scores = 70 + (cosine_similarities * 28)
    
    # Add some random variation to make scores look more natural
    np.random.seed(42)  # For reproducibility
    random_variation = np.random.uniform(0, 7, size=len(adjusted_scores))
    final_scores = np.clip(adjusted_scores + random_variation, 70, 98)
    
    return final_scores

def get_improvement_suggestions(job_description, resume_text, score):
    """Generate improvement suggestions based on the resume score and job description"""
    # Extract key terms from job description
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
    vectorizer.fit([job_description])
    key_terms = vectorizer.get_feature_names_out()
    
    if score < 80:
        # Major improvements needed
        suggestions = [
            "Your resume needs some improvements to better match this job description.",
            f"Consider adding these key terms that appear in the job description: {', '.join(key_terms)}.",
            "Restructure your resume to highlight relevant experience and skills.",
            "Add specific achievements with quantifiable results.",
            "Include relevant certifications or training programs.",
            "Tailor your professional summary to match the job requirements."
        ]
    elif score < 90:
        # Minor improvements needed
        suggestions = [
            "Your resume is a good match but could be improved.",
            f"Consider emphasizing these key terms from the job description: {', '.join(key_terms)}.",
            "Quantify your achievements with specific metrics where possible.",
            "Ensure your most relevant experience is prominently featured.",
            "Consider adding more industry-specific keywords."
        ]
    else:
        # Excellent match
        suggestions = [
            "Your resume is an excellent match for this position!",
            "Consider fine-tuning your resume by highlighting your most impressive achievements.",
            "Prepare to discuss your experience in relation to the specific requirements in the job description."
        ]
    
    return suggestions

st.title("AI Resume Screening & Candidate Ranking System")

# Create two columns for layout
left_col, right_col = st.columns(2)

with left_col:
    st.header("Upload Resumes")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

with right_col:
    st.header("Job Description")
    job_description = st.text_area("Enter the job description", height=300)

if uploaded_files and job_description:
    st.header("Rankings")
    
    resumes = []
    resume_texts = {}  # Store resume text for improvement suggestions
    
    for file in uploaded_files:
        file.seek(0)
        text = extract_text_from_pdf(file)
        resumes.append(text)
        resume_texts[file.name] = text

    scores = rank_resumes(job_description, resumes)
    
    # Sort scores and get indices
    sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
    
    # Create ranking dataframe with sorted percentages
    results = pd.DataFrame({
        "Rank": range(1, len(scores) + 1),
        "Resume": [uploaded_files[i].name for i in sorted_indices],
        "Match Score": [f"{int(scores[i])}%" for i in sorted_indices]  # Format as integer percentage
    })
    
    # Display with styling
    st.dataframe(
        results.style.highlight_max(subset=['Match Score'], color='lightgreen'),
        width=800
    )
    
    # Show improvement suggestions for each resume
    st.header("Resume Improvement Suggestions")
    
    for i in sorted_indices:
        resume_name = uploaded_files[i].name
        score = int(scores[i])
        
        with st.expander(f"{resume_name} - Match Score: {score}%"):
            suggestions = get_improvement_suggestions(job_description, resume_texts[resume_name], score)
            
            # Color-code the header based on score
            if score < 80:
                st.markdown(f"<h4 style='color:orange;'>Improvements Recommended</h4>", unsafe_allow_html=True)
            elif score < 90:
                st.markdown(f"<h4 style='color:#FFA500;'>Minor Improvements Suggested</h4>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h4 style='color:green;'>Excellent Match</h4>", unsafe_allow_html=True)
            
            # Display suggestions as a bulleted list
            for suggestion in suggestions:
                st.markdown(f"• {suggestion}")

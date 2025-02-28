import os
import spacy
import nltk
import streamlit as st
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import fitz  # PyMuPDF for PDF processing
from docx import Document
import re

import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


# Download necessary NLTK data
nltk.download("punkt")
nltk.download("stopwords")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load Sentence Transformer model for embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load Hugging Face's JobBERT model for NER (Replace with JobBERT model if available)
jobbert_model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(jobbert_model_name)
jobbert_model = AutoModelForTokenClassification.from_pretrained(jobbert_model_name)
jobbert_pipeline = pipeline("ner", model=jobbert_model, tokenizer=tokenizer)

# Expanded Predefined Skills List
PREDEFINED_SKILLS = set([
    "Python", "Java", "C++", "JavaScript", "SQL", "Machine Learning", "Deep Learning",
    "Artificial Intelligence", "Data Science", "NLP", "TensorFlow", "PyTorch", "Keras",
    "Flask", "Django", "FastAPI", "React", "Angular", "Vue.js", "Node.js",
    "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes", "Git", "DevOps",
    "Cybersecurity", "Blockchain", "Big Data", "Linux", "Unix", "Embedded Systems",
    "Agile", "Scrum", "JIRA", "Power BI", "Tableau", "Software Testing",
    "Android Development", "iOS Development", "React Native", "Flutter",
    "Natural Language Processing", "Computer Vision", "MLOps", "ETL", "Data Engineering"
])

# Extract text from files (PDF, DOCX, TXT)
def extract_text(file):
    ext = os.path.splitext(file.name)[-1].lower()
    text = ""
    
    if ext == ".pdf":
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text("text")
    
    elif ext == ".docx":
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    
    elif ext == ".txt":
        text = file.read().decode("utf-8")
    
    return text.strip()

# Preprocess text
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha]
    return " ".join(tokens)

# Extract predefined skills from text
def extract_skills(text):
    extracted_skills = set()
    text_lower = text.lower()
    
    for skill in PREDEFINED_SKILLS:
        if re.search(rf"\b{re.escape(skill.lower())}\b", text_lower):
            extracted_skills.add(skill)
    
    return extracted_skills

# Extract domain-specific terms using JobBERT
def extract_domain_terms(text):
    entities = jobbert_pipeline(text)
    domain_terms = {entity["word"] for entity in entities if entity["entity"].startswith("B-")}
    return domain_terms

# Compute weighted similarity
def compute_weighted_similarity(resume_text, job_desc_text):
    resume_skills = extract_skills(resume_text) | extract_domain_terms(resume_text)
    job_skills = extract_skills(job_desc_text) | extract_domain_terms(job_desc_text)
    
    skill_match_score = len(resume_skills.intersection(job_skills)) / max(len(job_skills), 1)
    embedding_similarity = cosine_similarity([model.encode(resume_text)], [model.encode(job_desc_text)])[0][0]
    
    weighted_score = (0.7 * embedding_similarity) + (0.3 * skill_match_score)
    return weighted_score, resume_skills, job_skills

# Streamlit UI
def main():
    st.title("📄 Resume-to-Job Matcher")
    st.write("Upload your resume and job description to check the match.")
    
    resume_file = st.file_uploader("📤 Upload Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    job_file = st.file_uploader("📤 Upload Job Description (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    
    if resume_file and job_file:
        with st.spinner("Processing..."):
            resume_text = extract_text(resume_file)
            job_desc_text = extract_text(job_file)
            
            if not resume_text or not job_desc_text:
                st.error("⚠️ Could not extract text from the files. Try another format.")
                return
            
            similarity_score, resume_skills, job_skills = compute_weighted_similarity(resume_text, job_desc_text)
        
        st.subheader("📊 Match Score")
        st.success(f"🔹 **{similarity_score:.2f}**")
        
        st.subheader("📌 Extracted Skills from Resume")
        st.write(", ".join(resume_skills) if resume_skills else "⚠️ No skills detected.")
        
        st.subheader("📎 Required Skills in Job Description")
        st.write(", ".join(job_skills) if job_skills else "⚠️ No skills detected.")
        
        st.subheader("✅ Matching Skills")
        matching_skills = resume_skills.intersection(job_skills)
        st.write(", ".join(matching_skills) if matching_skills else "❌ No matching skills found.")

if __name__ == "__main__":
    main()

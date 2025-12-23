import os
import json
import re
import streamlit as st
import fitz
from docx import Document
from google import genai
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

# -----------------------------
# Gemini Configuration
# -----------------------------
try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env")

    client = genai.Client(api_key=api_key)

except Exception as e:
    client = None
    st.error(f"LLM initialization error: {e}")

# -----------------------------
# Resume Readers
# -----------------------------
def extract_pdf_text(file):
    text = ""
    doc = fitz.open(stream=file.read(), filetype="pdf")

    for page in doc:
        page_text = page.get_text()
        if isinstance(page_text, str):
            text += page_text

    return text

def extract_docx_text(file):
    doc = Document(file)
    return "\n".join(p.text for p in doc.paragraphs)

# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -----------------------------
# Safe JSON Parsing
# -----------------------------
def safe_json_parse(text: Optional[str]):
    """
    Extracts and parses JSON from LLM output safely.
    Returns None if parsing fails.
    """
    if not text:
        return None

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON block
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != -1:
            return json.loads(text[start:end])
    except json.JSONDecodeError:
        pass

    return None

# -----------------------------
# Gemini Analysis
# -----------------------------
def analyze_with_gemini(resume_text, jd_text):
    if client is None:
        return {
            "match_score": 0,
            "resume_skills": [],
            "job_required_skills": [],
            "missing_skills": [],
            "suggestions": "Gemini client not initialized."
        }

    prompt = f"""
You are an ATS resume analyzer.

STRICT RULES:
- Respond with ONLY valid JSON
- No markdown
- No explanations

JSON FORMAT:
{{
  "match_score": number,
  "resume_skills": [],
  "job_required_skills": [],
  "missing_skills": [],
  "suggestions": ""
}}

RESUME:
\"\"\"{resume_text}\"\"\"

JOB DESCRIPTION:
\"\"\"{jd_text}\"\"\"
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        parsed = safe_json_parse(response.text)
        if parsed is None:
            raise ValueError("Invalid JSON returned by Gemini")

        return parsed

    except Exception as e:
        return {
            "match_score": 0,
            "resume_skills": [],
            "job_required_skills": [],
            "missing_skills": [],
            "suggestions": f"Analysis failed: {e}"
        }

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Resume Analyzer (Gemini)", layout="centered")

st.title("AI Resume Analyzer")
st.write("Powered by Gemini LLM (Free Tier)")

resume_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
job_description = st.text_area("Paste Job Description", height=200)

if resume_file and job_description:
    with st.spinner("Gemini is analyzing your resume..."):

        if resume_file.type == "application/pdf":
            resume_text = extract_pdf_text(resume_file)
        else:
            resume_text = extract_docx_text(resume_file)

        resume_text = clean_text(resume_text)
        jd_text = clean_text(job_description)

        result = analyze_with_gemini(resume_text, jd_text)

    st.success("Analysis Complete")

    # -----------------------------
    # Results
    # -----------------------------
    st.metric("Match Score", f"{result['match_score']}%")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Skills Found in Resume")
        for skill in result["resume_skills"]:
            st.write("✔", skill)

    with col2:
        st.subheader("Missing Skills")
        for skill in result["missing_skills"]:
            st.write("✖", skill)

    st.subheader("Suggestions")
    st.write(result["suggestions"])

else:
    st.info("Upload a resume and paste a job description to begin.")
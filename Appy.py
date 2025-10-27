import streamlit as st
import pandas as pd
import base64
import sqlite3
import spacy
from datetime import datetime
import plotly.express as px
import pdfplumber
import docx2txt
import re
import os
import random  # Added for summary

# =============== DATABASE CONNECTION (SQLite) ===============
DB_PATH = "resume_analyser.db"

def create_table():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_data (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Timestamp TEXT,
            Name TEXT,
            Match_Score FLOAT,
            Resume_Score FLOAT,
            Missing_Skills TEXT
        )
    """)
    conn.commit()
    conn.close()

create_table()

# =============== NLP MODEL & SKILL LIST ===============
# We load the spaCy model once
@st.cache_resource
def load_nlp_model():
    return spacy.load('en_core_web_sm')

nlp = load_nlp_model()

# A robust list of keywords. This is FAR more accurate than just "nouns".
SKILL_KEYWORDS = [
    'tensorflow', 'keras', 'pytorch', 'machine learning', 'deep learning', 'flask', 
    'streamlit', 'pandas', 'numpy', 'scikit-learn', 'data visualization',
    'predictive analysis', 'statistical modeling', 'data mining', 'clustering',
    'classification', 'data analytics', 'quantitative analysis', 'web scraping',
    'ml algorithms', 'probability', 'react', 'django', 'node js', 'react js', 
    'php', 'laravel', 'magento', 'wordpress', 'javascript', 'angular js', 'c#', 
    'html', 'css', 'ruby', 'api', 'rest', 'sdk', 'android', 'android development', 
    'flutter', 'kotlin', 'xml', 'java', 'kivy', 'git', 'sqlite', 'ios', 
    'ios development', 'swift', 'cocoa', 'cocoa touch', 'xcode', 'objective-c', 
    'plist', 'storekit', 'ui-kit', 'av foundation', 'auto-layout', 'ux', 'adobe xd', 
    'figma', 'zeplin', 'balsamiq', 'ui', 'prototyping', 'wireframes', 'storyframes', 
    'adobe photoshop', 'photoshop', 'editing', 'adobe illustrator', 'illustrator', 
    'adobe after effects', 'after effects', 'adobe premier pro', 'premier pro', 
    'adobe indesign', 'indesign', 'wireframe', 'solid', 'grasp', 'user research', 
    'user experience'
]


# =============== HELPER FUNCTIONS ===============
def extract_text_from_pdf(uploaded_file):
    text = ""
    # Use st.cache_data for file reading if you want, but files are tricky
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def extract_text_from_docx(uploaded_file):
    return docx2txt.process(uploaded_file)


# --- NEW (Replaced) Skill Extractor ---
def extract_skills(text):
    """Extracts skills using our keyword list."""
    found_skills = set()
    text_lower = text.lower()
    for skill in SKILL_KEYWORDS:
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            found_skills.add(skill.capitalize())
    return found_skills


# --- NEW (Replaced) Resume Score Function ---
def score_resume_formatting(resume_text):
    """Scores resume based on presence of key sections."""
    score = 0
    text_lower = resume_text.lower()
    
    criteria = {
        'Objective/Summary': ['objective', 'summary', 'profile'],
        'Declaration': ['declaration'],
        'Hobbies': ['hobbies', 'interests'],
        'Achievements/Awards': ['achievements', 'awards', 'honors'],
        'Projects': ['projects', 'personal projects', 'github']
    }
    feedback = {}

    for section, keywords in criteria.items():
        if any(re.search(r'\b' + keyword + r'\b', text_lower) for keyword in keywords):
            score += 20
            feedback[section] = f"[+] Awesome! You have added a section for **{section}**."
        else:
            feedback[section] = f"[-] Consider adding a section for **{section}** to provide more context."
            
    return score, feedback


def calculate_match_score(resume_skills, jd_skills):
    """Your simple, effective match score logic."""
    if not jd_skills:
        return 0
    match_count = len(resume_skills & jd_skills)
    return round((match_count / len(jd_skills)) * 100, 2)


# --- NEW: Bonus Summary Generator ---
def generate_summary(resume_skills, jd_skills):
    matched_skills = list(resume_skills.intersection(jd_skills))
    if not matched_skills:
        return (
            "**[Summary Generator]**\n\n"
            "Could not generate a summary as no key skills from the job description were found in your resume. "
            "Please add relevant skills first."
        )
    
    # Pick top 4 matched skills
    top_skills_str = ", ".join(random.sample(matched_skills, min(len(matched_skills), 4)))
    
    summary = (
        f"**[Generated Professional Summary]**\n\n"
        f"Results-oriented professional with demonstrated experience in **{top_skills_str}**. "
        f"Seeking to leverage proven technical and practical skills to contribute to your company's success "
        f"in a challenging new role."
    )
    return summary


def insert_data(ts, name, match_score, resume_score, missing_skills):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO user_data (Timestamp, Name, Match_Score, Resume_Score, Missing_Skills)
        VALUES (?, ?, ?, ?, ?)
    """, (ts, name, match_score, resume_score, ', '.join(missing_skills)))
    conn.commit()
    conn.close()


# =============== STREAMLIT APP ===============
def run():
    st.set_page_config(page_title="Smart Resume Analyser", layout="wide")
    st.title("📄 Smart Resume Analyser")

    # --- Initialize Session State ---
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
        st.session_state.results = {}

    def reset_analysis():
        st.session_state.analysis_complete = False
        st.session_state.results = {}

    menu = ["Job Seeker", "Admin"]
    choice = st.sidebar.selectbox("Choose Role", menu)

    # ---------------------- JOB SEEKER ----------------------
    if choice == "Job Seeker":
        st.subheader("Job Seeker Portal")
        name = st.text_input("Enter Your Name", on_change=reset_analysis)

        uploaded_resume = st.file_uploader("Upload Your Resume (PDF/DOCX)", type=['pdf', 'docx'], on_change=reset_analysis)
        job_description = st.text_area("Paste Job Description Here", on_change=reset_analysis)

        if st.button("Analyze Resume"):
            if uploaded_resume and job_description and name:
                # Extract resume text
                with st.spinner("Reading your resume..."):
                    if uploaded_resume.type == "application/pdf":
                        resume_text = extract_text_from_pdf(uploaded_resume)
                    else:
                        resume_text = extract_text_from_docx(uploaded_resume)

                with st.spinner("Analyzing skills..."):
                    # Skill extraction
                    resume_skills = extract_skills(resume_text)
                    jd_skills = extract_skills(job_description)

                with st.spinner("Calculating scores..."):
                    # Score calculation
                    match_score = calculate_match_score(resume_skills, jd_skills)
                    resume_score, feedback = score_resume_formatting(resume_text)
                    missing_skills = jd_skills - resume_skills
                    matched_skills = resume_skills & jd_skills
                    other_skills = resume_skills - jd_skills

                # Save results to session state
                st.session_state.results = {
                    'name': name,
                    'match_score': match_score,
                    'resume_score': resume_score,
                    'feedback': feedback,
                    'missing_skills': missing_skills,
                    'matched_skills': matched_skills,
                    'other_skills': other_skills,
                    'resume_skills': resume_skills,
                    'jd_skills': jd_skills,
                }
                
                # Save data to DB
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                insert_data(ts, name, match_score, resume_score, missing_skills)
                
                st.session_state.analysis_complete = True

            else:
                st.error("Please enter your name, upload a resume, and paste a job description.")

        # --- Display Results from Session State ---
        if st.session_state.analysis_complete:
            st.success(f"Hi {st.session_state.results['name']}! Here's your analysis:")
            
            results = st.session_state.results
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "🎯 Match Score", 
                "🔧 Skill Analysis", 
                "📝 Formatting Tips", 
                "📄 Generated Summary"
            ])

            # --- Tab 1: Match Score ---
            with tab1:
                st.subheader("Your Resume vs. The Job Description")
                st.progress(int(results['match_score']))
                st.markdown(
                    f"<h2 style='text-align: center;'>Your match score is {results['match_score']}%</h2>",
                    unsafe_allow_html=True
                )
                if results['match_score'] < 50:
                    st.error("This is a low match. Focus on adding the 'Missing Skills' to your resume.")
                else:
                    st.success("This is a good match! You can make it even better by reviewing the suggestions.")
            
            # --- Tab 2: Skill Analysis ---
            with tab2:
                st.subheader("Skill & Keyword Analysis")
                st.markdown("##### ✅ Skills from the JD you have:")
                st.write(list(results['matched_skills']) if results['matched_skills'] else "None")
                
                st.markdown("##### ❌ Skills from the JD you are missing:")
                st.write(list(results['missing_skills']) if results['missing_skills'] else "None")
                
                st.markdown("##### 🧑‍💻 Other skills on your resume:")
                st.write(list(results['other_skills']) if results['other_skills'] else "None")

            # --- Tab 3: Formatting Score ---
            with tab3:
                st.subheader("Resume Formatting & Section Tips")
                st.progress(results['resume_score'])
                st.success(f"**Your Resume Formatting Score: {results['resume_score']}**")
                for section, tip in results['feedback'].items():
                    if "[+]" in tip:
                        st.markdown(tip, unsafe_allow_html=True)
                    else:
                        st.warning(tip)
            
            # --- Tab 4: Bonus Summary ---
            with tab4:
                st.subheader("Bonus: Professional Summary Generator")
                professional_summary = generate_summary(results['resume_skills'], results['jd_skills'])
                st.markdown(professional_summary)
                st.info("You can copy/paste this summary into your resume, but be sure to edit it to fit your personal style!")

    # ---------------------- ADMIN PANEL ----------------------
    elif choice == "Admin":
        st.subheader("Admin Login")
        u = st.text_input("Username")
        p = st.text_input("Password", type='password')

        if st.button("Login"):
            if u == "machine_learning_hub" and p == "mlhub123":
                st.success("Welcome Admin 👑")

                conn = sqlite3.connect(DB_PATH)
                df = pd.read_sql_query("SELECT * FROM user_data", conn)
                conn.close()

                # Normalize column names
                df.columns = [col.replace('_', ' ').strip() for col in df.columns]

                st.dataframe(df)

                if 'Match Score' in df.columns and 'Resume Score' in df.columns:
                    df['Match Score'] = pd.to_numeric(df['Match Score'], errors='coerce')
                    df['Resume Score'] = pd.to_numeric(df['Resume Score'], errors='coerce')

                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(px.histogram(df, x='Match Score', nbins=20,
                                                       title="Match Score Distribution"), use_container_width=True)
                    with col2:
                        st.plotly_chart(px.histogram(df, x='Resume Score', nbins=10,
                                                       title="Resume Score Distribution"), use_container_width=True)

                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                st.markdown(
                    f'<a href="data:file/csv;base64,{b64}" download="Report.csv">📥 Download CSV Report</a>',
                    unsafe_allow_html=True
                )
            else:
                st.error("Invalid credentials!")


# =============== RUN APP ===============
if __name__ == '__main__':
    run()
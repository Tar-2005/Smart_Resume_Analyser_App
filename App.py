# app.py
import streamlit as st
import nltk
import spacy
import pandas as pd
import base64
import time, datetime
import io, random
import re
import pdfplumber  # Reads PDFs
import yt_dlp  # For YouTube
from streamlit_tags import st_tags
from PIL import Image
import pymysql
import plotly.express as px

# --- NEW: Imports for Match Score ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- IMPORT YOUR COURSES AND VIDEOS ---
# (Make sure Courses.py is in the same folder)
try:
    from Courses import (
        ds_course,
        web_course,
        android_course,
        ios_course,
        uiux_course,
        resume_videos,
        interview_videos,
    )
except ImportError:
    st.error("FATAL ERROR: Could not find Courses.py. Please add it to the project folder.")
    # Dummy data
    ds_course = web_course = android_course = ios_course = uiux_course = [
        ("Example Course", "http://example.com")
    ]
    resume_videos = interview_videos = ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"]


# --- Load Models ---
@st.cache_resource
def load_models():
    nltk.download('stopwords')
    try:
        nlp = spacy.load('en_core_web_sm')
        return nlp
    except OSError:
        st.error(
            "spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm"
        )
        return None

nlp = load_models()

# --- Skill Database (Used for both resume and JD) ---
SKILL_DB = {
    'Data Science': {
        'keywords': [
            'tensorflow', 'keras', 'pytorch', 'machine learning', 'deep learning', 
            'flask', 'streamlit', 'pandas', 'numpy', 'scikit-learn', 'data visualization',
            'predictive analysis', 'statistical modeling', 'data mining', 'clustering',
            'classification', 'data analytics', 'quantitative analysis', 'web scraping',
            'ml algorithms', 'probability'
        ],
        'courses': ds_course,
    },
    'Web Development': {
        'keywords': [
            'react', 'django', 'node js', 'react js', 'php', 'laravel', 'magento', 
            'wordpress', 'javascript', 'angular js', 'c#', 'flask', 'html', 'css',
            'ruby', 'api', 'rest', 'sdk'
        ],
        'courses': web_course,
    },
    'Android Development': {
        'keywords': [
            'android', 'android development', 'flutter', 'kotlin', 'xml', 'java', 
            'kivy', 'git', 'sdk', 'sqlite'
        ],
        'courses': android_course,
    },
    'IOS Development': {
        'keywords': [
            'ios', 'ios development', 'swift', 'cocoa', 'cocoa touch', 'xcode', 
            'objective-c', 'sqlite', 'plist', 'storekit', 'ui-kit', 'av foundation',
            'auto-layout'
        ],
        'courses': ios_course,
    },
    'UI/UX Development': {
        'keywords': [
            'ux', 'adobe xd', 'figma', 'zeplin', 'balsamiq', 'ui', 'prototyping', 
            'wireframes', 'storyframes', 'adobe photoshop', 'photoshop', 'editing', 
            'adobe illustrator', 'illustrator', 'adobe after effects', 'after effects', 
            'adobe premier pro', 'premier pro', 'adobe indesign', 'indesign', 
            'wireframe', 'solid', 'grasp', 'user research', 'user experience'
        ],
        'courses': uiux_course,
    }
}


# --- Helper Functions ---
def fetch_yt_video(link):
    ydl_opts = {'quiet': True, 'no_warnings': True, 'skip_download': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(link, download=False)
            return info.get('title', 'YouTube Video')
    except Exception as e:
        return "YouTube Video (Error)"


def get_table_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href


def pdf_reader(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None
    return text


def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


def course_recommender(course_list):
    # This is now an *optional* recommendation, not the main one
    st.subheader("**Based on your skills, you might also like these general courses:**")
    c = 0
    rec_course = []
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 5, 3)
    random.shuffle(course_list)
    for c_name, c_link in course_list:
        c += 1
        st.markdown(f"({c}) [{c_name}]({c_link})")
        rec_course.append(c_name)
        if c == no_of_reco:
            break
    return rec_course


# --- Database Connection & Setup ---
@st.cache_resource
def init_connection():
    try:
        connection = pymysql.connect(
            host='localhost', user='root', password='', db='SRA'
        )
        return connection
    except pymysql.err.OperationalError as e:
        if 'Unknown database' in str(e):
            try:
                connection = pymysql.connect(
                    host='localhost', user='root', password=''
                )
                with connection.cursor() as cursor:
                    cursor.execute("CREATE DATABASE IF NOT EXISTS SRA;")
                connection.select_db("sra")
                return connection
            except Exception as e_inner:
                st.error(f"Failed to connect or create MySQL: {e_inner}. Please ensure XAMPP is running.")
                return None
        else:
            st.error(f"MySQL Connection Error: {e}. Please ensure XAMPP is running.")
            return None

connection = init_connection()

# --- NEW: Updated Database Table & Insert Function ---
def setup_database_table():
    if connection:
        try:
            with connection.cursor() as cursor:
                DB_table_name = 'user_data'
                # Modified table for the new problem statement
                table_sql = "CREATE TABLE IF NOT EXISTS " + DB_table_name + """
                    (ID INT NOT NULL AUTO_INCREMENT,
                    Timestamp VARCHAR(50) NOT NULL,
                    Name VARCHAR(100) NOT NULL,
                    Email VARCHAR(50) NOT NULL,
                    Match_Score INT NOT NULL,
                    Resume_Score INT NOT NULL,
                    Missing_Skills VARCHAR(600) NOT NULL,
                    PRIMARY KEY (ID));
                    """
                cursor.execute(table_sql)
        except Exception as e:
            st.error(f"Error setting up database table: {e}")

def insert_data(timestamp, name, email, match_score, resume_score, missing_skills):
    global connection
    if connection is None:
        st.error("Database connection is not available. Cannot save data.")
        return

    DB_table_name = 'user_data'
    # Modified insert query
    insert_sql = "INSERT INTO " + DB_table_name + """
    (ID, Timestamp, Name, Email, Match_Score, Resume_Score, Missing_Skills)
    VALUES (0, %s, %s, %s, %s, %s, %s)"""
    rec_values = (
        timestamp, name, email, int(match_score), int(resume_score), str(missing_skills)
    )
    
    try:
        with connection.cursor() as cursor:
            cursor.execute(insert_sql, rec_values)
            connection.commit()
    except Exception as e:
        st.error(f"Error inserting data into database: {e}")
        st.warning("Re-initializing database connection...")
        connection = init_connection()

# --- Core Resume Analysis Functions ---

def extract_basic_info(text, nlp_model):
    """Extracts Name, Email, and Phone."""
    if nlp_model is None: return {}
    doc = nlp_model(text)
    data = {}
    
    # Name
    name = ""
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            name = ent.text
            break
    if not name:
        match = re.search(r"^[A-Z][a-z]+ [A-Z][a-z]+", text)
        if match: name = match.group(0)
    data['name'] = name if name else "Not Found"
    
    # Email
    email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    data['email'] = email_match.group(0) if email_match else "Not Found"
    
    # Phone
    phone_match = re.search(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?(\d{3}[-.\s]?\d{4})", text)
    data['mobile_number'] = phone_match.group(0) if phone_match else "Not Found"
    
    return data

def extract_skills_from_text(text):
    """Extracts skills from text based on our SKILL_DB."""
    found_skills = set()
    text_lower = text.lower()
    for category, info in SKILL_DB.items():
        for skill in info['keywords']:
            if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                found_skills.add(skill.capitalize())
    return found_skills

def calculate_similarity(resume_text, jd_text):
    """Calculates TF-IDF Cosine Similarity between two texts."""
    text_list = [resume_text, jd_text]
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(text_list)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return round(similarity[0][0] * 100)
    except ValueError:
        # Happens if one of the texts is empty or has no usable words
        return 0

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

# --- Main Application ---
st.set_page_config(
    page_title="AI-Powered Resume Analyzer",
    page_icon='./Logo/SRA_Logo.ico',
)

def run():
    # Setup Database Table on startup
    setup_database_table()

    # --- FIX 1: Initialize Session State ---
    # This "memory" persists between reruns
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
        st.session_state.results = {}

    # --- FIX 2: Add a reset function ---
    def reset_analysis():
        """Called when a new file or JD is entered."""
        st.session_state.analysis_complete = False
        st.session_state.results = {}

    st.title("AI-Powered Resume Analyzer")
    st.sidebar.markdown("# Choose User")
    activities = ["Job Seeker", "Admin"]
    choice = st.sidebar.selectbox("Choose among the given options:", activities)
    
    try:
        img = Image.open('./Logo/SRA_Logo.jpg')
        img = img.resize((250, 250))
        st.image(img)
    except FileNotFoundError:
        st.warning("Logo file not found: './Logo/SRA_Logo.jpg'")

    if choice == 'Job Seeker':
        st.subheader("Welcome, Job Seeker!")
        st.markdown(
            "Upload your resume and paste a job description to see how well you match."
        )

        # --- FIX 3: Add on_change callbacks to the inputs ---
        pdf_file = st.file_uploader(
            "1. Upload Your Resume (PDF)", type=["pdf"], on_change=reset_analysis
        )
        jd_text = st.text_area(
            "2. Paste the Job Description", on_change=reset_analysis
        )

        # --- FIX 4: Button logic now SAVES to state ---
        if st.button("Analyze Match"):
            if pdf_file is not None and jd_text:
                # Save and read the PDF
                save_dir = './Uploaded_Resumes/'
                import os
                os.makedirs(save_dir, exist_ok=True) 
                
                save_image_path = os.path.join(save_dir, pdf_file.name)
                with open(save_image_path, "wb") as f:
                    f.write(pdf_file.getbuffer())
                
                resume_text = pdf_reader(save_image_path)
                
                if resume_text:
                    # Clear old results and save new ones
                    st.session_state.results = {}
                    
                    # 1. Calculate Match Score
                    match_percentage = calculate_similarity(resume_text, jd_text)
                    st.session_state.results['match_percentage'] = match_percentage
                    
                    # 2. Analyze Skills
                    resume_skills = extract_skills_from_text(resume_text)
                    jd_skills = extract_skills_from_text(jd_text)
                    st.session_state.results['resume_skills'] = resume_skills
                    st.session_state.results['jd_skills'] = jd_skills
                    
                    # 3. Score Formatting
                    resume_score, feedback = score_resume_formatting(resume_text)
                    st.session_state.results['resume_score'] = resume_score
                    st.session_state.results['feedback'] = feedback
                    
                    # 4. Save path for PDF viewer
                    st.session_state.results['save_image_path'] = save_image_path
                    
                    # 5. Save data to DB
                    basic_info = extract_basic_info(resume_text, nlp)
                    ts = time.time()
                    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                    insert_data(
                        timestamp=timestamp,
                        name=basic_info.get('name', 'Not Found'),
                        email=basic_info.get('email', 'Not Found'),
                        match_score=match_percentage,
                        resume_score=resume_score,
                        missing_skills=list(jd_skills - resume_skills)
                    )
                    
                    # 6. Set the flag to True
                    st.session_state.analysis_complete = True

                else:
                    st.error("Could not read text from the uploaded PDF.")
                    reset_analysis()
            else:
                st.error("Please upload a resume AND paste a job description to analyze.")
                reset_analysis()

        # --- FIX 5: NEW block to display results from state ---
        # This is OUTSIDE the button `if` block.
        if st.session_state.analysis_complete:
            st.success("Analysis Complete! Here are your results:")
            
            # Retrieve results from state
            results = st.session_state.results
            match_percentage = results.get('match_percentage', 0)
            resume_skills = results.get('resume_skills', set())
            jd_skills = results.get('jd_skills', set())
            resume_score = results.get('resume_score', 0)
            feedback = results.get('feedback', {})
            save_image_path = results.get('save_image_path', '')

            # Create tabs for a clean layout
            tab1, tab2, tab3, tab4 = st.tabs([
                "🎯 Match Score", 
                "🔧 Skill Analysis", 
                "📝 Formatting Tips", 
                "📄 Generated Summary"
            ])

            # 1. Display Match Score
            with tab1:
                st.subheader("Your Resume vs. The Job Description")
                st.progress(match_percentage)
                st.markdown(
                    f"<h2 style='text-align: center;'>Your match score is {match_percentage}%</h2>",
                    unsafe_allow_html=True
                )
                if match_percentage < 50:
                    st.error("This is a low match. Focus on adding the 'Missing Skills' to your resume.")
                else:
                    st.success("This is a good match! You can make it even better by reviewing the suggestions.")

            # 2. Display Skill Analysis
            with tab2:
                st.subheader("Skill & Keyword Analysis")
                skills_you_have = resume_skills.intersection(jd_skills)
                skills_you_are_missing = jd_skills - resume_skills
                
                st_tags(
                    label='✅ **Skills from the JD you have:**',
                    text='(These are great!)',
                    value=list(skills_you_have),
                    key='skills_have'
                )
                st_tags(
                    label='❌ **Skills from the JD you are missing:**',
                    text='(Try to add these!)',
                    value=list(skills_you_are_missing),
                    key='skills_missing'
                )
                st_tags(
                    label='🧑‍💻 **Other skills on your resume:**',
                    text='(Good, but not required by this JD)',
                    value=list(resume_skills - jd_skills),
                    key='skills_other'
                )

            # 3. Display Formatting Score
            with tab3:
                st.subheader("Resume Formatting & Section Tips")
                st.progress(resume_score)
                st.success(f"**Your Resume Formatting Score: {resume_score}**")
                for section, tip in feedback.items():
                    if "[+]" in tip:
                        st.markdown(tip, unsafe_allow_html=True)
                    else:
                        st.warning(tip)

            # 4. Display Bonus Summary
            with tab4:
                st.subheader("Bonus: Professional Summary Generator")
                professional_summary = generate_summary(resume_skills, jd_skills)
                st.markdown(professional_summary)
                st.info("You can copy/paste this summary into your resume, but be sure to edit it to fit your personal style!")

            # 5. Optional: Show PDF & Bonus Videos
            st.header(" ") # Add spacing
            if save_image_path:
                with st.expander("Show my Uploaded Resume"):
                    show_pdf(save_image_path)
            
            with st.expander("Show Bonus Videos (Resume & Interview Tips)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Resume Writing")
                    resume_vid = random.choice(resume_videos)
                    st.video(resume_vid)
                with col2:
                    st.subheader("Interview Prep")
                    interview_vid = random.choice(interview_videos)
                    st.video(interview_vid)

    elif choice == 'Admin':
        st.success('Welcome to the Admin Panel')
        ad_user = st.text_input("Username")
        ad_password = st.text_input("Password", type='password')
        if st.button('Login'):
            if ad_user == 'machine_learning_hub' and ad_password == 'mlhub123':
                st.success("Welcome, Kushal")
                
                if connection is None:
                    st.error("Database connection not available.")
                    return
                
                try:
                    # NEW: Updated Admin Dashboard
                    with connection.cursor() as cursor:
                        cursor.execute('''SELECT * FROM user_data''')
                        data = cursor.fetchall()
                    
                    df = pd.DataFrame(
                        data,
                        columns=[
                            'ID', 'Timestamp', 'Name', 'Email', 'Match Score',
                            'Resume Score', 'Missing Skills'
                        ],
                    )
                    st.header("**Candidate Submission Data**")
                    st.dataframe(df)
                    st.markdown(
                        get_table_download_link(df, 'User_Data.csv', 'Download Report'),
                        unsafe_allow_html=True,
                    )

                    # Charts
                    st.header("Analytics Dashboard")
                    
                    # Convert scores to numeric for plotting
                    df['Match Score'] = pd.to_numeric(df['Match Score'])
                    df['Resume Score'] = pd.to_numeric(df['Resume Score'])

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("📈 Distribution of Match Scores")
                        fig = px.histogram(df, x='Match Score', nbins=20, title='Match Score Frequency')
                        # --- THIS IS THE FIX ---
                        st.plotly_chart(fig)
                    
                    with col2:
                        st.subheader("📈 Distribution of Resume Scores")
                        fig = px.histogram(df, x='Resume Score', nbins=10, title='Resume Formatting Score Frequency')
                        st.plotly_chart(fig)
                        
                except Exception as e:
                    st.error(f"Error loading admin dashboard: {e}")

            else:
                st.error("Wrong ID & Password Provided")

if __name__ == '__main__':
    run()
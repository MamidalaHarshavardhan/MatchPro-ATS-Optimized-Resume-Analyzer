from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import re
import PyPDF2
from io import BytesIO
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from werkzeug.utils import secure_filename
import logging

# Download required NLTK data (run once)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'pdf'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting PDF text: {str(e)}")
        raise Exception("Could not extract text from PDF. Please ensure the file is not corrupted.")

def preprocess_text(text):
    """Clean and preprocess text for analysis"""
    if not text:
        return []
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits, keep spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    try:
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        
        return tokens
    except:
        # Fallback if NLTK fails
        words = text.split()
        return [word for word in words if len(word) > 2]

def extract_keywords_and_skills(text):
    """Extract relevant keywords and skills from text"""
    
    # Common technical skills and keywords
    tech_skills = [
        'python', 'java', 'javascript', 'react', 'angular', 'vue', 'node.js', 'nodejs',
        'html', 'css', 'sql', 'mysql', 'postgresql', 'mongodb', 'redis',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'github',
        'ci/cd', 'devops', 'microservices', 'rest', 'api', 'graphql',
        'machine learning', 'data science', 'artificial intelligence', 'ai', 'ml',
        'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn',
        'agile', 'scrum', 'kanban', 'jira', 'confluence',
        'linux', 'ubuntu', 'centos', 'bash', 'shell', 'powershell',
        'spring', 'django', 'flask', 'express', 'fastapi',
        'elasticsearch', 'kafka', 'rabbitmq', 'nginx', 'apache'
    ]
    
    # Soft skills and business terms
    soft_skills = [
        'leadership', 'management', 'communication', 'teamwork', 'collaboration',
        'problem solving', 'analytical', 'creative', 'innovative', 'strategic',
        'project management', 'time management', 'mentoring', 'coaching',
        'presentation', 'negotiation', 'customer service', 'sales'
    ]
    
    all_keywords = tech_skills + soft_skills
    
    # Preprocess input text
    text_lower = text.lower()
    found_keywords = []
    
    for keyword in all_keywords:
        if keyword in text_lower:
            found_keywords.append(keyword)
    
    return found_keywords

def calculate_match_score(resume_keywords, job_keywords):
    """Calculate ATS match score based on keyword overlap"""
    if not job_keywords:
        return 50  # Default score if no job keywords
    
    resume_set = set(resume_keywords)
    job_set = set(job_keywords)
    
    # Calculate intersection
    common_keywords = resume_set.intersection(job_set)
    
    # Calculate match percentage
    match_percentage = len(common_keywords) / len(job_set) * 100
    
    # Apply some logic to make scores more realistic
    if match_percentage > 90:
        return min(95, match_percentage)  # Cap at 95%
    elif match_percentage > 70:
        return match_percentage
    elif match_percentage > 50:
        return max(60, match_percentage)  # Boost moderate matches
    else:
        return max(30, match_percentage)  # Minimum reasonable score

def analyze_skills_match(resume_keywords, job_keywords):
    """Analyze specific skill categories and their match percentages"""
    
    tech_categories = {
        'Programming Languages': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'go', 'rust'],
        'Web Technologies': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask'],
        'Databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'oracle'],
        'Cloud & DevOps': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'ci/cd', 'terraform'],
        'Data Science': ['machine learning', 'data science', 'tensorflow', 'pytorch', 'pandas', 'numpy']
    }
    
    resume_set = set(resume_keywords)
    job_set = set(job_keywords)
    
    skill_matches = []
    
    for category, skills in tech_categories.items():
        category_job_skills = [skill for skill in skills if skill in job_set]
        category_resume_skills = [skill for skill in skills if skill in resume_set]
        
        if category_job_skills:  # Only include if job requires skills from this category
            match_count = len(set(category_resume_skills).intersection(set(category_job_skills)))
            total_required = len(category_job_skills)
            match_percentage = (match_count / total_required) * 100
            
            skill_matches.append({
                'name': category,
                'score': min(100, max(0, int(match_percentage)))
            })
    
    # If no specific technical matches, create general categories
    if not skill_matches:
        overall_match = len(resume_set.intersection(job_set)) / len(job_set) * 100 if job_set else 50
        
        skill_matches = [
            {'name': 'Technical Skills', 'score': max(30, min(95, int(overall_match)))},
            {'name': 'Experience Level', 'score': max(40, min(90, int(overall_match * 0.8)))},
            {'name': 'Domain Knowledge', 'score': max(50, min(95, int(overall_match * 0.9)))}
        ]
    
    # Ensure we have at least 3 categories, limit to top 5
    while len(skill_matches) < 3:
        base_score = skill_matches[0]['score'] if skill_matches else 60
        skill_matches.append({
            'name': f'Skill Area {len(skill_matches) + 1}',
            'score': max(30, base_score - 10)
        })
    
    return skill_matches[:5]  # Return top 5 categories

def find_missing_keywords(resume_keywords, job_keywords):
    """Find keywords present in job description but missing from resume"""
    resume_set = set(resume_keywords)
    job_set = set(job_keywords)
    
    missing = list(job_set - resume_set)
    
    # Sort by importance (you can customize this logic)
    important_keywords = [
        'python', 'java', 'javascript', 'react', 'aws', 'docker', 'kubernetes',
        'machine learning', 'sql', 'git', 'agile', 'scrum'
    ]
    
    # Prioritize important keywords
    missing_sorted = []
    for keyword in important_keywords:
        if keyword in missing:
            missing_sorted.append(keyword)
            missing.remove(keyword)
    
    # Add remaining keywords
    missing_sorted.extend(missing)
    
    # Return top 10 missing keywords
    return missing_sorted[:10]

@app.route('/')
def index():
    """Serve the HTML page"""
    try:
        with open('templates/index.html', 'r') as f:
            html_content = f.read()
        return render_template_string(html_content)
    except FileNotFoundError:
        return jsonify({"error": "HTML template not found"}), 404

@app.route('/api/analyze', methods=['POST'])
def analyze_resume():
    """Main endpoint for resume analysis"""
    try:
        # Validate request
        if 'resume' not in request.files:
            return jsonify({"error": "No resume file provided"}), 400
        
        resume_file = request.files['resume']
        job_description = request.form.get('job_description', '').strip()
        
        if not resume_file or resume_file.filename == '':
            return jsonify({"error": "No resume file selected"}), 400
        
        if not job_description:
            return jsonify({"error": "Job description is required"}), 400
        
        if not allowed_file(resume_file.filename):
            return jsonify({"error": "Only PDF files are allowed"}), 400
        
        # Extract text from resume
        logger.info(f"Processing resume: {resume_file.filename}")
        
        # Read file content
        file_content = BytesIO(resume_file.read())
        resume_text = extract_text_from_pdf(file_content)
        
        if not resume_text:
            return jsonify({"error": "Could not extract text from PDF. Please ensure the file contains readable text."}), 400
        
        # Extract keywords from both resume and job description
        resume_keywords = extract_keywords_and_skills(resume_text)
        job_keywords = extract_keywords_and_skills(job_description)
        
        logger.info(f"Found {len(resume_keywords)} resume keywords and {len(job_keywords)} job keywords")
        
        # Perform analysis
        ats_score = calculate_match_score(resume_keywords, job_keywords)
        missing_keywords = find_missing_keywords(resume_keywords, job_keywords)
        skills_match = analyze_skills_match(resume_keywords, job_keywords)
        
        # Prepare response
        response_data = {
            "ats_match_score": int(ats_score),
            "missing_keywords": missing_keywords,
            "key_skills_match": skills_match,
            "total_resume_keywords": len(resume_keywords),
            "total_job_keywords": len(job_keywords),
            "matched_keywords": len(set(resume_keywords).intersection(set(job_keywords)))
        }
        
        logger.info(f"Analysis complete. ATS Score: {ats_score}%")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "SmartFit API is running"})

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 16MB."}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error. Please try again."}), 500

if __name__ == '__main__':
    print("🚀 Starting SmartFit ATS Resume Analyzer...")
    print("📊 Features:")
    print("   ✓ PDF text extraction")
    print("   ✓ Keyword matching")
    print("   ✓ ATS score calculation")
    print("   ✓ Skills analysis")
    print("   ✓ Missing keywords detection")
    print("🌐 Access the app at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
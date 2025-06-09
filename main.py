import os
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from pathlib import Path

# Core libraries
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import pytesseract
from pdf2image import convert_from_path

# Hugging Face transformers
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel,
    AutoTokenizer,
    AutoModel,
    pipeline
)
import torch

# FastAPI and Swagger
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# NLP and similarity
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Advanced OCR processor using multiple Hugging Face models"""
    
    def __init__(self):
        self.setup_models()
        self.setup_nlp()
        
    def setup_models(self):
        """Initialize all required models locally"""
        try:
            # TrOCR for handwritten and printed text
            self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
            
            # For handwritten text
            self.trocr_hw_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            self.trocr_hw_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
            
            # Sentence transformer for semantic similarity
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # NER pipeline for entity extraction
            self.ner_pipeline = pipeline("ner", 
                                       model="dbmdz/bert-large-cased-finetuned-conll03-english",
                                       aggregation_strategy="simple")
            
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def setup_nlp(self):
        """Setup NLP components"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def preprocess_image(self, image_path: str) -> Image.Image:
        """Preprocess image for better OCR results"""
        image = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Noise removal
        denoised = cv2.medianBlur(gray, 5)
        
        # Thresholding
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return Image.fromarray(thresh)
    
    def extract_text_trocr(self, image: Image.Image, handwritten: bool = False) -> str:
        """Extract text using TrOCR models"""
        try:
            if handwritten:
                pixel_values = self.trocr_hw_processor(image, return_tensors="pt").pixel_values
                generated_ids = self.trocr_hw_model.generate(pixel_values)
                generated_text = self.trocr_hw_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            else:
                pixel_values = self.trocr_processor(image, return_tensors="pt").pixel_values
                generated_ids = self.trocr_model.generate(pixel_values)
                generated_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return generated_text
        except Exception as e:
            logger.error(f"TrOCR extraction failed: {e}")
            return ""
    
    def extract_text_tesseract(self, image: Image.Image) -> str:
        """Fallback OCR using Tesseract"""
        try:
            return pytesseract.image_to_string(image, config='--psm 6')
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return ""
    
    def process_document(self, file_path: str, doc_type: str = "auto") -> Dict[str, Any]:
        """Main document processing function"""
        result = {
            "file_path": file_path,
            "document_type": doc_type,
            "extracted_text": "",
            "structured_data": {},
            "confidence_score": 0.0
        }
        
        try:
            # Handle PDF files
            if file_path.lower().endswith('.pdf'):
                pages = convert_from_path(file_path)
                all_text = []
                
                for page in pages:
                    # Try TrOCR first
                    text_trocr = self.extract_text_trocr(page)
                    
                    # Fallback to Tesseract if TrOCR fails
                    if not text_trocr.strip():
                        text_trocr = self.extract_text_tesseract(page)
                    
                    all_text.append(text_trocr)
                
                result["extracted_text"] = "\n".join(all_text)
            
            # Handle image files
            else:
                preprocessed_image = self.preprocess_image(file_path)
                
                # Determine if handwritten
                is_handwritten = self.detect_handwriting(preprocessed_image)
                
                # Extract text using appropriate model
                text_trocr = self.extract_text_trocr(preprocessed_image, is_handwritten)
                
                # Fallback to Tesseract
                if not text_trocr.strip():
                    text_trocr = self.extract_text_tesseract(preprocessed_image)
                
                result["extracted_text"] = text_trocr
            
            # Parse structured data based on document type
            result["structured_data"] = self.parse_structured_data(result["extracted_text"], doc_type)
            result["confidence_score"] = self.calculate_confidence(result["extracted_text"])
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            result["error"] = str(e)
        
        return result
    
    def detect_handwriting(self, image: Image.Image) -> bool:
        """Simple handwriting detection (can be improved with ML model)"""
        # This is a simplified version - in practice, you'd use a classifier
        # For now, we'll use some basic heuristics
        return False  # Default to printed text
    
    def parse_structured_data(self, text: str, doc_type: str) -> Dict[str, Any]:
        """Parse structured data based on document type"""
        structured_data = {}
        
        if doc_type in ["resume", "cv"]:
            structured_data = self.parse_resume(text)
        elif doc_type == "invoice":
            structured_data = self.parse_invoice(text)
        elif doc_type == "marksheet":
            structured_data = self.parse_marksheet(text)
        elif doc_type == "cheque":
            structured_data = self.parse_cheque(text)
        else:
            # Auto-detect and parse
            structured_data = self.auto_parse(text)
        
        return structured_data
    
    def parse_resume(self, text: str) -> Dict[str, Any]:
        """Parse resume/CV data"""
        resume_data = {
            "personal_info": {},
            "skills": {"technical": [], "soft": []},
            "experience": [],
            "education": [],
            "achievements": [],
            "tools_technologies": []
        }
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            resume_data["personal_info"]["email"] = emails[0]
        
        # Extract phone numbers
        phone_pattern = r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phones = re.findall(phone_pattern, text)
        if phones:
            resume_data["personal_info"]["phone"] = phones[0]
        
        # Extract skills using NER and keyword matching
        if self.ner_pipeline:
            entities = self.ner_pipeline(text)
            for entity in entities:
                if entity['entity_group'] == 'MISC':
                    resume_data["skills"]["technical"].append(entity['word'])
        
        # Common technical skills
        tech_skills = [
            'python', 'java', 'javascript', 'react', 'node.js', 'sql', 'mongodb',
            'docker', 'kubernetes', 'aws', 'azure', 'git', 'tensorflow', 'pytorch',
            'machine learning', 'data science', 'artificial intelligence'
        ]
        
        text_lower = text.lower()
        for skill in tech_skills:
            if skill in text_lower:
                resume_data["skills"]["technical"].append(skill.title())
        
        # Extract years of experience
        exp_pattern = r'(\d+)(?:\+)?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)'
        exp_matches = re.findall(exp_pattern, text.lower())
        if exp_matches:
            resume_data["experience_years"] = max([int(x) for x in exp_matches])
        
        return resume_data
    
    def parse_invoice(self, text: str) -> Dict[str, Any]:
        """Parse invoice data"""
        invoice_data = {
            "invoice_number": "",
            "date": "",
            "amount": "",
            "vendor": "",
            "items": []
        }
        
        # Extract invoice number
        inv_pattern = r'(?:invoice|inv)(?:\s*#|\s*no\.?|\s*number)?\s*:?\s*([A-Z0-9-]+)'
        inv_match = re.search(inv_pattern, text, re.IGNORECASE)
        if inv_match:
            invoice_data["invoice_number"] = inv_match.group(1)
        
        # Extract amount
        amount_pattern = r'(?:total|amount|sum)?\s*:?\s*[$₹€£]?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)'
        amount_match = re.search(amount_pattern, text, re.IGNORECASE)
        if amount_match:
            invoice_data["amount"] = amount_match.group(1)
        
        return invoice_data
    
    def parse_marksheet(self, text: str) -> Dict[str, Any]:
        """Parse marksheet/transcript data"""
        marksheet_data = {
            "student_name": "",
            "roll_number": "",
            "subjects": [],
            "grades": [],
            "gpa": "",
            "percentage": ""
        }
        
        # Extract percentage
        perc_pattern = r'(\d+(?:\.\d+)?)\s*%'
        perc_matches = re.findall(perc_pattern, text)
        if perc_matches:
            marksheet_data["percentage"] = max(perc_matches)
        
        # Extract GPA
        gpa_pattern = r'(?:gpa|cgpa)\s*:?\s*(\d+\.\d+)'
        gpa_match = re.search(gpa_pattern, text, re.IGNORECASE)
        if gpa_match:
            marksheet_data["gpa"] = gpa_match.group(1)
        
        return marksheet_data
    
    def parse_cheque(self, text: str) -> Dict[str, Any]:
        """Parse cheque data"""
        cheque_data = {
            "amount_words": "",
            "amount_figures": "",
            "payee": "",
            "date": "",
            "cheque_number": ""
        }
        
        # Extract amount in figures
        amount_pattern = r'₹\s*(\d+(?:,\d{3})*(?:\.\d{2})?)'
        amount_match = re.search(amount_pattern, text)
        if amount_match:
            cheque_data["amount_figures"] = amount_match.group(1)
        
        return cheque_data
    
    def auto_parse(self, text: str) -> Dict[str, Any]:
        """Auto-detect document type and parse accordingly"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['resume', 'cv', 'experience', 'skills']):
            return self.parse_resume(text)
        elif any(word in text_lower for word in ['invoice', 'bill', 'amount', 'total']):
            return self.parse_invoice(text)
        elif any(word in text_lower for word in ['marks', 'grade', 'percentage', 'gpa']):
            return self.parse_marksheet(text)
        elif any(word in text_lower for word in ['pay', 'cheque', 'bank']):
            return self.parse_cheque(text)
        else:
            return {"type": "unknown", "raw_text": text}
    
    def calculate_confidence(self, text: str) -> float:
        """Calculate confidence score based on text quality"""
        if not text.strip():
            return 0.0
        
        # Simple confidence calculation based on:
        # - Text length
        # - Presence of common words
        # - Character distribution
        
        score = 0.0
        
        # Length factor
        if len(text) > 100:
            score += 0.3
        elif len(text) > 50:
            score += 0.2
        else:
            score += 0.1
        
        # Word count factor
        words = text.split()
        if len(words) > 20:
            score += 0.3
        elif len(words) > 10:
            score += 0.2
        else:
            score += 0.1
        
        # Character variety
        unique_chars = len(set(text.lower()))
        if unique_chars > 20:
            score += 0.4
        elif unique_chars > 10:
            score += 0.3
        else:
            score += 0.2
        
        return min(score, 1.0)

class JobMatcher:
    """Job matching system with skill analysis"""
    
    def __init__(self):
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.job_descriptions = self.create_job_descriptions()
    
    def create_job_descriptions(self) -> List[Dict[str, Any]]:
        """Create 10 diverse Python job descriptions"""
        jobs = [
            {
                "id": 1,
                "title": "Senior Python Developer",
                "company": "TechCorp Solutions",
                "experience": "5-7 years",
                "location": "Bangalore",
                "skills": [
                    "Python", "Django", "Flask", "PostgreSQL", "Redis", "Docker", 
                    "AWS", "Git", "RESTful APIs", "Microservices"
                ],
                "soft_skills": [
                    "Team Leadership", "Problem Solving", "Communication", 
                    "Agile Methodology", "Code Review"
                ],
                "tools": [
                    "PyCharm", "Jupyter", "Postman", "Jenkins", "Kubernetes"
                ],
                "description": "We are seeking a Senior Python Developer to lead our backend development team. You will be responsible for designing scalable web applications, mentoring junior developers, and implementing best practices in software development.",
                "requirements": [
                    "5+ years of Python development experience",
                    "Strong experience with Django/Flask frameworks",
                    "Database design and optimization skills",
                    "Cloud platform experience (AWS/Azure)",
                    "Leadership and mentoring experience"
                ]
            },
            {
                "id": 2,
                "title": "Python Data Scientist",
                "company": "DataInsights AI",
                "experience": "3-5 years",
                "location": "Mumbai",
                "skills": [
                    "Python", "Pandas", "NumPy", "Scikit-learn", "TensorFlow", 
                    "PyTorch", "SQL", "Statistics", "Machine Learning", "Deep Learning"
                ],
                "soft_skills": [
                    "Analytical Thinking", "Research Skills", "Communication", 
                    "Presentation Skills", "Curiosity"
                ],
                "tools": [
                    "Jupyter", "Apache Spark", "Tableau", "Power BI", "Git"
                ],
                "description": "Join our AI team to build cutting-edge machine learning models. You'll work on predictive analytics, natural language processing, and computer vision projects.",
                "requirements": [
                    "Strong Python programming skills",
                    "Experience with ML/DL frameworks",
                    "Statistical analysis expertise",
                    "Data visualization skills",
                    "PhD/Masters in relevant field preferred"
                ]
            },
            {
                "id": 3,
                "title": "Python Backend Engineer",
                "company": "StartupTech",
                "experience": "2-4 years",
                "location": "Remote",
                "skills": [
                    "Python", "FastAPI", "Django", "MongoDB", "PostgreSQL", 
                    "Redis", "Celery", "Docker", "Linux", "Git"
                ],
                "soft_skills": [
                    "Self-motivated", "Remote Collaboration", "Problem Solving", 
                    "Time Management", "Adaptability"
                ],
                "tools": [
                    "VS Code", "Docker", "Postman", "GitHub Actions", "MongoDB Compass"
                ],
                "description": "Build robust backend systems for our growing startup. You'll work in a fast-paced environment with modern technologies and agile practices.",
                "requirements": [
                    "2+ years Python backend experience",
                    "API design and development",
                    "Database management skills",
                    "Remote work experience",
                    "Startup mindset"
                ]
            },
            {
                "id": 4,
                "title": "Python DevOps Engineer",
                "company": "CloudFirst Technologies",
                "experience": "4-6 years",
                "location": "Hyderabad",
                "skills": [
                    "Python", "Ansible", "Terraform", "Docker", "Kubernetes", 
                    "AWS", "CI/CD", "Jenkins", "Monitoring", "Linux"
                ],
                "soft_skills": [
                    "System Thinking", "Troubleshooting", "Collaboration", 
                    "Documentation", "Continuous Learning"
                ],
                "tools": [
                    "Jenkins", "GitLab CI", "Prometheus", "Grafana", "ELK Stack"
                ],
                "description": "Automate infrastructure and deployment processes using Python. Work closely with development teams to implement DevOps best practices.",
                "requirements": [
                    "Strong Python scripting skills",
                    "Cloud platform experience",
                    "Infrastructure as Code experience",
                    "Container orchestration knowledge",
                    "Monitoring and logging expertise"
                ]
            },
            {
                "id": 5,
                "title": "Junior Python Developer",
                "company": "EduTech Solutions",
                "experience": "0-2 years",
                "location": "Pune",
                "skills": [
                    "Python", "Django", "HTML", "CSS", "JavaScript", "SQLite", 
                    "Git", "Bootstrap", "jQuery", "Basic Linux"
                ],
                "soft_skills": [
                    "Eagerness to Learn", "Team Player", "Attention to Detail", 
                    "Communication", "Patience"
                ],
                "tools": [
                    "VS Code", "Git", "Chrome DevTools", "SQLite Browser"
                ],
                "description": "Perfect opportunity for fresh graduates to start their Python development career. You'll work on educational technology projects with mentorship from senior developers.",
                "requirements": [
                    "Bachelor's degree in Computer Science",
                    "Basic Python programming knowledge",
                    "Understanding of web development basics",
                    "Good communication skills",
                    "Willingness to learn"
                ]
            },
            {
                "id": 6,
                "title": "Python ML Engineer",
                "company": "AI Innovations Lab",
                "experience": "3-5 years",
                "location": "Chennai",
                "skills": [
                    "Python", "TensorFlow", "PyTorch", "MLflow", "Kubernetes", 
                    "Docker", "Apache Airflow", "SQL", "NoSQL", "Model Deployment"
                ],
                "soft_skills": [
                    "Innovation", "Research Oriented", "Problem Solving", 
                    "Collaboration", "Technical Communication"
                ],
                "tools": [
                    "MLflow", "Kubeflow", "TensorBoard", "Weights & Biases", "DVC"
                ],
                "description": "Deploy and maintain machine learning models in production. Work on MLOps pipelines and model monitoring systems.",
                "requirements": [
                    "ML model deployment experience",
                    "Container and orchestration knowledge",
                    "ML pipeline development",
                    "Model monitoring and maintenance",
                    "Cloud ML platform experience"
                ]
            },
            {
                "id": 7,
                "title": "Python Full Stack Developer",
                "company": "WebSolutions Pro",
                "experience": "3-5 years",
                "location": "Delhi",
                "skills": [
                    "Python", "Django", "React", "JavaScript", "PostgreSQL", 
                    "Redis", "HTML5", "CSS3", "RESTful APIs", "GraphQL"
                ],
                "soft_skills": [
                    "Versatility", "UI/UX Awareness", "Client Communication", 
                    "Project Management", "Creativity"
                ],
                "tools": [
                    "VS Code", "React DevTools", "Postman", "Figma", "Git"
                ],
                "description": "Develop end-to-end web applications using Python backend and modern frontend technologies. Work directly with clients on custom solutions.",
                "requirements": [
                    "Full stack development experience",
                    "Frontend framework proficiency",
                    "Database design skills",
                    "Client interaction experience",
                    "Project delivery experience"
                ]
            },
            {
                "id": 8,
                "title": "Python Automation Engineer",
                "company": "QualityFirst Testing",
                "experience": "2-4 years",
                "location": "Noida",
                "skills": [
                    "Python", "Selenium", "Pytest", "Robot Framework", "API Testing", 
                    "Jenkins", "TestRail", "Git", "Linux", "SQL"
                ],
                "soft_skills": [
                    "Attention to Detail", "Analytical Thinking", "Patience", 
                    "Documentation", "Quality Focus"
                ],
                "tools": [
                    "Selenium IDE", "Postman", "JIRA", "TestRail", "Jenkins"
                ],
                "description": "Build comprehensive test automation frameworks using Python. Ensure software quality through automated testing strategies.",
                "requirements": [
                    "Test automation experience",
                    "Web and API testing knowledge",
                    "Framework development skills",
                    "CI/CD integration experience",
                    "Quality assurance background"
                ]
            },
            {
                "id": 9,
                "title": "Python Research Engineer",
                "company": "Academic Research Institute",
                "experience": "4-8 years",
                "location": "Kolkata",
                "skills": [
                    "Python", "Research", "Statistics", "Data Analysis", "Scientific Computing", 
                    "NumPy", "SciPy", "Matplotlib", "LaTeX", "R"
                ],
                "soft_skills": [
                    "Research Methodology", "Critical Thinking", "Academic Writing", 
                    "Presentation Skills", "Peer Collaboration"
                ],
                "tools": [
                    "Jupyter", "LaTeX", "MATLAB", "R Studio", "Reference Managers"
                ],
                "description": "Conduct computational research using Python for scientific applications. Publish findings in peer-reviewed journals.",
                "requirements": [
                    "PhD in relevant field",
                    "Research publication record",
                    "Scientific programming experience",
                    "Statistical analysis expertise",
                    "Grant writing experience"
                ]
            },
            {
                "id": 10,
                "title": "Python Security Engineer",
                "company": "CyberSecure Systems",
                "experience": "4-7 years",
                "location": "Gurgaon",
                "skills": [
                    "Python", "Cybersecurity", "Penetration Testing", "Cryptography", 
                    "Network Security", "OWASP", "Linux", "Bash", "SQL Injection"
                ],
                "soft_skills": [
                    "Security Mindset", "Ethical Hacking", "Risk Assessment", 
                    "Incident Response", "Compliance Awareness"
                ],
                "tools": [
                    "Burp Suite", "Metasploit", "Wireshark", "Nmap", "OWASP ZAP"
                ],
                "description": "Develop security tools and conduct vulnerability assessments using Python. Protect organizational assets from cyber threats.",
                "requirements": [
                    "Cybersecurity experience",
                    "Penetration testing skills",
                    "Security tool development",
                    "Compliance knowledge",
                    "Incident response experience"
                ]
            }
        ]
        return jobs
    
    def calculate_skill_match(self, resume_skills: List[str], job_skills: List[str]) -> float:
        """Calculate skill match percentage"""
        if not resume_skills or not job_skills:
            return 0.0
        
        resume_skills_lower = [skill.lower().strip() for skill in resume_skills]
        job_skills_lower = [skill.lower().strip() for skill in job_skills]
        
        matches = len(set(resume_skills_lower) & set(job_skills_lower))
        total_job_skills = len(job_skills_lower)
        
        return (matches / total_job_skills) * 100 if total_job_skills > 0 else 0.0
    
    def calculate_semantic_similarity(self, resume_text: str, job_description: str) -> float:
        """Calculate semantic similarity using sentence transformers"""
        try:
            embeddings = self.sentence_model.encode([resume_text, job_description])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity * 100)
        except Exception as e:
            logger.error(f"Semantic similarity calculation failed: {e}")
            return 0.0
    
    def match_resume_to_jobs(self, resume_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Match resume to all job descriptions and rank them"""
        matches = []
        
        # Combine all resume text for semantic analysis
        resume_text = f"""
        Skills: {', '.join(resume_data.get('skills', {}).get('technical', []))}
        Soft Skills: {', '.join(resume_data.get('skills', {}).get('soft', []))}
        Tools: {', '.join(resume_data.get('tools_technologies', []))}
        Experience: {resume_data.get('experience_years', 0)} years
        """
        
        for job in self.job_descriptions:
            # Calculate different match scores
            technical_score = self.calculate_skill_match(
                resume_data.get('skills', {}).get('technical', []),
                job['skills']
            )
            
            soft_skills_score = self.calculate_skill_match(
                resume_data.get('skills', {}).get('soft', []),
                job['soft_skills']
            )
            
            tools_score = self.calculate_skill_match(
                resume_data.get('tools_technologies', []),
                job['tools']
            )
            
            # Semantic similarity
            job_text = f"{job['description']} {' '.join(job['requirements'])}"
            semantic_score = self.calculate_semantic_similarity(resume_text, job_text)
            
            # Experience match
            resume_exp = resume_data.get('experience_years', 0)
            job_exp_range = job['experience']
            exp_score = self.calculate_experience_match(resume_exp, job_exp_range)
            
            # Overall match score (weighted average)
            overall_score = (
                technical_score * 0.35 +
                semantic_score * 0.25 +
                soft_skills_score * 0.15 +
                tools_score * 0.15 +
                exp_score * 0.10
            )
            
            match_result = {
                "job_id": job['id'],
                "job_title": job['title'],
                "company": job['company'],
                "location": job['location'],
                "overall_match_score": round(overall_score, 2),
                "technical_skills_match": round(technical_score, 2),
                "soft_skills_match": round(soft_skills_score, 2),
                "tools_match": round(tools_score, 2),
                "semantic_similarity": round(semantic_score, 2),
                "experience_match": round(exp_score, 2),
                "matched_skills": list(set(resume_data.get('skills', {}).get('technical', [])) & 
                                     set([s.lower() for s in job['skills']])),
                "missing_skills": [skill for skill in job['skills'] 
                                 if skill.lower() not in [s.lower() for s in resume_data.get('skills', {}).get('technical', [])]],
                "job_description": job['description']
            }
            
            matches.append(match_result)
        
        # Sort by overall match score
        matches.sort(key=lambda x: x['overall_match_score'], reverse=True)
        
        return matches
    
    def calculate_experience_match(self, resume_exp: int, job_exp_range: str) -> float:
        """Calculate experience match score"""
        try:
            # Parse job experience range
            exp_parts = job_exp_range.lower().replace('years', '').replace('year', '').strip()
            
            if '-' in exp_parts:
                min_exp, max_exp = map(int, exp_parts.split('-'))
                
                if min_exp <= resume_exp <= max_exp:
                    return 100.0
                elif resume_exp < min_exp:
                    # Under-qualified
                    diff = min_exp - resume_exp
                    return max(0, 100 - (diff * 20))
                else:
                    # Over-qualified
                    diff = resume_exp - max_exp
                    return max(50, 100 - (diff * 10))
            else:
                # Single number or "0-2", "3+" format
                if '+' in exp_parts:
                    min_exp = int(exp_parts.replace('+', ''))
                    return 100.0 if resume_exp >= min_exp else max(0, 100 - (min_exp - resume_exp) * 20)
                else:
                    target_exp = int(exp_parts)
                    diff = abs(resume_exp - target_exp)
                    return max(0, 100 - (diff * 15))
        
        except Exception as e:
            logger.error(f"Experience match calculation failed: {e}")
            return 50.0  # Default neutral score
    
    def get_top_matches(self, resume_data: Dict[str, Any], top_n: int = 5) -> List[Dict[str, Any]]:
        """Get top N job matches"""
        matches = self.match_resume_to_jobs(resume_data)
        return matches[:top_n]

# Pydantic models for API
class ResumeData(BaseModel):
    personal_info: Dict[str, Any] = {}
    skills: Dict[str, List[str]] = {"technical": [], "soft": []}
    experience_years: int = 0
    tools_technologies: List[str] = []
    achievements: List[Dict[str, Any]] = []

class JobMatchResponse(BaseModel):
    job_id: int
    job_title: str
    company: str
    location: str
    overall_match_score: float
    technical_skills_match: float
    soft_skills_match: float
    tools_match: float
    semantic_similarity: float
    experience_match: float
    matched_skills: List[str]
    missing_skills: List[str]
    job_description: str

class DocumentProcessResponse(BaseModel):
    file_path: str
    document_type: str
    extracted_text: str
    structured_data: Dict[str, Any]
    confidence_score: float
    error: Optional[str] = None

# FastAPI Application
app = FastAPI(
    title="Advanced OCR & Job Matching System",
    description="Process documents with OCR and match resumes to job descriptions",
    version="1.0.0"
)

# Initialize processors
document_processor = DocumentProcessor()
job_matcher = JobMatcher()

@app.post("/process-document/", response_model=DocumentProcessResponse)
async def process_document(
    file: UploadFile = File(...),
    document_type: str = "auto"
):
    """Process uploaded document using OCR"""
    import tempfile
    import shutil
    
    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Create safe filename
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in (' ', '.', '_')).rstrip()
        temp_path = os.path.join(temp_dir, safe_filename)
        
        # Save uploaded file
        with open(temp_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        # Verify file exists
        if not os.path.exists(temp_path):
            raise FileNotFoundError(f"Failed to save file at {temp_path}")
        
        print(f"Processing file: {temp_path}, Size: {os.path.getsize(temp_path)} bytes")
        
        # Process document
        result = document_processor.process_document(temp_path, document_type)
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return DocumentProcessResponse(**result)
    
    except Exception as e:
        # Clean up on error
        try:
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass
        
        print(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

@app.post("/match-jobs/", response_model=List[JobMatchResponse])
async def match_jobs(resume_data: ResumeData, top_n: int = 5):
    """Match resume data to job descriptions"""
    try:
        matches = job_matcher.get_top_matches(resume_data.dict(), top_n)
        return [JobMatchResponse(**match) for match in matches]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Job matching failed: {str(e)}")

@app.post("/process-and-match/")
async def process_and_match(
    file: UploadFile = File(...),
    document_type: str = "resume",
    top_n: int = 5
):
    """Process resume document and match to jobs in one step"""
    import tempfile
    import shutil
    
    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Create safe filename
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in (' ', '.', '_')).rstrip()
        temp_path = os.path.join(temp_dir, safe_filename)
        
        # Save uploaded file
        with open(temp_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        # Verify file exists
        if not os.path.exists(temp_path):
            raise FileNotFoundError(f"Failed to save file at {temp_path}")
        
        print(f"Processing file: {temp_path}, Size: {os.path.getsize(temp_path)} bytes")
        
        # Process document
        doc_result = document_processor.process_document(temp_path, document_type)
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Extract resume data
        if document_type in ["resume", "cv"] and doc_result.get("structured_data"):
            resume_data = doc_result["structured_data"]
            
            # Match to jobs
            matches = job_matcher.get_top_matches(resume_data, top_n)
            
            return {
                "document_processing": doc_result,
                "job_matches": matches,
                "summary": {
                    "total_jobs_analyzed": len(job_matcher.job_descriptions),
                    "top_matches_returned": len(matches),
                    "best_match_score": matches[0]["overall_match_score"] if matches else 0,
                    "processing_confidence": doc_result.get("confidence_score", 0)
                }
            }
        else:
            return {
                "document_processing": doc_result,
                "error": "Document type not suitable for job matching or processing failed"
            }
    
    except Exception as e:
        # Clean up on error
        try:
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass
        
        print(f"Error in process and match: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Process and match failed: {str(e)}")

@app.get("/job-descriptions/")
async def get_job_descriptions():
    """Get all available job descriptions"""
    return {"jobs": job_matcher.job_descriptions}

@app.get("/job-descriptions/{job_id}")
async def get_job_description(job_id: int):
    """Get specific job description"""
    job = next((job for job in job_matcher.job_descriptions if job["id"] == job_id), None)
    if job:
        return job
    else:
        raise HTTPException(status_code=404, detail="Job not found")

@app.post("/analyze-skill-gap/")
async def analyze_skill_gap(resume_data: ResumeData, job_id: int):
    """Analyze skill gap between resume and specific job"""
    try:
        job = next((job for job in job_matcher.job_descriptions if job["id"] == job_id), None)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        resume_skills = resume_data.skills.get("technical", [])
        job_skills = job["skills"]
        
        matched_skills = list(set([s.lower() for s in resume_skills]) & set([s.lower() for s in job_skills]))
        missing_skills = [skill for skill in job_skills if skill.lower() not in [s.lower() for s in resume_skills]]
        extra_skills = [skill for skill in resume_skills if skill.lower() not in [s.lower() for s in job_skills]]
        
        skill_gap_analysis = {
            "job_title": job["title"],
            "company": job["company"],
            "total_required_skills": len(job_skills),
            "matched_skills": len(matched_skills),
            "missing_skills_count": len(missing_skills),
            "match_percentage": (len(matched_skills) / len(job_skills)) * 100 if job_skills else 0,
            "matched_skills_list": matched_skills,
            "missing_skills_list": missing_skills,
            "additional_skills": extra_skills,
            "recommendations": {
                "priority_skills_to_learn": missing_skills[:3],  # Top 3 missing skills
                "suggested_learning_path": [
                    f"Focus on {skill} - high demand in {job['title']} roles" 
                    for skill in missing_skills[:3]
                ],
                "strengths": matched_skills,
                "readiness_level": "Ready" if len(matched_skills) / len(job_skills) > 0.7 else "Needs Development"
            }
        }
        
        return skill_gap_analysis
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Skill gap analysis failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Advanced OCR & Job Matching System",
        "version": "1.0.0",
        "description": "Process documents with OCR and match resumes to job descriptions",
        "endpoints": {
            "process_document": "/process-document/",
            "match_jobs": "/match-jobs/",
            "process_and_match": "/process-and-match/",
            "job_descriptions": "/job-descriptions/",
            "skill_gap_analysis": "/analyze-skill-gap/",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "supported_document_types": [
            "resume", "cv", "invoice", "marksheet", "cheque", "challan", "handwritten"
        ],
        "supported_file_formats": [
            "PDF", "PNG", "JPG", "JPEG", "TIFF", "BMP"
        ]
    }

@app.post("/debug-upload/")
async def debug_upload(file: UploadFile = File(...)):
    """Debug endpoint to test file upload without processing"""
    import tempfile
    import shutil
    
    try:
        # Get file info
        file_info = {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": 0
        }
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        print(f"Created temp directory: {temp_dir}")
        
        # Create safe filename
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in (' ', '.', '_')).rstrip()
        temp_path = os.path.join(temp_dir, safe_filename)
        
        # Read and save file
        content = await file.read()
        file_info["size"] = len(content)
        
        with open(temp_path, "wb") as temp_file:
            temp_file.write(content)
        
        # Verify file
        if os.path.exists(temp_path):
            actual_size = os.path.getsize(temp_path)
            file_info["saved_successfully"] = True
            file_info["saved_size"] = actual_size
            file_info["temp_path"] = temp_path
        else:
            file_info["saved_successfully"] = False
            file_info["error"] = "File not found after saving"
        
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return {
            "status": "success",
            "file_info": file_info,
            "temp_dir_used": temp_dir,
            "python_temp_dir": tempfile.gettempdir(),
            "current_working_dir": os.getcwd()
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
            "python_temp_dir": tempfile.gettempdir(),
            "current_working_dir": os.getcwd()
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "trocr_printed": bool(document_processor.trocr_model),
            "trocr_handwritten": bool(document_processor.trocr_hw_model),
            "sentence_transformer": bool(job_matcher.sentence_model),
            "ner_pipeline": bool(document_processor.ner_pipeline)
        }
    }

# Example usage and testing functions
def test_system():
    """Test the OCR and job matching system"""
    print("Testing OCR & Job Matching System...")
    
    # Test job matching with sample resume data
    sample_resume = {
        "personal_info": {
            "email": "john.doe@email.com",
            "phone": "+91-9876543210"
        },
        "skills": {
            "technical": ["Python", "Django", "PostgreSQL", "Docker", "AWS", "Git"],
            "soft": ["Team Leadership", "Problem Solving", "Communication"]
        },
        "experience_years": 5,
        "tools_technologies": ["PyCharm", "Jupyter", "Postman", "Jenkins"],
        "achievements": [
            {"title": "Employee of the Year", "year": "2023", "issued_by": "TechCorp"}
        ]
    }
    
    matcher = JobMatcher()
    matches = matcher.get_top_matches(sample_resume, 3)
    
    print(f"\nTop 3 Job Matches for Sample Resume:")
    for i, match in enumerate(matches, 1):
        print(f"\n{i}. {match['job_title']} at {match['company']}")
        print(f"   Overall Match: {match['overall_match_score']:.1f}%")
        print(f"   Technical Skills: {match['technical_skills_match']:.1f}%")
        print(f"   Location: {match['location']}")
        print(f"   Matched Skills: {match['matched_skills']}")
        print(f"   Missing Skills: {match['missing_skills'][:3]}")  # Show first 3

if __name__ == "__main__":
    # Test the system
    test_system()
    
    # Run the FastAPI server
    print("\nStarting FastAPI server...")
    print("Access Swagger UI at: http://localhost:8000/docs")
    print("Access ReDoc at: http://localhost:8000/redoc")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

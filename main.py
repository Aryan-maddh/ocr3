#!/usr/bin/env python3
"""
Complete OCR Job Matching System
Features:
- Multi-OCR engine support (PaddleOCR, EasyOCR, TrOCR)
- Document parsing (Resume, CV, Invoices, Cheques, etc.)
- Job description matching with scoring
- Skills extraction and competency mapping
- Local LLM integration (Ollama Llama3.2:3b)
- FastAPI with Swagger documentation
"""

import os
import json
import re
import asyncio
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import logging

# Core libraries
import numpy as np
import pandas as pd
from PIL import Image
import cv2

# OCR libraries
import easyocr
import paddleocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# NLP and ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from fuzzywuzzy import fuzz, process

# API Framework
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Ollama client
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCRConfig:
    """Configuration for OCR engines"""
    def __init__(self):
        self.easyocr_langs = ['en']
        self.paddleocr_lang = 'en'
        self.trocr_model = 'microsoft/trocr-base-printed'
        self.device = 'cpu'  # Change to 'cuda' if GPU available

class DocumentTypes:
    """Supported document types"""
    RESUME = "resume"
    CV = "cv"
    MARKSHEET = "marksheet"
    INVOICE = "invoice"
    CHEQUE = "cheque"
    CHALLAN = "challan"
    HANDWRITTEN = "handwritten"
    GENERAL = "general"

class JobDescriptions:
    """Sample Python job descriptions"""
    @staticmethod
    def get_python_jobs():
        return [
            {
                "id": 1,
                "title": "Senior Python Developer - Full Stack",
                "company": "TechCorp Solutions",
                "location": "Bangalore, India",
                "experience": "5-8 years",
                "skills": [
                    "Python", "Django", "Flask", "FastAPI", "React", "JavaScript",
                    "PostgreSQL", "Redis", "Docker", "AWS", "Git", "REST APIs",
                    "GraphQL", "Celery", "pytest", "Linux"
                ],
                "soft_skills": ["Problem solving", "Team leadership", "Communication", "Agile methodology"],
                "tools": ["PyCharm", "VS Code", "Jira", "Jenkins", "GitHub Actions"],
                "requirements": [
                    "5+ years Python development experience",
                    "Strong knowledge of web frameworks",
                    "Experience with cloud platforms",
                    "Database design and optimization",
                    "API development and integration"
                ],
                "responsibilities": [
                    "Lead development of scalable web applications",
                    "Mentor junior developers",
                    "Architecture design and code reviews",
                    "Performance optimization"
                ]
            },
            {
                "id": 2,
                "title": "Python Data Engineer",
                "company": "DataFlow Analytics",
                "location": "Mumbai, India",
                "experience": "3-6 years",
                "skills": [
                    "Python", "Apache Spark", "Hadoop", "Kafka", "Airflow",
                    "SQL", "NoSQL", "ETL", "Pandas", "NumPy", "Docker",
                    "Kubernetes", "AWS", "GCP", "Snowflake"
                ],
                "soft_skills": ["Analytical thinking", "Attention to detail", "Collaboration"],
                "tools": ["Databricks", "Tableau", "Power BI", "Jupyter", "Git"],
                "requirements": [
                    "Strong Python programming skills",
                    "Experience with big data technologies",
                    "ETL pipeline development",
                    "Cloud platform experience"
                ],
                "responsibilities": [
                    "Design and maintain data pipelines",
                    "Optimize data processing workflows",
                    "Collaborate with data scientists",
                    "Ensure data quality and governance"
                ]
            },
            {
                "id": 3,
                "title": "Python Machine Learning Engineer",
                "company": "AI Innovations Ltd",
                "location": "Hyderabad, India",
                "experience": "4-7 years",
                "skills": [
                    "Python", "scikit-learn", "TensorFlow", "PyTorch", "Keras",
                    "MLflow", "Kubeflow", "Docker", "Kubernetes", "AWS SageMaker",
                    "Pandas", "NumPy", "Matplotlib", "Seaborn", "SQL"
                ],
                "soft_skills": ["Research mindset", "Problem solving", "Communication"],
                "tools": ["Jupyter", "MLflow", "Weights & Biases", "Git", "DVC"],
                "requirements": [
                    "Strong ML algorithms knowledge",
                    "Model deployment experience",
                    "Statistical analysis skills",
                    "MLOps practices"
                ],
                "responsibilities": [
                    "Develop and deploy ML models",
                    "Model performance monitoring",
                    "Research new ML techniques",
                    "Collaborate with product teams"
                ]
            },
            {
                "id": 4,
                "title": "Python Backend Developer",
                "company": "CloudTech Systems",
                "location": "Pune, India",
                "experience": "2-5 years",
                "skills": [
                    "Python", "FastAPI", "Django", "Flask", "SQLAlchemy",
                    "PostgreSQL", "MongoDB", "Redis", "RabbitMQ", "Docker",
                    "AWS", "Elasticsearch", "pytest", "Git"
                ],
                "soft_skills": ["Team collaboration", "Problem solving", "Time management"],
                "tools": ["PyCharm", "Postman", "Docker", "Jenkins", "Grafana"],
                "requirements": [
                    "Strong Python fundamentals",
                    "RESTful API development",
                    "Database design skills",
                    "Testing methodologies"
                ],
                "responsibilities": [
                    "Develop robust backend services",
                    "API design and implementation",
                    "Database optimization",
                    "Code reviews and testing"
                ]
            },
            {
                "id": 5,
                "title": "Python DevOps Engineer",
                "company": "InfraTech Solutions",
                "location": "Chennai, India",
                "experience": "3-6 years",
                "skills": [
                    "Python", "Bash", "Terraform", "Ansible", "Docker",
                    "Kubernetes", "AWS", "Azure", "Jenkins", "GitLab CI",
                    "Monitoring", "Logging", "Prometheus", "Grafana"
                ],
                "soft_skills": ["System thinking", "Troubleshooting", "Communication"],
                "tools": ["Jenkins", "Terraform", "Ansible", "Helm", "ArgoCD"],
                "requirements": [
                    "Infrastructure automation experience",
                    "CI/CD pipeline development",
                    "Cloud platform expertise",
                    "Monitoring and logging"
                ],
                "responsibilities": [
                    "Automate deployment processes",
                    "Manage cloud infrastructure",
                    "Monitor system performance",
                    "Ensure high availability"
                ]
            },
            {
                "id": 6,
                "title": "Python QA Automation Engineer",
                "company": "QualityFirst Tech",
                "location": "Noida, India",
                "experience": "2-4 years",
                "skills": [
                    "Python", "Selenium", "pytest", "Robot Framework",
                    "API Testing", "Postman", "TestNG", "CI/CD", "Git",
                    "SQL", "Linux", "Docker", "Jenkins"
                ],
                "soft_skills": ["Attention to detail", "Analytical thinking", "Communication"],
                "tools": ["Selenium Grid", "Allure", "JIRA", "TestRail", "BrowserStack"],
                "requirements": [
                    "Test automation framework development",
                    "API and UI testing experience",
                    "Bug tracking and reporting",
                    "Continuous integration"
                ],
                "responsibilities": [
                    "Design test automation frameworks",
                    "Execute automated test suites",
                    "Bug reporting and tracking",
                    "Performance testing"
                ]
            },
            {
                "id": 7,
                "title": "Python Blockchain Developer",
                "company": "CryptoTech Innovations",
                "location": "Bangalore, India",
                "experience": "3-5 years",
                "skills": [
                    "Python", "Solidity", "Web3.py", "Ethereum", "Smart Contracts",
                    "DeFi", "IPFS", "Blockchain", "Cryptography", "Django",
                    "PostgreSQL", "Docker", "AWS"
                ],
                "soft_skills": ["Innovation", "Security mindset", "Research skills"],
                "tools": ["Truffle", "Hardhat", "MetaMask", "Remix", "Ganache"],
                "requirements": [
                    "Blockchain technology understanding",
                    "Smart contract development",
                    "Cryptocurrency knowledge",
                    "Security best practices"
                ],
                "responsibilities": [
                    "Develop blockchain applications",
                    "Smart contract implementation",
                    "DApp frontend integration",
                    "Security auditing"
                ]
            },
            {
                "id": 8,
                "title": "Python Game Developer",
                "company": "GameStudio Pro",
                "location": "Mumbai, India",
                "experience": "2-5 years",
                "skills": [
                    "Python", "Pygame", "Unity3D", "C#", "Blender",
                    "3D Graphics", "Game Physics", "AI", "Networking",
                    "Git", "Agile", "Mobile Development"
                ],
                "soft_skills": ["Creativity", "Team collaboration", "Problem solving"],
                "tools": ["Unity", "Blender", "Photoshop", "JIRA", "Perforce"],
                "requirements": [
                    "Game development experience",
                    "3D graphics knowledge",
                    "Physics simulation",
                    "Multi-platform development"
                ],
                "responsibilities": [
                    "Develop game mechanics",
                    "Implement AI systems",
                    "Performance optimization",
                    "Cross-platform compatibility"
                ]
            },
            {
                "id": 9,
                "title": "Python Security Engineer",
                "company": "SecureNet Solutions",
                "location": "Delhi, India",
                "experience": "4-7 years",
                "skills": [
                    "Python", "Cybersecurity", "Penetration Testing", "OWASP",
                    "Cryptography", "Network Security", "Vulnerability Assessment",
                    "Linux", "Docker", "AWS Security", "SIEM"
                ],
                "soft_skills": ["Ethical mindset", "Analytical thinking", "Communication"],
                "tools": ["Metasploit", "Nmap", "Wireshark", "Burp Suite", "Splunk"],
                "requirements": [
                    "Security testing experience",
                    "Vulnerability assessment skills",
                    "Compliance knowledge",
                    "Incident response"
                ],
                "responsibilities": [
                    "Conduct security assessments",
                    "Develop security tools",
                    "Monitor security incidents",
                    "Security training and awareness"
                ]
            },
            {
                "id": 10,
                "title": "Python Research Engineer",
                "company": "Research Labs India",
                "location": "Hyderabad, India",
                "experience": "3-6 years",
                "skills": [
                    "Python", "Research", "Statistics", "Mathematics",
                    "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
                    "Publications", "Patents", "Jupyter", "Git"
                ],
                "soft_skills": ["Research mindset", "Innovation", "Presentation skills"],
                "tools": ["Jupyter", "LaTeX", "Research databases", "Git", "Docker"],
                "requirements": [
                    "Advanced degree preferred",
                    "Research publication experience",
                    "Algorithm development",
                    "Statistical analysis"
                ],
                "responsibilities": [
                    "Conduct cutting-edge research",
                    "Publish research papers",
                    "Develop novel algorithms",
                    "Collaborate with academia"
                ]
            }
        ]

class OCREngine:
    """Multi-engine OCR system"""
    
    def __init__(self):
        self.config = OCRConfig()
        self.easy_reader = None
        self.paddle_ocr = None
        self.trocr_processor = None
        self.trocr_model = None
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize OCR engines"""
        try:
            # Initialize EasyOCR
            self.easy_reader = easyocr.Reader(self.config.easyocr_langs)
            logger.info("EasyOCR initialized successfully")
            
            # Initialize PaddleOCR
            self.paddle_ocr = paddleocr.PaddleOCR(
                use_angle_cls=True,
                lang=self.config.paddleocr_lang,
                use_gpu=False
            )
            logger.info("PaddleOCR initialized successfully")
            
            # Initialize TrOCR
            self.trocr_processor = TrOCRProcessor.from_pretrained(self.config.trocr_model)
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained(self.config.trocr_model)
            logger.info("TrOCR initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing OCR engines: {e}")
    
    def extract_text_easyocr(self, image_path: str) -> str:
        """Extract text using EasyOCR"""
        try:
            results = self.easy_reader.readtext(image_path)
            text = ' '.join([result[1] for result in results])
            return text
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return ""
    
    def extract_text_paddleocr(self, image_path: str) -> str:
        """Extract text using PaddleOCR"""
        try:
            results = self.paddle_ocr.ocr(image_path, cls=True)
            text_parts = []
            for line in results:
                if line:
                    for item in line:
                        if len(item) > 1:
                            text_parts.append(item[1][0])
            return ' '.join(text_parts)
        except Exception as e:
            logger.error(f"PaddleOCR error: {e}")
            return ""
    
    def extract_text_trocr(self, image_path: str) -> str:
        """Extract text using TrOCR"""
        try:
            image = Image.open(image_path).convert('RGB')
            pixel_values = self.trocr_processor(images=image, return_tensors="pt").pixel_values
            generated_ids = self.trocr_model.generate(pixel_values)
            generated_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return generated_text
        except Exception as e:
            logger.error(f"TrOCR error: {e}")
            return ""
    
    def extract_text_multi_engine(self, image_path: str, doc_type: str = DocumentTypes.GENERAL) -> Dict[str, Any]:
        """Extract text using multiple OCR engines and combine results"""
        results = {}
        
        # Get results from all engines
        results['easyocr'] = self.extract_text_easyocr(image_path)
        results['paddleocr'] = self.extract_text_paddleocr(image_path)
        results['trocr'] = self.extract_text_trocr(image_path)
        
        # Choose best result based on document type and text length
        best_text = self._select_best_ocr_result(results, doc_type)
        
        return {
            'all_results': results,
            'best_result': best_text,
            'confidence_scores': self._calculate_confidence_scores(results)
        }
    
    def _select_best_ocr_result(self, results: Dict[str, str], doc_type: str) -> str:
        """Select the best OCR result based on document type and quality metrics"""
        # Simple heuristic: choose the longest non-empty result
        valid_results = {k: v for k, v in results.items() if v.strip()}
        
        if not valid_results:
            return ""
        
        # For different document types, prefer different engines
        if doc_type == DocumentTypes.HANDWRITTEN:
            # TrOCR is better for handwritten text
            if results['trocr'].strip():
                return results['trocr']
        elif doc_type in [DocumentTypes.INVOICE, DocumentTypes.CHEQUE, DocumentTypes.CHALLAN]:
            # PaddleOCR is often better for structured documents
            if results['paddleocr'].strip():
                return results['paddleocr']
        
        # Default: return the longest result
        return max(valid_results.items(), key=lambda x: len(x[1]))[1]
    
    def _calculate_confidence_scores(self, results: Dict[str, str]) -> Dict[str, float]:
        """Calculate confidence scores for OCR results"""
        scores = {}
        for engine, text in results.items():
            # Simple confidence based on text length and character diversity
            if not text.strip():
                scores[engine] = 0.0
            else:
                # Basic confidence calculation
                length_score = min(len(text) / 100, 1.0)  # Normalize by expected length
                char_diversity = len(set(text.lower())) / max(len(text), 1)
                scores[engine] = (length_score + char_diversity) / 2
        
        return scores

class TextProcessor:
    """Advanced text processing and information extraction"""
    
    def __init__(self):
        # Load spaCy model (download with: python -m spacy download en_core_web_sm)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Common skills database
        self.technical_skills = [
            "Python", "Java", "JavaScript", "C++", "C#", "Go", "Rust", "Ruby", "PHP", "Swift",
            "Django", "Flask", "FastAPI", "React", "Vue.js", "Angular", "Node.js", "Express",
            "SQL", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
            "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Terraform", "Ansible",
            "Git", "Jenkins", "CI/CD", "Linux", "REST API", "GraphQL", "Machine Learning",
            "Deep Learning", "TensorFlow", "PyTorch", "scikit-learn", "Pandas", "NumPy"
        ]
        
        self.soft_skills = [
            "Leadership", "Communication", "Problem solving", "Team collaboration",
            "Time management", "Adaptability", "Critical thinking", "Creativity",
            "Project management", "Agile", "Scrum", "Public speaking", "Negotiation"
        ]
    
    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract technical and soft skills from text"""
        text_lower = text.lower()
        
        # Extract technical skills
        found_technical = []
        for skill in self.technical_skills:
            if skill.lower() in text_lower:
                found_technical.append(skill)
        
        # Extract soft skills
        found_soft = []
        for skill in self.soft_skills:
            if skill.lower() in text_lower:
                found_soft.append(skill)
        
        return {
            "technical_skills": found_technical,
            "soft_skills": found_soft
        }
    
    def extract_experience(self, text: str) -> Dict[str, Any]:
        """Extract experience information"""
        # Pattern for experience (e.g., "5 years", "2-3 years", "3+ years")
        exp_patterns = [
            r'(\d+)\s*[-+]\s*(\d+)\s*years?',
            r'(\d+)\+?\s*years?',
            r'(\d+)\s*to\s*(\d+)\s*years?'
        ]
        
        experiences = []
        for pattern in exp_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            experiences.extend(matches)
        
        return {"experience_mentions": experiences}
    
    def extract_education(self, text: str) -> List[str]:
        """Extract education information"""
        education_keywords = [
            "bachelor", "master", "phd", "doctorate", "diploma", "certificate",
            "b.tech", "m.tech", "b.sc", "m.sc", "mba", "bca", "mca",
            "engineering", "computer science", "information technology"
        ]
        
        found_education = []
        text_lower = text.lower()
        
        for edu in education_keywords:
            if edu in text_lower:
                found_education.append(edu)
        
        return found_education
    
    def extract_contact_info(self, text: str) -> Dict[str, Any]:
        """Extract contact information"""
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        
        # Phone pattern (Indian format)
        phone_pattern = r'[\+]?[91]?[-.\s]?[6-9]\d{9}'
        phones = re.findall(phone_pattern, text)
        
        return {
            "emails": emails,
            "phones": phones
        }
    
    def extract_achievements(self, text: str) -> List[str]:
        """Extract achievements and awards"""
        achievement_keywords = [
            "award", "recognition", "achievement", "honor", "certification",
            "published", "patent", "winner", "recipient", "excellence"
        ]
        
        achievements = []
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in achievement_keywords):
                achievements.append(sentence.strip())
        
        return achievements

class JobMatcher:
    """Job matching and scoring system"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.job_descriptions = JobDescriptions.get_python_jobs()
    
    def calculate_skill_match_score(self, cv_skills: List[str], job_skills: List[str]) -> float:
        """Calculate skill match score between CV and job"""
        if not cv_skills or not job_skills:
            return 0.0
        
        cv_skills_lower = [skill.lower() for skill in cv_skills]
        job_skills_lower = [skill.lower() for skill in job_skills]
        
        matches = 0
        for job_skill in job_skills_lower:
            # Exact match
            if job_skill in cv_skills_lower:
                matches += 1
            else:
                # Fuzzy match
                best_match = process.extractOne(job_skill, cv_skills_lower)
                if best_match and best_match[1] > 80:  # 80% similarity threshold
                    matches += 0.8
        
        return matches / len(job_skills) if job_skills else 0.0
    
    def calculate_text_similarity(self, cv_text: str, job_text: str) -> float:
        """Calculate text similarity using TF-IDF"""
        try:
            # Combine texts for vectorization
            texts = [cv_text, job_text]
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except Exception as e:
            logger.error(f"Text similarity calculation error: {e}")
            return 0.0
    
    def match_jobs(self, cv_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Match CV against all job descriptions"""
        matches = []
        
        cv_technical_skills = cv_data.get('skills', {}).get('technical_skills', [])
        cv_soft_skills = cv_data.get('skills', {}).get('soft_skills', [])
        cv_text = cv_data.get('text', '')
        
        for job in self.job_descriptions:
            # Calculate various similarity scores
            technical_score = self.calculate_skill_match_score(cv_technical_skills, job['skills'])
            soft_skill_score = self.calculate_skill_match_score(cv_soft_skills, job['soft_skills'])
            tools_score = self.calculate_skill_match_score(cv_technical_skills, job['tools'])
            
            # Create job description text for similarity calculation
            job_text = f"{' '.join(job['skills'])} {' '.join(job['requirements'])} {' '.join(job['responsibilities'])}"
            text_similarity = self.calculate_text_similarity(cv_text, job_text)
            
            # Calculate weighted overall score
            overall_score = (
                technical_score * 0.4 +
                soft_skill_score * 0.2 +
                tools_score * 0.2 +
                text_similarity * 0.2
            )
            
            match_result = {
                "job_id": job['id'],
                "job_title": job['title'],
                "company": job['company'],
                "location": job['location'],
                "experience": job['experience'],
                "overall_score": round(overall_score * 100, 2),
                "technical_skill_match": round(technical_score * 100, 2),
                "soft_skill_match": round(soft_skill_score * 100, 2),
                "tools_match": round(tools_score * 100, 2),
                "text_similarity": round(text_similarity * 100, 2),
                "matched_skills": list(set(cv_technical_skills) & set([skill.lower() for skill in job['skills']])),
                "missing_skills": list(set([skill.lower() for skill in job['skills']]) - set([skill.lower() for skill in cv_technical_skills]))
            }
            
            matches.append(match_result)
        
        # Sort by overall score
        matches.sort(key=lambda x: x['overall_score'], reverse=True)
        return matches

class OllamaClient:
    """Client for Ollama local LLM"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "llama3.2:3b"
    
    def generate_text(self, prompt: str) -> str:
        """Generate text using Ollama"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return ""
        except Exception as e:
            logger.error(f"Ollama request error: {e}")
            return ""
    
    def enhance_cv_analysis(self, cv_text: str) -> str:
        """Enhance CV analysis using LLM"""
        prompt = f"""
        Analyze the following CV/Resume text and provide insights:
        
        CV Text:
        {cv_text}
        
        Please provide:
        1. Key strengths of the candidate
        2. Areas for improvement
        3. Suitable job roles
        4. Missing skills that are commonly required
        5. Overall assessment
        
        Keep the response concise and professional.
        """
        
        return self.generate_text(prompt)
    
    def generate_job_recommendations(self, cv_analysis: Dict[str, Any]) -> str:
        """Generate personalized job recommendations"""
        skills = cv_analysis.get('skills', {})
        technical_skills = ', '.join(skills.get('technical_skills', []))
        soft_skills = ', '.join(skills.get('soft_skills', []))
        
        prompt = f"""
        Based on a candidate's profile with the following skills:
        
        Technical Skills: {technical_skills}
        Soft Skills: {soft_skills}
        
        Provide personalized career advice including:
        1. Best matching job roles
        2. Skills to develop
        3. Career progression path
        4. Industry recommendations
        
        Keep it concise and actionable.
        """
        
        return self.generate_text(prompt)

# Pydantic models for API
class OCRResponse(BaseModel):
    text: str
    confidence_scores: Dict[str, float]
    extracted_info: Dict[str, Any]

class JobMatchResponse(BaseModel):
    matches: List[Dict[str, Any]]
    total_jobs: int
    best_match: Dict[str, Any]

class AnalysisResponse(BaseModel):
    cv_analysis: Dict[str, Any]
    job_matches: List[Dict[str, Any]]
    llm_insights: str
    recommendations: str

# FastAPI Application
app = FastAPI(
    title="OCR Job Matching System",
    description="Advanced OCR system with job matching and AI insights",
    version="1.0.0"
)

# Global instances
ocr_engine = OCREngine()
text_processor = TextProcessor()
job_matcher = JobMatcher()
ollama_client = OllamaClient()

@app.post("/ocr/extract", response_model=OCRResponse)
async def extract_text_from_image(
    file: UploadFile = File(...),
    doc_type: str = Form(default=DocumentTypes.GENERAL)
):
    """Extract text from uploaded image using multiple OCR engines"""
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Extract text using multiple OCR engines
        ocr_results = ocr_engine.extract_text_multi_engine(temp_path, doc_type)
        
        # Process extracted text
        extracted_info = {}
        if ocr_results['best_result']:
            skills = text_processor.extract_skills(ocr_results['best_result'])
            experience = text_processor.extract_experience(ocr_results['best_result'])
            education = text_processor.extract_education(ocr_results['best_result'])
            contact = text_processor.extract_contact_info(ocr_results['best_result'])
            achievements = text_processor.extract_achievements(ocr_results['best_result'])
            
            extracted_info = {
                "skills": skills,
                "experience": experience,
                "education": education,
                "contact": contact,
                "achievements": achievements
            }
        
        # Clean up temporary file
        os.remove(temp_path)
        
        return OCRResponse(
            text=ocr_results['best_result'],
            confidence_scores=ocr_results['confidence_scores'],
            extracted_info=extracted_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing error: {str(e)}")

@app.post("/jobs/match", response_model=JobMatchResponse)
async def match_jobs_with_cv(
    file: UploadFile = File(...),
    doc_type: str = Form(default=DocumentTypes.CV)
):
    """Match CV against available job descriptions"""
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Extract text and information
        ocr_results = ocr_engine.extract_text_multi_engine(temp_path, doc_type)
        
        cv_data = {
            "text": ocr_results['best_result'],
            "skills": text_processor.extract_skills(ocr_results['best_result']),
            "experience": text_processor.extract_experience(ocr_results['best_result']),
            "education": text_processor.extract_education(ocr_results['best_result'])
        }
        
        # Match jobs
        job_matches = job_matcher.match_jobs(cv_data)
        
        # Clean up
        os.remove(temp_path)
        
        return JobMatchResponse(
            matches=job_matches,
            total_jobs=len(job_matches),
            best_match=job_matches[0] if job_matches else {}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Job matching error: {str(e)}")

@app.post("/analysis/complete", response_model=AnalysisResponse)
async def complete_cv_analysis(
    file: UploadFile = File(...),
    doc_type: str = Form(default=DocumentTypes.CV),
    use_llm: bool = Form(default=True)
):
    """Complete CV analysis with OCR, job matching, and LLM insights"""
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Extract text and information
        ocr_results = ocr_engine.extract_text_multi_engine(temp_path, doc_type)
        
        # Process CV data
        skills = text_processor.extract_skills(ocr_results['best_result'])
        experience = text_processor.extract_experience(ocr_results['best_result'])
        education = text_processor.extract_education(ocr_results['best_result'])
        contact = text_processor.extract_contact_info(ocr_results['best_result'])
        achievements = text_processor.extract_achievements(ocr_results['best_result'])
        
        cv_data = {
            "text": ocr_results['best_result'],
            "skills": skills,
            "experience": experience,
            "education": education,
            "contact": contact,
            "achievements": achievements
        }
        
        # Match jobs
        job_matches = job_matcher.match_jobs(cv_data)
        
        # Get LLM insights if requested
        llm_insights = ""
        recommendations = ""
        
        if use_llm:
            llm_insights = ollama_client.enhance_cv_analysis(ocr_results['best_result'])
            recommendations = ollama_client.generate_job_recommendations(cv_data)
        
        # Clean up
        os.remove(temp_path)
        
        return AnalysisResponse(
            cv_analysis=cv_data,
            job_matches=job_matches[:5],  # Top 5 matches
            llm_insights=llm_insights,
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.get("/jobs/all")
async def get_all_jobs():
    """Get all available job descriptions"""
    return {"jobs": JobDescriptions.get_python_jobs()}

@app.get("/jobs/{job_id}")
async def get_job_by_id(job_id: int):
    """Get specific job description by ID"""
    jobs = JobDescriptions.get_python_jobs()
    job = next((j for j in jobs if j['id'] == job_id), None)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {"job": job}

@app.post("/skills/extract")
async def extract_skills_from_text(text: str = Form(...)):
    """Extract skills from provided text"""
    skills = text_processor.extract_skills(text)
    return {"skills": skills}

@app.post("/llm/analyze")
async def analyze_text_with_llm(
    text: str = Form(...),
    analysis_type: str = Form(default="general")
):
    """Analyze text using local LLM"""
    try:
        if analysis_type == "cv":
            result = ollama_client.enhance_cv_analysis(text)
        else:
            result = ollama_client.generate_text(f"Analyze the following text: {text}")
        
        return {"analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM analysis error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ocr_engines": ["EasyOCR", "PaddleOCR", "TrOCR"],
        "supported_documents": [
            DocumentTypes.RESUME,
            DocumentTypes.CV,
            DocumentTypes.MARKSHEET,
            DocumentTypes.INVOICE,
            DocumentTypes.CHEQUE,
            DocumentTypes.CHALLAN,
            DocumentTypes.HANDWRITTEN
        ],
        "llm_model": "llama3.2:3b"
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "OCR Job Matching System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "Endpoint not found"}
    )

@app.exception_handler(500)
async def server_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error"}
    )

if __name__ == "__main__":
    print("Starting OCR Job Matching System...")
    print("Make sure to install required dependencies:")
    print("pip install fastapi uvicorn easyocr paddlepaddle paddleocr transformers torch")
    print("pip install scikit-learn spacy fuzzywuzzy python-levenshtein pillow opencv-python")
    print("python -m spacy download en_core_web_sm")
    print("Install Ollama and run: ollama pull llama3.2:3b")
    print("\nStarting server on http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
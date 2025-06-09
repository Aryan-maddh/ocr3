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
            
            # Debug: Print extracted text for debugging
            print(f"Extracted text preview: {result['extracted_text'][:500]}...")
            
            # Auto-detect document type if needed
            if doc_type == "auto":
                detected_type = self.detect_document_type(result["extracted_text"])
                result["document_type"] = detected_type
                print(f"Auto-detected document type: {detected_type}")
            
            # Parse structured data based on document type
            result["structured_data"] = self.parse_structured_data(result["extracted_text"], result["document_type"])
            result["confidence_score"] = self.calculate_confidence(result["extracted_text"])
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            result["error"] = str(e)
        
        return result
    
    def detect_document_type(self, text: str) -> str:
        """Detect document type based on content"""
        if not text or len(text.strip()) < 10:
            return "unknown"
        
        text_lower = text.lower()
        
        # Define keyword patterns for different document types
        resume_keywords = [
            'experience', 'education', 'skills', 'work experience', 'employment',
            'qualifications', 'achievements', 'career', 'resume', 'cv', 'curriculum vitae',
            'profile', 'summary', 'objective', 'projects', 'internship', 'bachelor',
            'master', 'degree', 'certification', 'languages', 'technical skills'
        ]
        
        invoice_keywords = [
            'invoice', 'bill', 'amount', 'total', 'payment', 'due', 'tax', 'gst',
            'invoice number', 'date', 'vendor', 'customer', 'quantity', 'rate',
            'subtotal', 'discount', 'billing', 'invoice no'
        ]
        
        marksheet_keywords = [
            'marks', 'grade', 'percentage', 'gpa', 'cgpa', 'result', 'exam',
            'semester', 'subject', 'score', 'transcript', 'marksheet', 'mark sheet',
            'academic', 'university', 'college', 'student', 'roll number',
            'registration number', 'course', 'pass', 'fail'
        ]
        
        cheque_keywords = [
            'pay', 'cheque', 'check', 'bank', 'account', 'rupees', 'only',
            'signature', 'date', 'amount', 'payee', 'drawer', 'branch',
            'micr', 'ifsc', 'cheque no', 'cheque number'
        ]
        
        hall_ticket_keywords = [
            'hall ticket', 'admit card', 'admission', 'exam', 'examination',
            'roll number', 'seat number', 'center', 'time', 'date',
            'instructions', 'candidate', 'subject code'
        ]
        
        # Count matches for each document type
        resume_score = sum(1 for keyword in resume_keywords if keyword in text_lower)
        invoice_score = sum(1 for keyword in invoice_keywords if keyword in text_lower)
        marksheet_score = sum(1 for keyword in marksheet_keywords if keyword in text_lower)
        cheque_score = sum(1 for keyword in cheque_keywords if keyword in text_lower)
        hall_ticket_score = sum(1 for keyword in hall_ticket_keywords if keyword in text_lower)
        
        # Debug: Print scores
        print(f"Document type scores - Resume: {resume_score}, Invoice: {invoice_score}, "
              f"Marksheet: {marksheet_score}, Cheque: {cheque_score}, Hall Ticket: {hall_ticket_score}")
        
        # Determine document type based on highest score
        scores = {
            'resume': resume_score,
            'invoice': invoice_score,
            'marksheet': marksheet_score,
            'cheque': cheque_score,
            'hall_ticket': hall_ticket_score
        }
        
        max_score = max(scores.values())
        if max_score >= 2:  # Minimum threshold
            detected_type = max(scores, key=scores.get)
            return detected_type
        
        # If no clear match, try pattern-based detection
        if re.search(r'\b(?:phone|email|experience|skills)\b', text_lower):
            return 'resume'
        elif re.search(r'\b(?:invoice|bill|amount|total)\b', text_lower):
            return 'invoice'
        elif re.search(r'\b(?:marks|grade|percentage|gpa)\b', text_lower):
            return 'marksheet'
        elif re.search(r'\b(?:pay|rupees|bank)\b', text_lower):
            return 'cheque'
        elif re.search(r'\b(?:hall ticket|admit card|exam)\b', text_lower):
            return 'hall_ticket'
        
        return "unknown"
    
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
        elif doc_type == "hall_ticket":
            structured_data = self.parse_hall_ticket(text)
        else:
            # For unknown types, still try to extract basic information
            structured_data = self.extract_basic_info(text)
        
        return structured_data
    
    def extract_basic_info(self, text: str) -> Dict[str, Any]:
        """Extract basic information from any document"""
        basic_info = {
            "document_type": "unknown",
            "extracted_entities": [],
            "emails": [],
            "phone_numbers": [],
            "dates": [],
            "numbers": [],
            "text_preview": text[:500] if text else ""
        }
        
        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        basic_info["emails"] = re.findall(email_pattern, text)
        
        # Extract phone numbers
        phone_pattern = r'(\+\d{1,3}[-.\s]?)?\(?\d{3,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{4,6}'
        basic_info["phone_numbers"] = re.findall(phone_pattern, text)
        
        # Extract dates
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{2,4}',
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{2,4}'
        ]
        for pattern in date_patterns:
            basic_info["dates"].extend(re.findall(pattern, text, re.IGNORECASE))
        
        # Extract numbers (amounts, IDs, etc.)
        number_pattern = r'\b\d{4,}\b'
        basic_info["numbers"] = re.findall(number_pattern, text)
        
        # Use NER if available
        if self.ner_pipeline:
            try:
                entities = self.ner_pipeline(text[:1000])  # Limit text length for NER
                basic_info["extracted_entities"] = [
                    {"text": ent["word"], "label": ent["entity_group"], "confidence": ent["score"]}
                    for ent in entities if ent["score"] > 0.5
                ]
            except Exception as e:
                logger.error(f"NER processing failed: {e}")
        
        return basic_info
    
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
        phone_pattern = r'(\+\d{1,3}[-.\s]?)?\(?\d{3,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{4,6}'
        phones = re.findall(phone_pattern, text)
        if phones:
            resume_data["personal_info"]["phone"] = phones[0]
        
        # Extract name (first few words that are likely names)
        lines = text.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if line and not any(char.isdigit() for char in line) and len(line.split()) <= 4:
                # Skip common resume headers
                if not any(word in line.lower() for word in ['resume', 'cv', 'curriculum', 'profile']):
                    resume_data["personal_info"]["name"] = line
                    break
        
        # Extract skills using NER and keyword matching
        if self.ner_pipeline:
            try:
                entities = self.ner_pipeline(text)
                for entity in entities:
                    if entity['entity_group'] in ['MISC', 'ORG']:
                        resume_data["skills"]["technical"].append(entity['word'])
            except Exception as e:
                logger.error(f"NER processing failed: {e}")
        
        # Common technical skills
        tech_skills = [
            'python', 'java', 'javascript', 'react', 'node.js', 'sql', 'mongodb',
            'docker', 'kubernetes', 'aws', 'azure', 'git', 'tensorflow', 'pytorch',
            'machine learning', 'data science', 'artificial intelligence', 'html',
            'css', 'php', 'c++', 'c#', 'ruby', 'go', 'scala', 'r', 'matlab',
            'mysql', 'postgresql', 'oracle', 'firebase', 'angular', 'vue.js'
        ]
        
        text_lower = text.lower()
        for skill in tech_skills:
            if skill in text_lower:
                if skill.title() not in resume_data["skills"]["technical"]:
                    resume_data["skills"]["technical"].append(skill.title())
        
        # Extract years of experience
        exp_patterns = [
            r'(\d+)(?:\+)?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
            r'experience[:\s]*(\d+)(?:\+)?\s*(?:years?|yrs?)',
            r'(\d+)(?:\+)?\s*(?:years?|yrs?)\s*experience'
        ]
        
        for pattern in exp_patterns:
            exp_matches = re.findall(pattern, text.lower())
            if exp_matches:
                resume_data["experience_years"] = max([int(x) for x in exp_matches])
                break
        
        # Extract education
        education_keywords = ['bachelor', 'master', 'phd', 'degree', 'diploma', 'certification', 'university', 'college']
        education_lines = []
        for line in text.split('\n'):
            if any(keyword in line.lower() for keyword in education_keywords):
                education_lines.append(line.strip())
        
        resume_data["education"] = education_lines[:3]  # Top 3 education entries
        
        return resume_data
    
    def parse_invoice(self, text: str) -> Dict[str, Any]:
        """Parse invoice data"""
        invoice_data = {
            "invoice_number": "",
            "date": "",
            "amount": "",
            "vendor": "",
            "items": [],
            "tax_amount": "",
            "total_amount": ""
        }
        
        # Extract invoice number
        inv_patterns = [
            r'(?:invoice|inv)(?:\s*#|\s*no\.?|\s*number)?\s*:?\s*([A-Z0-9-]+)',
            r'(?:bill|receipt)\s*(?:no|number|#)?\s*:?\s*([A-Z0-9-]+)'
        ]
        
        for pattern in inv_patterns:
            inv_match = re.search(pattern, text, re.IGNORECASE)
            if inv_match:
                invoice_data["invoice_number"] = inv_match.group(1)
                break
        
        # Extract amounts
        amount_patterns = [
            r'(?:total|grand total|amount)\s*:?\s*[$₹€£]?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'[$₹€£]\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'(?:rs\.?|rupees)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)'
        ]
        
        for pattern in amount_patterns:
            amount_match = re.search(pattern, text, re.IGNORECASE)
            if amount_match:
                invoice_data["total_amount"] = amount_match.group(1)
                break
        
        # Extract dates
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{2,4}'
        ]
        
        for pattern in date_patterns:
            date_match = re.search(pattern, text, re.IGNORECASE)
            if date_match:
                invoice_data["date"] = date_match.group(0)
                break
        
        return invoice_data
    
    def parse_marksheet(self, text: str) -> Dict[str, Any]:
        """Parse marksheet/transcript data"""
        marksheet_data = {
            "student_name": "",
            "roll_number": "",
            "subjects": [],
            "grades": [],
            "gpa": "",
            "percentage": "",
            "institution": "",
            "semester": ""
        }
        
        # Extract percentage
        perc_patterns = [
            r'(?:percentage|%)\s*:?\s*(\d+(?:\.\d+)?)\s*%?',
            r'(\d+(?:\.\d+)?)\s*%',
            r'(?:total|overall)\s*(?:percentage|%)\s*:?\s*(\d+(?:\.\d+)?)'
        ]
        
        for pattern in perc_patterns:
            perc_matches = re.findall(pattern, text, re.IGNORECASE)
            if perc_matches:
                percentages = [float(p) for p in perc_matches if p]
                marksheet_data["percentage"] = str(max(percentages))
                break
        
        # Extract GPA
        gpa_patterns = [
            r'(?:gpa|cgpa)\s*:?\s*(\d+\.\d+)',
            r'(?:grade point|point average)\s*:?\s*(\d+\.\d+)'
        ]
        
        for pattern in gpa_patterns:
            gpa_match = re.search(pattern, text, re.IGNORECASE)
            if gpa_match:
                marksheet_data["gpa"] = gpa_match.group(1)
                break
        
        # Extract roll number
        roll_patterns = [
            r'(?:roll|reg|registration)\s*(?:no|number|#)\s*:?\s*([A-Z0-9]+)',
            r'(?:student|enrollment)\s*(?:id|number)\s*:?\s*([A-Z0-9]+)'
        ]
        
        for pattern in roll_patterns:
            roll_match = re.search(pattern, text, re.IGNORECASE)
            if roll_match:
                marksheet_data["roll_number"] = roll_match.group(1)
                break
        
        return marksheet_data
    
    def parse_cheque(self, text: str) -> Dict[str, Any]:
        """Parse cheque data"""
        cheque_data = {
            "amount_words": "",
            "amount_figures": "",
            "payee": "",
            "date": "",
            "cheque_number": "",
            "bank_name": ""
        }
        
        # Extract amount in figures
        amount_patterns = [
            r'₹\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'rs\.?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'rupees\s*(\d+(?:,\d{3})*(?:\.\d{2})?)'
        ]
        
        for pattern in amount_patterns:
            amount_match = re.search(pattern, text, re.IGNORECASE)
            if amount_match:
                cheque_data["amount_figures"] = amount_match.group(1)
                break
        
        # Extract cheque number
        cheque_no_pattern = r'(?:cheque|check)\s*(?:no|number|#)\s*:?\s*(\d+)'
        cheque_match = re.search(cheque_no_pattern, text, re.IGNORECASE)
        if cheque_match:
            cheque_data["cheque_number"] = cheque_match.group(1)
        
        return cheque_data
    
    def parse_hall_ticket(self, text: str) -> Dict[str, Any]:
        """Parse hall ticket/admit card data"""
        hall_ticket_data = {
            "candidate_name": "",
            "roll_number": "",
            "exam_name": "",
            "exam_date": "",
            "exam_time": "",
            "center": "",
            "seat_number": ""
        }
        
        # Extract roll/seat number
        roll_patterns = [
            r'(?:roll|seat|reg)\s*(?:no|number)\s*:?\s*([A-Z0-9]+)',
            r'(?:candidate|student)\s*(?:id|number)\s*:?\s*([A-Z0-9]+)'
        ]
        
        for pattern in roll_patterns:
            roll_match = re.search(pattern, text, re.IGNORECASE)
            if roll_match:
                hall_ticket_data["roll_number"] = roll_match.group(1)
                break
        
        # Extract exam name
        exam_patterns = [
            r'(?:examination|exam)\s*:?\s*([A-Z\s]+)',
            r'([A-Z\s]+)\s*examination',
            r'([A-Z\s]+)\s*exam'
        ]
        
        for pattern in exam_patterns:
            exam_match = re.search(pattern, text, re.IGNORECASE)
            if exam_match:
                hall_ticket_data["exam_name"] = exam_match.group(1).strip()
                break
        
        return hall_ticket_data
    
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

# [Rest of the JobMatcher class and FastAPI code remains the same...]
# The JobMatcher class and FastAPI endpoints don't need changes for the OCR fix

class JobMatcher:
    """Job matching system with skill analysis"""
    
    def __init__(self):
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.job_descriptions = self.create_job_descriptions()
    
    def create_job_descriptions(self) -> List[Dict[str, Any]]:
        # ... (same as original code)
        pass
    
    # ... (rest of JobMatcher methods remain the same)

# Example usage and testing functions
def test_ocr_system():
    """Test the OCR system with document type detection"""
    print("Testing Enhanced OCR System...")
    
    processor = DocumentProcessor()
    
    # Test with sample text snippets
    test_cases = [
        {
            "text": "John Doe\nSoftware Engineer\nPhone: +91-9876543210\nEmail: john@example.com\nExperience: 5 years\nSkills: Python, Java, AWS",
            "expected_type": "resume"
        },
        {
            "text": "Invoice No: INV-001\nDate: 15/01/2024\nTotal Amount: Rs. 5000\nTax: 18%\nGrand Total: Rs. 5900",
            "expected_type": "invoice"
        },
        {
            "text": "Student Name: Jane Smith\nRoll Number: 12345\nPercentage: 85.5%\nGPA: 8.5\nResult: Pass",
            "expected_type": "marksheet"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        detected_type = processor.detect_document_type(test_case["text"])
        print(f"Expected: {test_case['expected_type']}, Detected: {detected_type}")
        
        structured_data = processor.parse_structured_data(test_case["text"], detected_type)
        print(f"Structured Data: {structured_data}")

if __name__ == "__main__":
    test_ocr_system()

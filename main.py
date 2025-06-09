import os
import json
import re
import csv
import io
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
    pipeline
)
import torch

# FastAPI and related
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

# NLP and similarity
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Advanced OCR Document Processing System",
    description="OCR system with document type detection and structured data extraction",
    version="1.0.0"
)

class ProcessingResult(BaseModel):
    file_name: str
    document_type: str
    extracted_text: str
    structured_data: Dict[str, Any]
    confidence_score: float
    processing_time: float
    error: Optional[str] = None

class DocumentProcessor:
    """Advanced OCR processor using multiple Hugging Face models"""
    
    def __init__(self):
        self.setup_models()
        
    def setup_models(self):
        """Initialize all required models locally"""
        try:
            # TrOCR for printed text (lighter model for faster processing)
            print("Loading TrOCR models...")
            self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
            
            # For handwritten text
            self.trocr_hw_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            self.trocr_hw_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
            
            # Sentence transformer for semantic similarity
            print("Loading sentence transformer...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Fallback to Tesseract only if models fail to load
            self.trocr_processor = None
            self.trocr_model = None
            self.trocr_hw_processor = None
            self.trocr_hw_model = None
            self.sentence_model = None
            logger.warning("Using Tesseract OCR only due to model loading failure")
    
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
            if not self.trocr_processor or not self.trocr_model:
                return ""
                
            if handwritten and self.trocr_hw_processor and self.trocr_hw_model:
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
        start_time = datetime.now()
        
        result = {
            "file_path": file_path,
            "document_type": doc_type,
            "extracted_text": "",
            "structured_data": {},
            "confidence_score": 0.0,
            "processing_time": 0.0
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
            
            # Auto-detect document type if needed
            if doc_type == "auto":
                detected_type = self.detect_document_type(result["extracted_text"])
                result["document_type"] = detected_type
            
            # Parse structured data based on document type
            result["structured_data"] = self.parse_structured_data(result["extracted_text"], result["document_type"])
            result["confidence_score"] = self.calculate_confidence(result["extracted_text"])
            
            # Calculate processing time
            end_time = datetime.now()
            result["processing_time"] = (end_time - start_time).total_seconds()
            
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
        
        return "unknown"
    
    def detect_handwriting(self, image: Image.Image) -> bool:
        """Simple handwriting detection"""
        return False  # Default to printed text for now
    
    def parse_structured_data(self, text: str, doc_type: str) -> Dict[str, Any]:
        """Parse structured data based on document type"""
        if doc_type in ["resume", "cv"]:
            return self.parse_resume(text)
        elif doc_type == "invoice":
            return self.parse_invoice(text)
        elif doc_type == "marksheet":
            return self.parse_marksheet(text)
        elif doc_type == "cheque":
            return self.parse_cheque(text)
        elif doc_type == "hall_ticket":
            return self.parse_hall_ticket(text)
        else:
            return self.extract_basic_info(text)
    
    def extract_basic_info(self, text: str) -> Dict[str, Any]:
        """Extract basic information from any document"""
        basic_info = {
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
            r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{2,4}'
        ]
        for pattern in date_patterns:
            basic_info["dates"].extend(re.findall(pattern, text, re.IGNORECASE))
        
        # Extract numbers
        number_pattern = r'\b\d{4,}\b'
        basic_info["numbers"] = re.findall(number_pattern, text)
        
        return basic_info
    
    def parse_resume(self, text: str) -> Dict[str, Any]:
        """Parse resume/CV data"""
        resume_data = {
            "name": "",
            "email": "",
            "phone": "",
            "skills": [],
            "experience_years": "",
            "education": []
        }
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            resume_data["email"] = emails[0]
        
        # Extract phone numbers
        phone_pattern = r'(\+\d{1,3}[-.\s]?)?\(?\d{3,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{4,6}'
        phones = re.findall(phone_pattern, text)
        if phones:
            resume_data["phone"] = phones[0]
        
        # Extract name (first non-empty line that doesn't contain numbers)
        lines = text.split('\n')
        for line in lines[:5]:
            line = line.strip()
            if line and not any(char.isdigit() for char in line) and len(line.split()) <= 4:
                if not any(word in line.lower() for word in ['resume', 'cv', 'curriculum', 'profile']):
                    resume_data["name"] = line
                    break
        
        # Extract technical skills
        tech_skills = [
            'python', 'java', 'javascript', 'react', 'node.js', 'sql', 'mongodb',
            'docker', 'kubernetes', 'aws', 'azure', 'git', 'tensorflow', 'pytorch',
            'machine learning', 'data science', 'html', 'css', 'php', 'c++', 'c#'
        ]
        
        text_lower = text.lower()
        found_skills = []
        for skill in tech_skills:
            if skill in text_lower:
                found_skills.append(skill.title())
        resume_data["skills"] = found_skills
        
        # Extract years of experience
        exp_patterns = [
            r'(\d+)(?:\+)?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
            r'experience[:\s]*(\d+)(?:\+)?\s*(?:years?|yrs?)'
        ]
        
        for pattern in exp_patterns:
            exp_matches = re.findall(pattern, text.lower())
            if exp_matches:
                resume_data["experience_years"] = max([int(x) for x in exp_matches])
                break
        
        return resume_data
    
    def parse_invoice(self, text: str) -> Dict[str, Any]:
        """Parse invoice data"""
        invoice_data = {
            "invoice_number": "",
            "date": "",
            "total_amount": "",
            "vendor": ""
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
            "percentage": "",
            "gpa": "",
            "institution": ""
        }
        
        # Extract percentage
        perc_patterns = [
            r'(?:percentage|%)\s*:?\s*(\d+(?:\.\d+)?)\s*%?',
            r'(\d+(?:\.\d+)?)\s*%'
        ]
        
        for pattern in perc_patterns:
            perc_matches = re.findall(pattern, text, re.IGNORECASE)
            if perc_matches:
                percentages = [float(p) for p in perc_matches if p]
                if percentages:
                    marksheet_data["percentage"] = str(max(percentages))
                break
        
        # Extract GPA
        gpa_patterns = [
            r'(?:gpa|cgpa)\s*:?\s*(\d+\.\d+)'
        ]
        
        for pattern in gpa_patterns:
            gpa_match = re.search(pattern, text, re.IGNORECASE)
            if gpa_match:
                marksheet_data["gpa"] = gpa_match.group(1)
                break
        
        # Extract roll number
        roll_patterns = [
            r'(?:roll|reg|registration)\s*(?:no|number|#)\s*:?\s*([A-Z0-9]+)'
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
            "amount_figures": "",
            "payee": "",
            "date": "",
            "cheque_number": ""
        }
        
        # Extract amount in figures
        amount_patterns = [
            r'₹\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'rs\.?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)'
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
            "center": ""
        }
        
        # Extract roll number
        roll_patterns = [
            r'(?:roll|seat|reg)\s*(?:no|number)\s*:?\s*([A-Z0-9]+)'
        ]
        
        for pattern in roll_patterns:
            roll_match = re.search(pattern, text, re.IGNORECASE)
            if roll_match:
                hall_ticket_data["roll_number"] = roll_match.group(1)
                break
        
        return hall_ticket_data
    
    def calculate_confidence(self, text: str) -> float:
        """Calculate confidence score based on text quality"""
        if not text.strip():
            return 0.0
        
        score = 0.0
        
        # Length factor
        if len(text) > 100:
            score += 0.4
        elif len(text) > 50:
            score += 0.3
        else:
            score += 0.2
        
        # Word count factor
        words = text.split()
        if len(words) > 20:
            score += 0.4
        elif len(words) > 10:
            score += 0.3
        else:
            score += 0.2
        
        # Character variety
        unique_chars = len(set(text.lower()))
        if unique_chars > 20:
            score += 0.2
        else:
            score += 0.1
        
        return min(score, 1.0)

# Initialize the processor
processor = DocumentProcessor()

# FastAPI Endpoints
@app.get("/")
async def root():
    return {"message": "Advanced OCR Document Processing System", "version": "1.0.0"}

@app.post("/process-document/", response_model=ProcessingResult)
async def process_document(
    file: UploadFile = File(...),
    document_type: str = Form("auto")
):
    """Process uploaded document and extract structured data"""
    
    # Validate file type
    allowed_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Save uploaded file temporarily
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        temp_file_path = temp_dir / file.filename
        
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the document
        result = processor.process_document(str(temp_file_path), document_type)
        
        # Clean up temp file
        temp_file_path.unlink()
        
        # Return structured response
        return ProcessingResult(
            file_name=file.filename,
            document_type=result.get("document_type", "unknown"),
            extracted_text=result.get("extracted_text", ""),
            structured_data=result.get("structured_data", {}),
            confidence_score=result.get("confidence_score", 0.0),
            processing_time=result.get("processing_time", 0.0),
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-document-csv/")
async def process_document_csv(
    file: UploadFile = File(...),
    document_type: str = Form("auto")
):
    """Process uploaded document and return results as CSV"""
    
    # Validate file type
    allowed_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Save uploaded file temporarily
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        temp_file_path = temp_dir / file.filename
        
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the document
        result = processor.process_document(str(temp_file_path), document_type)
        
        # Clean up temp file
        temp_file_path.unlink()
        
        # Create CSV data
        csv_data = create_csv_from_result(file.filename, result)
        
        # Return CSV as streaming response
        return StreamingResponse(
            io.StringIO(csv_data),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={Path(file.filename).stem}_processed.csv"}
        )
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def create_csv_from_result(filename: str, result: Dict[str, Any]) -> str:
    """Convert processing result to CSV format"""
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write headers
    writer.writerow([
        'filename', 'document_type', 'confidence_score', 'processing_time',
        'extracted_text_preview', 'field_name', 'field_value'
    ])
    
    # Basic information
    basic_info = [
        filename,
        result.get('document_type', ''),
        result.get('confidence_score', ''),
        result.get('processing_time', ''),
        result.get('extracted_text', '')[:200] + '...' if result.get('extracted_text', '') else ''
    ]
    
    # Write structured data
    structured_data = result.get('structured_data', {})
    
    if structured_data:
        for field_name, field_value in structured_data.items():
            if isinstance(field_value, (list, dict)):
                field_value = str(field_value)
            
            row = basic_info + [field_name, field_value]
            writer.writerow(row)
    else:
        # If no structured data, write basic info only
        row = basic_info + ['', '']
        writer.writerow(row)
    
    return output.getvalue()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/supported-formats")
async def supported_formats():
    """Get list of supported file formats"""
    return {
        "supported_formats": [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"],
        "document_types": ["resume", "invoice", "marksheet", "cheque", "hall_ticket", "auto"]
    }

if __name__ == "__main__":
    print("Starting Advanced OCR Document Processing System...")
    print("Swagger UI will be available at: http://localhost:8000/docs")
    print("ReDoc will be available at: http://localhost:8000/redoc")
    
    uvicorn.run(
        app, 
        host="0.0.0.0",
        port=8000,
        reload=True
    )

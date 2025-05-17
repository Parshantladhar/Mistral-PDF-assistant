"""
Document processing utilities for Mistral Docs Assistant.
"""
import os
import io
import logging
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from langdetect import detect

# For PDF processing
from PyPDF2 import PdfReader

# For DOCX processing
import docx2txt

# For text analysis
import re
from collections import Counter

# Configure logging
logger: logging.Logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Metadata for a processed document."""
    filename: str
    file_type: str
    file_size: int  # in bytes
    char_count: int
    word_count: int
    processed_at: datetime
    page_count: Optional[int] = None
    language: Optional[str] = None
    document_hash: Optional[str] = None

class DocumentProcessor:
    """Process different document types and extract text."""
    
    SUPPORTED_EXTENSIONS: Dict[str, str] = {
        '.pdf': 'application/pdf',
        '.txt': 'text/plain',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    }
    
    def __init__(self):
        self.processors: Dict[str, callable] = {
            '.pdf': self._process_pdf,
            '.txt': self._process_txt,
            '.docx': self._process_docx,
        }
    
    def is_supported(self, filename: str) -> bool:
        """Check if the file type is supported."""
        ext: str = os.path.splitext(filename)[1].lower()
        return ext in self.SUPPORTED_EXTENSIONS
    
    def process_document(self, 
                        file_obj: Any, 
                        filename: str) -> Tuple[str, DocumentMetadata]:
        """Process a document and return text and metadata."""
        if not self.is_supported(filename):
            logger.error(f"Unsupported file type: {filename}")
            raise ValueError(f"Unsupported file type: {filename}")
        
        ext: str = os.path.splitext(filename)[1].lower()
        processor: callable = self.processors.get(ext)
        
        if not processor:
            logger.error(f"No processor found for {ext}")
            raise ValueError(f"No processor found for {ext}")
        
        # Get file size
        file_obj.seek(0, os.SEEK_END)
        file_size: int = file_obj.tell()
        file_obj.seek(0)
        
        if file_size == 0:
            logger.warning(f"Empty file: {filename}")
            text = "ERROR: Empty file"
            metadata = DocumentMetadata(
                filename=filename,
                file_type=self.SUPPORTED_EXTENSIONS.get(ext, 'unknown'),
                file_size=0,
                char_count=0,
                word_count=0,
                page_count=1,
                processed_at=datetime.now(),
                document_hash="",
                language=None
            )
            return text, metadata
        
        # Process the document
        try:
            text, page_count = processor(file_obj)
        except Exception as e:
            logger.error(f"Failed to process {filename}: {str(e)}")
            text = f"ERROR: Failed to extract text - {str(e)}"
            page_count = 1
        
        # Calculate document hash
        file_obj.seek(0)
        file_hash: str = hashlib.md5(file_obj.read()).hexdigest()
        file_obj.seek(0)
        
        # Calculate stats
        word_count: int = len(re.findall(r'\b\w+\b', text)) if not text.startswith("ERROR:") else 0
        
        # Detect language
        language: Optional[str] = None
        if text and not text.startswith("ERROR:") and text.strip():
            try:
                language = detect(text)
            except Exception as e:
                logger.warning(f"Language detection failed for {filename}: {str(e)}")
        
        # Create metadata
        metadata = DocumentMetadata(
            filename=filename,
            file_type=self.SUPPORTED_EXTENSIONS.get(ext, 'unknown'),
            file_size=file_size,
            char_count=len(text),
            word_count=word_count,
            page_count=page_count,
            processed_at=datetime.now(),
            document_hash=file_hash,
            language=language
        )
        
        return text, metadata
    
    def _process_pdf(self, file_obj: io.BytesIO) -> Tuple[str, int]:
        """Extract text from PDF document."""
        try:
            pdf_reader = PdfReader(file_obj)
            page_count: int = len(pdf_reader.pages)
            
            text: str = ""
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n\n"
                else:
                    logger.warning("Empty text extracted from PDF page")
            
            if not text.strip():
                logger.warning("No text extracted from PDF")
                text = "ERROR: No text could be extracted from the PDF"
                page_count = 1
            
            return text, page_count
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
    
    def _process_txt(self, file_obj: io.BytesIO) -> Tuple[str, int]:
        """Extract text from TXT document."""
        try:
            text: str = file_obj.read().decode('utf-8', errors='ignore')
            if not text.strip():
                logger.warning("No text extracted from TXT")
                text = "ERROR: No text could be extracted from the TXT file"
            # Count pages by assuming a page is ~3000 characters
            page_count: int = max(1, len(text) // 3000)
            return text, page_count
        except Exception as e:
            logger.error(f"Error processing TXT: {str(e)}")
            raise
    
    def _process_docx(self, file_obj: io.BytesIO) -> Tuple[str, int]:
        """Extract text from DOCX document."""
        try:
            text: str = docx2txt.process(file_obj)
            if not text.strip():
                logger.warning("No text extracted from DOCX")
                text = "ERROR: No text could be extracted from the DOCX file"
            # docx2txt doesn't provide page count, so we estimate
            page_count: int = max(1, len(text) // 3000)
            return text, page_count
        except Exception as e:
            logger.error(f"Error processing DOCX: {str(e)}")
            raise

def analyze_text(text: str) -> Dict[str, Any]:
    """Analyze text and return statistics."""
    if not text or text.startswith("ERROR:"):
        return {
            "word_count": 0,
            "sentence_count": 0,
            "avg_sentence_length": 0,
            "common_words": [],
            "reading_time_minutes": 0,
            "char_count": len(text)
        }
    
    words: List[str] = re.findall(r'\b\w+\b', text.lower())
    sentences: List[str] = re.split(r'[.!?]+', text)
    
    # Count words
    word_count: int = len(words)
    
    # Count sentences (excluding empty)
    sentence_count: int = sum(1 for s in sentences if s.strip())
    
    # Calculate average sentence length
    avg_sentence_length: float = word_count / max(1, sentence_count)
    
    # Find most common words
    common_words: List[Tuple[str, int]] = Counter(words).most_common(10)
    
    # Estimate reading time (average reading speed: 200 words per minute)
    reading_time_minutes: float = word_count / 200
    
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": round(avg_sentence_length, 1),
        "common_words": common_words,
        "reading_time_minutes": round(reading_time_minutes, 1),
        "char_count": len(text)
    }

def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """Extract potential keywords from text using regex and frequency analysis."""
    if not text or text.startswith("ERROR:"):
        logger.warning("No valid text for keyword extraction")
        return []
    
    try:
        words = re.findall(r'\b[a-zA-Z]{3,15}\b', text.lower())
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                      'be', 'been', 'being', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
        filtered_words = [word for word in words if word not in stop_words]
        word_freq = Counter(filtered_words)
        keywords = [word for word, _ in word_freq.most_common(top_n)]
        logger.info(f"Extracted {len(keywords)} keywords")
        return keywords
    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}")
        return []
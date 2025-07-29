import fitz  # PyMuPDF
import re
def extract_text_from_pdf(pdf_path):
    """Extract and clean text from a resume PDF."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                page_text = page.get_text()
                text += page_text + " "
        
        # Clean up common PDF extraction issues
        text = text.replace('\n', ' ')  # Replace newlines with spaces
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        text = text.strip()
        
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def extract_text_from_txt(txt_path):
    """Extract text from a job description text file."""
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading JD: {e}")
        return ""



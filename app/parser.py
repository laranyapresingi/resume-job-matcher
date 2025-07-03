import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """Extract text from a resume PDF."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        return text.strip()
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

from PIL import Image
import io
import pytesseract
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def ocr_extract_text(file_bytes: bytes, content_type: str) -> str:
    if content_type in ["image/jpeg", "image/png"]:
        image = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(image)
        return text
    elif content_type == "application/pdf":
        from pdf2image import convert_from_bytes
        pages = convert_from_bytes(file_bytes)
        text = ""
        for page in pages:
            text += pytesseract.image_to_string(page) + "\n"
        return text
    else:
        return ""

def redact_pii(text: str) -> str:
    results = analyzer.analyze(text=text, language="en")
    anonymized_text = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized_text.text

def process_ocr_and_redact(file_bytes: bytes, content_type: str) -> tuple[str, str]:
    """
    Returns a tuple: (original_text, redacted_text)
    """
    original_text = ocr_extract_text(file_bytes, content_type)
    print("Original OCR Text:\n", original_text)
    redacted_text = redact_pii(original_text)
    print("Redacted OCR Text:\n", redacted_text)
    return original_text, redacted_text

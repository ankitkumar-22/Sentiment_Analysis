import cv2
import numpy as np
import pytesseract

def extract_text(image):
    """Extract text from image using OCR"""
    return pytesseract.image_to_string(image)
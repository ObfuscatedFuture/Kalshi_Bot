# src/ingest/loader.py

import re

def clean_transcript(text):
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r"-\s*\n", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\xa0", " ")
    return text.strip()

def load_cleaned_transcript(pdf_path):
    txt_path = pdf_path.replace(".pdf", ".cleaned.txt")
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    return clean_transcript(text)

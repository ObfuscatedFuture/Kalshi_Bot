import pdfplumber
import re
import os
from fed_pressers.config.paths import fed_paths


# ============================================================
# SPEAKER DETECTION
# ============================================================

POWELL_HEADER_RE = re.compile(r"^CHAIR POWELL\.", re.IGNORECASE)
OTHER_SPEAKER_RE = re.compile(r"^[A-Z][A-Z .'-]+\.")  # OTHER SPEAKERS


def is_powell_header(line: str) -> bool:
    return bool(POWELL_HEADER_RE.match(line.strip()))


def is_other_speaker_header(line: str) -> bool:
    s = line.strip()
    if is_powell_header(s):
        return False
    return bool(OTHER_SPEAKER_RE.match(s))


# ============================================================
# PDF EXTRACTOR
# ============================================================

def extract_text_from_pdf(path):
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n".join(pages)


# ============================================================
# POWELL-ONLY FILTER
# ============================================================

def extract_powell_only(text):
    lines = text.split("\n")

    keep = []
    in_powell = False

    for line in lines:
        s = line.strip()
        if not s:
            continue

        if is_powell_header(s):
            in_powell = True
            keep.append(s.replace("CHAIR POWELL.", "").strip())
            continue

        if is_other_speaker_header(s):
            in_powell = False
            continue

        if in_powell:
            keep.append(s)

    final = " ".join(keep)
    final = re.sub(r"\s+", " ", final).strip()
    return final


# ============================================================
# FED PAGE / FOOTER CLEANER
# ============================================================

PAGE_ARTIFACT_RE = re.compile(
    r"""
    Page\s+\d+\s+of\s+\d+|
    Chair\s+Powell’s\s+Press\s+Conference|
    Federal\s+Reserve|
    PRELIMINARY|
    FINAL|
    January\s+\d{1,2},\s+\d{4}|
    February\s+\d{1,2},\s+\d{4}|
    March\s+\d{1,2},\s+\d{4}|
    April\s+\d{1,2},\s+\d{4}|
    May\s+\d{1,2},\s+\d{4}|
    June\s+\d{1,2},\s+\d{4}|
    July\s+\d{1,2},\s+\d{4}|
    August\s+\d{1,2},\s+\d{4}|
    September\s+\d{1,2},\s+\d{4}|
    October\s+\d{1,2},\s+\d{4}|
    November\s+\d{1,2},\s+\d{4}|
    December\s+\d{1,2},\s+\d{4}
    """,
    re.IGNORECASE | re.VERBOSE
)

def clean_fed_artifacts(text: str) -> str:
    # Remove page artifacts (safe deletions)
    text = PAGE_ARTIFACT_RE.sub(" ", text)

    # Convert ALL dash variants to spaces (never delete)
    text = re.sub(r"[–—−]", " ", text)
    text = re.sub(r"\s-\s", " ", text)

    # Handle line-break hyphens conservatively (split, don't merge)
    text = re.sub(r"(\w)-\s+(\w)", r"\1 \2", text)

    # Normalize quotes (optional, safe)
    text = text.replace("’", "'")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# BATCH PROCESSOR
# ============================================================

def process_pdfs(pdf_paths):
    for pdf_path in pdf_paths:
        print(f"\nProcessing: {pdf_path}")

        raw = extract_text_from_pdf(pdf_path)
        powell_text = extract_powell_only(raw)
        powell_text = clean_fed_artifacts(powell_text)

        out_path = pdf_path.replace(".pdf", ".cleaned.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(powell_text)

        print(f"✔ Saved Powell-only transcript → {out_path}")


# ============================================================
# RUN
# ============================================================

# cd projects
# python -m fed_pressers.tools.cleaning

if __name__ == "__main__":
    print("\n=== Processing Fed Press Conferences (Powell Only) ===")
    process_pdfs(fed_paths)

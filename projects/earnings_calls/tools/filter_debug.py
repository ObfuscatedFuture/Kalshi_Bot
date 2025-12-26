import pdfplumber
import re
import os

# ============================================================
# DETECTOR LISTS
# ============================================================

ANALYST_TITLES = [
    "analyst",
    "research analyst",
    "senior analyst",
    "equity analyst",
    "equities analyst",
    "consumer analyst",
    "md & senior analyst",
    "md and equity research analyst",
]

ANALYST_FIRM_KEYWORDS = [
    "jpmorgan", "wells fargo", "ubs", "oppenheimer", "morgan stanley",
    "bernstein", "barclays", "bofa", "bank of america", "citigroup",
    "cowen", "goldman", "raymond james", "piper sandler", "telsey",
    "evercore", "jefferies", "research division", "equity research",
    "equities", "bmo", "capital markets", "deutsche bank", "advisors", "Gordon Haskett"
]

MANAGEMENT_TITLES = [
    "chief", "ceo", "cfo", "coo", "executive", "officer",
    "president", "chairman", "vp", "evp", "senior vice president"
]


# ============================================================
# SPEAKER HEADER DETECTION
# ============================================================
def is_speaker_header(line: str) -> bool:
    """
    Detect generic earnings-call speaker headers.
    Handles formats like:
      Name - Firm - Title
      Name–Firm, Analyst
      Name– Firm – Analyst
      Name—Firm—Analyst
      Name-Firm - Analyst
    """
    line = line.strip()

    # Must start with capitalized name
    if not re.match(r"^[A-Z][A-Za-z .,'-]+", line):
        return False

    # Any dash variant with or without surrounding spaces
    # Examples matched:
    #   " - ", "- ", " -", "-", "–", "—"
    if not re.search(r"\s?[–—-]\s?", line):
        return False

    return True



def is_management_header(line: str) -> bool:
    l = line.lower()
    return any(title in l for title in MANAGEMENT_TITLES)


def is_operator(line: str) -> bool:
    return line.strip().lower().startswith("operator")

def is_analyst_header(line: str) -> bool:
    l = line.lower()

    # Structured header (Name - Firm - Title)
    if is_speaker_header(line):
        if is_management_header(line):
            return False
        if is_operator(line):
            return False
        if any(title in l for title in ANALYST_TITLES):
            return True
        if any(firm in l for firm in ANALYST_FIRM_KEYWORDS):
            return True

    # Direct ", Analyst" pattern
    if re.search(r",\s*analyst\b", l):
        return True

    # --- NEW FALLBACK CATCH-ALL ---
    # Handles cases like:
    #   "Robert F. Ohmes Q Analyst, BofA Securities"
    #   "Steve Zaccone Citi Analyst"
    if any(title in l for title in ANALYST_TITLES) and \
       any(firm in l for firm in ANALYST_FIRM_KEYWORDS):
        return True

    return False




# ============================================================
# PDF EXTRACTOR
# ============================================================

def extract_text_from_pdf(path):
    out = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                out.append(t)
    return "\n".join(out)


# ============================================================
# ANALYST REMOVAL ENGINE
# ============================================================
def remove_analyst_blocks(text):
    lines = text.split("\n")

    cleaned = []
    removed_blocks = []
    in_analyst = False
    block = []

    for line in lines:
        s = line.strip()
        if not s:
            continue

        # Operator lines MUST REMAIN (per Kalshi rules)
        if is_operator(s):
            # If operator appears mid-analyst-block, treat operator as mgmt boundary
            if in_analyst:
                removed_blocks.append("\n".join(block))
                block = []
                in_analyst = False
            cleaned.append(s)
            continue

        # Analyst header → begin analyst block
        if is_analyst_header(s):
            in_analyst = True
            block.append(s)
            continue

        # Management header → keep + close analyst block
        if is_management_header(s):
            if in_analyst:
                removed_blocks.append("\n".join(block))
                block = []
                in_analyst = False
            cleaned.append(s)
            continue

        # Inside analyst block → remove
        if in_analyst:
            block.append(s)
            continue

        # Default → keep
        cleaned.append(s)

    # leftover analyst block
    if block:
        removed_blocks.append("\n".join(block))

    final = " ".join(cleaned)
    final = re.sub(r"\s+", " ", final).strip()

    return final, removed_blocks


# ============================================================
# BATCH PROCESSOR
# ============================================================

def process_pdfs(pdf_paths):
    for pdf_path in pdf_paths:
        print(f"\nProcessing: {pdf_path}")

        raw = extract_text_from_pdf(pdf_path)
        cleaned, _ = remove_analyst_blocks(raw)

        out_path = pdf_path.replace(".pdf", ".cleaned.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        print(f"✔ Saved cleaned transcript → {out_path}")


# ============================================================
# YOUR PATH LISTS
# ============================================================

# hd_paths = [
#     'dataset/HD-earnings/1Q23-transcript-HD.pdf',
#     'dataset/HD-earnings/2Q23-transcript-HD.pdf',
#     'dataset/HD-earnings/3Q23-transcript-HD.pdf',
#     'dataset/HD-earnings/4Q23-transcript-HD.pdf',
#     'dataset/HD-earnings/1Q24-transcript-HD.pdf',
#     'dataset/HD-earnings/2Q24-transcript-HD.pdf',
#     'dataset/HD-earnings/3Q24-transcript-HD.pdf',
#     'dataset/HD-earnings/4Q24-transcript-HD.pdf',
#     'dataset/HD-earnings/1Q25-transcript-HD.pdf',
#     'dataset/HD-earnings/2Q25-transcript-HD.pdf',
#     'dataset/HD-earnings/3Q25-transcript-HD.pdf',
# ]
#
# lowes_paths = [
#     'dataset/Lowe-earnings/1Q23-transcript-LOWE.pdf',
#     'dataset/Lowe-earnings/2Q23-transcript-LOWE.pdf',
#     'dataset/Lowe-earnings/3Q23-transcript-LOWE.pdf',
#     'dataset/Lowe-earnings/4Q23-transcript-LOWE.pdf',
#     'dataset/Lowe-earnings/1Q24-transcript-LOWE.pdf',
#     'dataset/Lowe-earnings/2Q24-transcript-LOWE.pdf',
#     'dataset/Lowe-earnings/3Q24-transcript-LOWE.pdf',
#     'dataset/Lowe-earnings/4Q24-transcript-LOWE.pdf',
#     'dataset/Lowe-earnings/1Q25-transcript-LOWE.pdf',
#     'dataset/Lowe-earnings/2Q25-transcript-LOWE.pdf'
# ]
#
# target_paths = [
#     'dataset/Target-earnings/1Q23-transcript-TGT.pdf',
#     'dataset/Target-earnings/2Q23-transcript-TGT.pdf',
#     'dataset/Target-earnings/3Q23-transcript-TGT.pdf',
#     'dataset/Target-earnings/4Q23-transcript-TGT.pdf',
#     'dataset/Target-earnings/1Q24-transcript-TGT.pdf',
#     'dataset/Target-earnings/2Q24-transcript-TGT.pdf',
#     'dataset/Target-earnings/3Q24-transcript-TGT.pdf',
#     'dataset/Target-earnings/4Q24-transcript-TGT.pdf',
#     'dataset/Target-earnings/1Q25-transcript-TGT.pdf',
#     'dataset/Target-earnings/2Q25-transcript-TGT.pdf'
# ]
#
# walmart_paths = [
#     'dataset/Walmart-earnings/1Q23-transcript-WMT.pdf',
#     'dataset/Walmart-earnings/2Q23-transcript-WMT.pdf',
#     'dataset/Walmart-earnings/3Q23-transcript-WMT.pdf',
#     'dataset/Walmart-earnings/4Q23-transcript-WMT.pdf',
#     'dataset/Walmart-earnings/1Q24-transcript-WMT.pdf',
#     'dataset/Walmart-earnings/2Q24-transcript-WMT.pdf',
#     'dataset/Walmart-earnings/3Q24-transcript-WMT.pdf',
#     'dataset/Walmart-earnings/4Q24-transcript-WMT.pdf',
#     'dataset/Walmart-earnings/1Q25-transcript-WMT.pdf',
#     'dataset/Walmart-earnings/2Q25-transcript-WMT.pdf'
# ]
snow_paths = [
    'dataset/SNOW-earnings/1Q21-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/2Q21-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/3Q21-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/4Q21-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/1Q22-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/2Q22-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/3Q22-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/4Q22-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/1Q23-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/2Q23-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/3Q23-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/4Q23-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/1Q24-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/2Q24-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/3Q24-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/4Q24-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/1Q25-transcript-SNOW.pdf',
    'dataset/SNOW-earnings/2Q25-transcript-SNOW.pdf'
]

# ============================================================
# RUN EVERYTHING
# ============================================================

if __name__ == "__main__":
    #print("\n=== Processing Home Depot ===")
    #process_pdfs(hd_paths)

    #print("\n=== Processing Lowe's ===")
    #process_pdfs(lowes_paths)

    #print("\n=== Processing Target ===")
    #process_pdfs(target_paths)

    #print("\n=== Processing Walmart ===")
    #process_pdfs(walmart_paths)

    print("\n=== Processing Snowflake ===")
    process_pdfs(snow_paths)
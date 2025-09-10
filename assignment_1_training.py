#!/usr/bin/env python3
"""
extract_autism_papers_llm_pdfplumber.py

Usage:
    python extract_autism_papers_llm_pdfplumber.py /path/to/pdf_dir /path/to/output.csv
"""

import sys, json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import pdfplumber
from openai import OpenAI
import re

# Initialize client (expects OPENAI_API_KEY in env)
client = OpenAI(api_key="")

# -------------------------
# PROMPT TEMPLATE
# -------------------------
PROMPT_TEMPLATE = """You are an expert evidence extraction assistant.

From the autism treatment article text below, extract these fields:

- title
- authors (if detectable, else null)
- year (if detectable, else null)
- population_age (string, e.g. "29 to 60 months", or "mean 4.2 years")
- language_level (e.g., minimally verbal, verbal, nonverbal; list allowed)
- setting (list: clinic, school, home, telehealth, preschool, etc.)
- target (list: social communication, irritability, sleep, language, etc.)
- outcome_family (list: outcome measure families used)
- study_design (e.g., randomized controlled trial, cohort, case study)
- comparator (string, e.g., "control group", "standard care")
- followup_length (string, e.g., "12 months", "8 weeks")
- sample_size (integer if possible)
- sessions_info (string if available, e.g., "2 daily 20-minute sessions for 8 weeks")

Return ONLY valid JSON in this format:
{
  "title": "...",
  "authors": "...",
  "year": "...",
  "population_age": "...",
  "language_level": ["..."],
  "setting": ["..."],
  "target": ["..."],
  "outcome_family": ["..."],
  "study_design": "...",
  "comparator": "...",
  "followup_length": "...",
  "sample_size": 0,
  "sessions_info": "..."
}
"""

# -------------------------
# Helpers
# -------------------------
def extract_text_from_pdf(path: Path) -> str:
    """Extract all text from PDF using pdfplumber."""
    text_parts = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                text_parts.append(text)
    return "\n\n".join(text_parts)

def chunk_text(text, max_chars=6000):
    """Split long text into chunks for LLM input."""
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def query_llm(text: str):
    """Ask LLM to extract fields from a chunk of text."""
    prompt = PROMPT_TEMPLATE + "\n\nARTICLE TEXT:\n" + text
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = response.choices[0].message.content
    try:
        return json.loads(content)
    except Exception:
        return {"raw_response": content}

def merge_records(records):
    """Merge multiple chunk results into one final record."""
    final = {}
    for rec in records:
        for k, v in rec.items():
            if v and v != "null":
                if isinstance(v, list):
                    if k not in final:
                        final[k] = []
                    final[k] = list(set(final[k] + v))
                else:
                    if k not in final or not final[k]:
                        final[k] = v
    return final

def process_pdf(pdf_path: Path, raw_out_dir: Path):
    raw_out_dir.mkdir(parents=True, exist_ok=True)
    text = extract_text_from_pdf(pdf_path)

    raw_txt_path = raw_out_dir / (pdf_path.stem + ".txt")
    raw_txt_path.write_text(text, encoding="utf-8")

    chunks = chunk_text(text)
    results = [query_llm(chunk) for chunk in chunks]
    merged = merge_records(results)

    merged["paper_id"] = pdf_path.stem
    merged["source_pdf"] = str(pdf_path)
    merged["raw_text_path"] = str(raw_txt_path)
    return merged

# -------------------------
# Main
# -------------------------
def main(pdf_dir: str, out_csv: str):
    pdf_dir = Path(pdf_dir)
    out_csv = Path(out_csv)
    out_jsonl = out_csv.with_suffix(".jsonl")
    raw_out_dir = out_csv.parent / "raw_texts"
    raw_out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(list(pdf_dir.glob("*.pdf")))
    if not pdfs:
        print("No PDFs found in", pdf_dir)
        return

    # Prepare CSV header
    fieldnames = [
        "paper_id", "title", "authors", "year",
        "population_age", "language_level", "setting", "target",
        "outcome_family", "study_design", "comparator", "followup_length",
        "sample_size", "sessions_info", "source_pdf", "raw_text_path"
    ]
    pd.DataFrame(columns=fieldnames).to_csv(out_csv, index=False)

    with open(out_jsonl, "w", encoding="utf-8") as jf:
        pass

    for p in tqdm(pdfs, desc="Processing PDFs"):
        try:
            rec = process_pdf(p, raw_out_dir)

            # Append to CSV
            pd.DataFrame([rec]).to_csv(out_csv, mode="a", header=False, index=False)

            # Append to JSONL
            with open(out_jsonl, "a", encoding="utf-8") as jf:
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

            print(f"Processed {p.name} ✅")
        except Exception as e:
            print(f"ERROR processing {p}: {e}")

    print(f"\nAll done! Outputs saved at:\nCSV -> {out_csv}\nJSONL -> {out_jsonl}\nRaw texts -> {raw_out_dir}")


if __name__ == "__main__":
    pdf_dir = r"C:\Users\vg48912\Documents\Training\Data"
    out_csv = "training_out.csv"
    main(pdf_dir, out_csv)



input_file = r"C:\Users\vg48912\Documents\JASPER\training_out.csv"
df = pd.read_csv(input_file)
jsons = df['paper_id'].tolist()

records = []
for jso in jsons:
    cleaned = re.sub(r"^```json|```$", "", jso.strip(), flags=re.MULTILINE).strip()
    data = json.loads(cleaned)
    records.append(data)

final_df = pd.DataFrame(records)
final_df.to_csv(r"C:\Users\vg48912\Documents\JASPER\assignment_1_output.csv", index=False)


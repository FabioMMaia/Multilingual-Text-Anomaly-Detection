import os
from pypdf import PdfReader

BASE = r"puplication\related_works"

def extract_first_pages(pdf_path, n_pages=2):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for i in range(min(n_pages, len(reader.pages))):
            text += reader.pages[i].extract_text() or ""
        return text[:3000]
    except Exception as e:
        return f"ERROR: {e}"

for folder in ['ad_nlp', os.path.join('llm_ad','citations_from_original'),
               os.path.join('llm_ad','scopus'), os.path.join('llm_ad','web_of_knowledge')]:
    full_path = os.path.join(BASE, folder)
    pdfs = [f for f in os.listdir(full_path) if f.endswith('.pdf')]
    print(f"\n{'='*70}")
    print(f"FOLDER: {folder}")
    print(f"{'='*70}")
    for pdf in sorted(pdfs):
        path = os.path.join(full_path, pdf)
        text = extract_first_pages(path, n_pages=1)
        # Print just enough to identify title/abstract
        print(f"\n--- {pdf} ---")
        print(text[:800])

import sys, re
sys.path.append('src')
from auto_ingest.stage3_parsing.regex_parser import parse_with_regex
from auto_ingest.utils.patterns import SECTION_PATTERNS
from auto_ingest.stage0_preprocess.extractor import extract_text_without_footers, clean_text

pdf_path = 'data/legal_pdfs/CPC_1908.pdf'
text = clean_text(extract_text_without_footers(pdf_path))

chunks = parse_with_regex(text, SECTION_PATTERNS[0], 'CPC 1908', 'CODIFIED_ACT')

for c in chunks:
    meta = c['metadata']
    if meta.get('type') == 'section' and str(meta.get('section_number')) == '1':
        print(f"Section 1 Part: {meta.get('part')}  Text snippet: {repr(c['text'][:100])}")
        print('Length of section 1 chunk:', len(c['text']))

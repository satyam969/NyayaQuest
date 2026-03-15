import os
import requests

# List of official legal PDFs from India Code and Ministries
LEGAL_DOCS = {
    "BNS_2023": "https://mha.gov.in/sites/default/files/250883_english_01042024.pdf",
    "BNSS_2023": "https://mha.gov.in/sites/default/files/250880_english_01042024.pdf",
    "BSA_2023": "https://mha.gov.in/sites/default/files/250882_english_02042024.pdf",
    "CPC_1908": "https://lddashboard.legislative.gov.in/sites/default/files/A1908-05.pdf",
    "RTI_2005": "https://rti.gov.in/rti-act.pdf",
    "Consumer_Protection_2019": "https://consumeraffairs.nic.in/sites/default/files/CP%20Act%202019.pdf",
    "IT_Act_2000": "https://www.meity.gov.in/writereaddata/files/itbill2000.pdf",
    "Code_on_Wages_2019": "https://labour.gov.in/sites/default/files/Code_on_Wages_2019.pdf",
    "Industrial_Relations_Code_2020": "https://labour.gov.in/sites/default/files/IR_Code_2020.pdf",
    "Social_Security_Code_2020": "https://labour.gov.in/sites/default/files/SS_Code_2020.pdf",
    "OSH_Code_2020": "https://labour.gov.in/sites/default/files/OSH_Code_2020.pdf"
}

OUTPUT_DIR = "data/legal_pdfs"

def download_docs():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    for name, url in LEGAL_DOCS.items():
        filename = f"{name}.pdf"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        if os.path.exists(filepath):
            print(f"Skipping {filename} (already exists)")
            continue

        print(f"Downloading {filename}...")
        for attempt in range(3):
            try:
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()
                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Successfully downloaded {filename}")
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {filename}: {e}")
                if attempt == 2:
                    print(f"Final failure for {filename}")

if __name__ == "__main__":
    download_docs()

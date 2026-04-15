import subprocess
import sys
import shutil
import os

# list of all the documents
scripts_to_run = [
    "src/ingest/ingest_consumer_protection_2019.py",      
    "src/ingest/ingest_labour_codes.py",      
    "src/ingest/ingest_RTI_2005.py",      
    "src/ingest/ingest_structural_BNS_2023.py",    
    "src/ingest/ingest_CPC_1980.py",      
]

def clean_database():
    print("🧹 Wiping the old database to prevent duplicates...")
    db_path = "chroma_db_groq_legal"
    
    if os.path.exists(db_path):
        try:
            shutil.rmtree(db_path)
            print("✅ Entire database folder successfully deleted from your hard drive!\n")
        except Exception as e:
            print(f"⚠️ Could not delete folder (Make sure Streamlit isn't running in another terminal!): {e}\n")
    else:
        print("✅ No old database folder found. Starting fresh!\n")

def run_all():
    print("🌟 Starting Master Database Ingestion...\n")
    
    # 1. Physically delete the old database folder
    clean_database()
    
    # 2. Run all the individual scripts sequentially
    for script in scripts_to_run:
        print(f"⏳ Running: {script}...")
        
        # This acts exactly like you typing "uv run python src/ingest/..." in the terminal
        result = subprocess.run(["uv", "run", "python", script])
        
        # If a script crashes, this stops the master script so you can fix the error
        if result.returncode != 0:
            print(f"\n❌ Error detected in {script}. Stopping the master runner.")
            sys.exit(1)
            
        print(f"✅ Successfully finished {script}\n")
        print("-" * 40 + "\n")

    print("🎉 All documents have been successfully ingested! Your database is completely clean and up to date.")

if __name__ == "__main__":
    run_all()
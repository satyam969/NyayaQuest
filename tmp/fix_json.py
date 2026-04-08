import json
import os

filepath = r"c:\Users\satya\Desktop\NyayaQuest\firebase-service-account.json"

with open(filepath, 'r') as f:
    content = f.read()

# Fix the accidental double backslashes
fixed_content = content.replace("\\\\", "\\")

try:
    data = json.loads(fixed_content)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print("✅ JSON fixed and saved successfully!")
except Exception as e:
    print(f"❌ Error fixing JSON: {e}")

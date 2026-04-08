import json

data = {
  "type": "service_account",
  "project_id": "nyayaquest",
  "private_key_id": "63617b53013728cabf144bd54c00a7586f56ae44",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC4DyiX4G154wFd\nhG2cYK0xXVnGVhD2WG2Rdo0aYKnholPsjp6IBsGPTpWjgjI1BrCQ8A4XQjO+yItD\n7eWNW7NjZ6JUs7gfKJ3ANLgK1OTb7m33dPmzh/i8ZIVM7UExPStJ6oyz5E31wzH4\n30iOtJYoI04zcUs0gwoqyv+qOSSi3hSmkRI1xTt9xsUzTPDYE/hwBhjlucPfIXXA\nsGeFS0MYx2DUjgV32134UUgTUaMd0usisK93rVCzN0J163NAKOrRU67CX9nPGWUS\nAj5lbOXcPhR1RX/LLFLEghgkJycgfg6ERR0YHZqOw3OIV8doDxyYeUWgWu/2B0NP\nPiJ1WKuXAgMBAAECggEAUz9b61FxAntnTy5Jyzw4qUahB6E+u8TkDbIygLcuruRW\nE/yvfDOeERyIdrM41R3o9yz0GpXxRH1866TqOcS7fp7NX4UHpmQe8WOGyDNhLY4K\nA63PEGHT4RWP6uq2HTie3ygsKmL81cbKz9bhgNXxwkaxpHe9/Yq1KwS0nP+Kb/S/\n9ClFgjv/WchCEFP2ePu4E0BUYDnQ95lKkzGFVzJ87Zx9nkhI634zhMCIrQZB+cMZ\nG+pEhpQU+zguQVIZQ0SUtuwe26UJNS9xu7fROHbveXLRTnOfdAr5ocrCnRu6Eol8\neZ5bTywxA1tUKL8bCiWxE1tTPHcYCYZvvbEN33W3jQKBgQDttSzAIm1KRrlcY1Q+\nQoxGgtUlKLCKbzISbBye+j5gVobbu0nWB75fY72KbiUScfG2AF06t2WeyV3+hnbp\nZHEtHL0fEwVrOebsjdEawG3Md0WJcxsbnEiD+2/YRjygFkJ107gvWZ84vzHThBgS\nyxRKZcNKYeeLnltjBvj38IzZ+wKBgQDGORyxJp7HRgpHro+P+UAN7o1jTOoxDUlM\FdPiK5MqZeXcNFjYsuK4JCM2LR9Xs0a6FJ7xujvJ+xyazTrZmtvSAW3JJnTUsNHf\nLFczdKFpGQTEDOxKhuJF9FUoVnZGF0YB99nVWHuxyxwET+TeRPVel+A2ms2AVAFi\nWhAvRwA+FQKBgQCz370VRvfclNf/CUreMg1j7ezMSZYNq0cAmb6urj41OESkUXz1\n8LYmCJuM/PwgkQiO6IejvtOu7EGsMKQayF67/FtAAGzBTvdnWYk57RMo/bgo4mlI\n42IcPU/NIJkPqshv0N43NmI91rAllneBARtBkO/OgXdtN0+AB+6t7+ElsQKBgQDC\nEgCJYKprp6NA8yMTbpDMExbSdeeEBuIQX/6GnOsEw6b8pTOnVdyrNJZU4HCjSJ6i\naLYFLLSE2Bn1ZaGMkxVM0qFOIxyXcFbKDXuCoVm2sAv+djiR7uVyX/lP+PbrQLYG\nD0dynaLdO2I+xonpI9KnvkKCs6UnUxfX1x2pa/ZM0QKBgEvm4aYBAu5DGcBnq1Ag\nbaUKgezD4BO2F47mt3v7yy5xjTeEIdSa87Ahl9aWsmhXtUfeW1RXX+5wXnX2VAKU\ncnzhWLb0sp7c/Y4wCpTMArvTLC4MSq3Yj4F/m34iy6339UEd2zAlSgzURAyiVBCZ\nbYdTAueFNc8Y1ZoF1nh2qWtC\n-----END PRIVATE KEY-----\n",
  "client_email": "firebase-adminsdk-fbsvc@nyayaquest.iam.gserviceaccount.com",
  "client_id": "105761329389215317125",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-fbsvc%40nyayaquest.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}

with open(r"c:\Users\satya\Desktop\NyayaQuest\firebase-service-account.json", 'w') as f:
    json.dump(data, f, indent=2)

print("✅ JSON written correctly via Python dict")

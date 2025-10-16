import os
from dotenv import load_dotenv

# Memuat variabel dari file .env
load_dotenv()

# Mengambil API Key. Jika tidak ada, nilainya akan None.
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
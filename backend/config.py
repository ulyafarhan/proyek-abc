import os
from dotenv import load_dotenv

# Memuat variabel dari file .env
load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
import google.generativeai as genai
from config import GEMINI_API_KEY
import pandas as pd
import json
from typing import List, Dict, Any

try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    print(f"Error saat konfigurasi Gemini: {e}")
    model = None

def generate_recovery_plan(user_profile: dict):
    # user_profile berisi ringkasan skor historis pengguna
    
    prompt = f"""
    Buatkan rencana pemulihan 1 minggu untuk pengguna dengan profil berikut: {user_profile}.
    Rencana harus mencakup 3 tantangan: 
    1. Satu tantangan membaca (misal: baca artikel panjang).
    2. Satu tantangan menulis (misal: tulis jurnal refleksi).
    3. Satu tantangan interaksi dunia nyata (misal: telepon teman).
    Berikan jawaban dalam format JSON.
    """
    
    response = model.generate_content(prompt)
    return json.loads(response.text)
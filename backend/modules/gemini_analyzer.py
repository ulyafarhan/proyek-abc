import google.generativeai as genai
from config import GEMINI_API_KEY
import pandas as pd
import json

try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    print(f"Error saat konfigurasi Gemini: {e}")
    model = None

def get_intelligent_analysis_from_gemini(df: pd.DataFrame):
    if not model or df.empty:
        return {"joker_keywords": [], "thanos_keywords": [], "analysis_summary": "Analisis Gemini tidak tersedia."}

    # Tahap 1: Saring komentar paling "berdampak" untuk dikirim ke Gemini
    # Kita ambil komentar dengan sentimen negatif terkuat
    impactful_comments = df.sort_values(by='compound', ascending=True).head(50)
    sample_text = "\n".join(impactful_comments['text'].dropna().astype(str).tolist())

    # Tahap 2: Rancang prompt cerdas
    prompt = f"""
    Anda adalah seorang psikolog digital dan linguis forensik yang menganalisis dinamika komunitas online.
    Konteks: Arketipe 'Joker' merepresentasikan nihilisme, kekacauan, dan penolakan makna. Arketipe 'Thanos' merepresentasikan logika dingin, ekstremisme rasional, dan fokus pada tujuan tunggal.

    Berdasarkan sampel komentar YouTube yang paling negatif dan emosional berikut:
    ---
    {sample_text}
    ---

    Tugas Anda:
    1.  **Ekstrak Kata Kunci Joker:** Identifikasi dan ekstrak 5 kata atau frasa spesifik (kata sifat, kata benda, ejekan) yang paling kuat mendorong sentimen komunitas ke arah arketipe Joker (kemarahan, sinisme, keputusasaan, ejekan tanpa tujuan).
    2.  **Ekstrak Kata Kunci Thanos:** Identifikasi dan ekstrak 5 kata atau frasa spesifik yang paling kuat mendorong sentimen ke arah arketipe Thanos (pemikiran absolut, solusi ekstrem, objektivitas dingin, penolakan emosi). Jika tidak ada, kembalikan array kosong.
    3.  **Berikan Ringkasan Analisis:** Berikan satu kalimat ringkasan tentang mengapa kata-kata tersebut dipilih dan apa indikasinya terhadap komunitas.

    Berikan jawaban HANYA dalam format JSON yang valid seperti ini:
    {{
      "joker_keywords": ["kata1", "frasa2", ...],
      "thanos_keywords": ["kata3", "frasa4", ...],
      "analysis_summary": "Ringkasan analisis Anda di sini."
    }}
    """
    try:
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response)
    except Exception as e:
        print(f"Error saat memanggil Gemini API: {e}")
        return {"joker_keywords": [], "thanos_keywords": [], "analysis_summary": f"Error: {e}"}
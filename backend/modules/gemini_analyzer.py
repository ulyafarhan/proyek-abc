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

def get_intelligent_analysis_from_gemini(df: pd.DataFrame):
    if not model or df.empty:
        return {"joker_keywords": [], "thanos_keywords": [], "analysis_summary": "Analisis Gemini tidak tersedia.", "community_vibe": "Tidak diketahui", "main_themes": []}

    impactful_comments = df.sort_values(by='compound', ascending=True).head(50)
    sample_text = "\n".join(impactful_comments['text'].dropna().astype(str).tolist())

    prompt = f"""
    Anda adalah seorang psikolog digital dan linguis forensik yang menganalisis dinamika komunitas online.
    Konteks: Arketipe 'Joker' merepresentasikan nihilisme, kekacauan, dan penolakan makna. Arketipe 'Thanos' merepresentasikan logika dingin, ekstremisme rasional, dan fokus pada tujuan tunggal.

    Berdasarkan sampel komentar YouTube yang paling negatif dan emosional berikut:
    ---
    {sample_text}
    ---

    Tugas Anda:
    1.  **Ekstrak Kata Kunci Joker:** Identifikasi dan ekstrak 5 kata atau frasa spesifik (kata sifat, kata benda, ejekan) yang paling kuat mendorong sentimen komunitas ke arah arketipe Joker.
    2.  **Ekstrak Kata Kunci Thanos:** Identifikasi dan ekstrak 5 kata atau frasa spesifik yang paling kuat mendorong sentimen ke arah arketipe Thanos. Jika tidak ada, kembalikan array kosong.
    3.  **Berikan Ringkasan Analisis:** Berikan satu kalimat ringkasan tentang mengapa kata-kata tersebut dipilih.
    4.  **Berikan 'Community Vibe':** Berikan satu atau dua kata (misal: "Marah & Sinis", "Logis & Kritis", "Netral") yang merangkum 'vibe' atau suasana keseluruhan dari sampel komentar ini.
    5.  **Ekstrak Topik Utama:** Identifikasi 3-5 topik atau tema utama yang paling sering dibicarakan dalam komentar-komentar ini (sebagai array string).

    Berikan jawaban HANYA dalam format JSON yang valid seperti ini:
    {{
      "joker_keywords": ["kata1", "frasa2"],
      "thanos_keywords": ["kata3", "frasa4"],
      "analysis_summary": "Ringkasan analisis Anda di sini.",
      "community_vibe": "Vibe komunitas di sini.",
      "main_themes": ["Topik 1", "Topik 2", "Topik 3"]
    }}
    """
    try:
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response)
    except Exception as e:
        print(f"Error saat memanggil Gemini API: {e}")
        return {"joker_keywords": [], "thanos_keywords": [], "analysis_summary": f"Error: {e}", "community_vibe": "Error", "main_themes": []}

def get_brainrot_analysis(activities: List[Dict[str, Any]]):
    if not model:
        return {"brainrot_score": -1, "analysis": "Analisis Gemini tidak tersedia."}

    summary = []
    content_types = {"tiktok": 0, "youtube_short": 0, "youtube_video": 0, "article": 0, "other": 0}
    
    for activity in activities:
        url = activity.get("url", "")
        if "tiktok.com" in url:
            content_types["tiktok"] += 1
        elif "youtube.com/shorts" in url:
            content_types["youtube_short"] += 1
        elif "youtube.com/watch" in url:
            content_types["youtube_video"] += 1
        elif "kompas.com" in url or "detik.com" in url or "bbc.com" in url:
            content_types["article"] += 1
        else:
            content_types["other"] += 1
            
    total_activities = len(activities)
    analysis_summary = (
        f"Total aktivitas: {total_activities}\n"
        f"- Konten Instan (TikTok/Shorts): {content_types['tiktok'] + content_types['youtube_short']}\n"
        f"- Konten Video (Durasi Penuh): {content_types['youtube_video']}\n"
        f"- Konten Artikel (Teks): {content_types['article']}\n"
    )

    prompt = f"""
    Anda adalah seorang sosiolog digital yang menganalisis fenomena 'Brain Rot'.
    Konteks: 'Brain Rot' adalah istilah untuk penurunan kognitif akibat konsumsi berlebihan konten digital yang instan, berulang, dan pasif.

    Berdasarkan ringkasan data aktivitas pengguna berikut:
    ---
    {analysis_summary}
    ---

    Tugas Anda:
    1.  **Berikan Skor Brain Rot:** Berikan skor dari 0 (Sangat Sehat) hingga 100 (Risiko Brain Rot Tinggi) berdasarkan rasio konten instan (TikTok/Shorts) terhadap konten yang lebih mendalam (Video Penuh/Artikel).
    2.  **Berikan Ringkasan Analisis:** Berikan 1-2 kalimat analisis yang menjelaskan skor tersebut dan apa indikasinya bagi pengguna, sesuai dengan konteks teoritis.

    Berikan jawaban HANYA dalam format JSON yang valid seperti ini:
    {{
      "brainrot_score": 85,
      "analysis": "Ringkasan analisis Anda di sini."
    }}
    """
    try:
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response)
    except Exception as e:
        print(f"Error saat memanggil Gemini API: {e}")
        return {"brainrot_score": -1, "analysis": f"Error: {e}"}
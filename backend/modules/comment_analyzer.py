import pandas as pd
from transformers import pipeline

print("Memuat model AI, ini mungkin butuh beberapa saat...")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
print("Model AI selesai dimuat.")

ENGAGEMENT_PHRASES = [
    'like if', 'subscribe', 'comment below', 'what do you think', 
    'setuju gak', 'klik link', 'cek bio', 'my reaction'
]

# Fungsi ini sekarang hanya untuk analisis sentimen cepat (digunakan untuk pra-filter)
def add_sentiment_scores_to_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or df['text'].dropna().empty:
        df['compound'] = 0.0
        return df
    
    sample_texts = df['text'].dropna().astype(str).tolist()
    results = sentiment_pipeline(sample_texts, truncation=True)
    
    # Buat pemetaan dari teks ke skor compound
    scores = {}
    for i, text in enumerate(sample_texts):
        score = results[i]['score']
        if results[i]['label'] == 'NEGATIVE':
            score = -score
        scores[text] = score
        
    df['compound'] = df['text'].map(scores).fillna(0.0)
    return df

def analyze_emotions_hf(df: pd.DataFrame) -> dict:
    full_text = " ".join(df['text'].dropna().astype(str))
    if not full_text:
        return {'anger': 0, 'disgust': 0, 'fear': 0, 'joy': 0, 'sadness': 0, 'surprise': 0}
    results = emotion_pipeline(full_text, truncation=True)
    return {res['label']: round(res['score'] * 100, 2) for res in results[0]}

def calculate_lexical_diversity(df: pd.DataFrame) -> float:
    full_text = " ".join(df['text'].dropna().astype(str).str.lower())
    words = full_text.split()
    if not words: return 0.0
    return round((len(set(words)) / len(words)) * 100, 2)

def calculate_reinforcement_score(df: pd.DataFrame) -> float:
    if df.empty: return 0.0
    engagement_count = sum(df['text'].str.contains(phrase, case=False, na=False, regex=False).sum() for phrase in ENGAGEMENT_PHRASES)
    return round(min(100.0, (engagement_count / len(df)) * 200), 2)

def calculate_archetype_scores_from_gemini(df: pd.DataFrame, gemini_analysis: dict) -> dict:
    """Tahap 3: Menghitung skor pada seluruh data berdasarkan kata kunci dari Gemini."""
    joker_keywords = gemini_analysis.get("joker_keywords", [])
    thanos_keywords = gemini_analysis.get("thanos_keywords", [])
    
    if df.empty: return {"joker_score": 0, "thanos_score": 0}

    full_text = " ".join(df['text'].dropna().astype(str).str.lower())
    words = full_text.split()
    if not words: return {"joker_score": 0, "thanos_score": 0}

    joker_count = sum(1 for word in words if word.lower() in joker_keywords)
    thanos_count = sum(1 for word in words if word.lower() in thanos_keywords)

    # Menghitung skor berdasarkan frekuensi kemunculan kata kunci per 1000 kata
    joker_score = round(min(100.0, (joker_count / len(words)) * 1000), 2)
    thanos_score = round(min(100.0, (thanos_count / len(words)) * 1000), 2)

    return {"joker_score": joker_score, "thanos_score": thanos_score}
import pandas as pd
from transformers import pipeline
import re

print("Memuat model AI, ini mungkin butuh beberapa saat...")
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
emotion_pipeline = pipeline("text-classification", model="joelito/roberta-large-go_emotions", top_k=None)
print("Model AI selesai dimuat.")

ENGAGEMENT_PHRASES = [
    'like if', 'subscribe', 'comment below', 'what do you think', 
    'setuju gak', 'klik link', 'cek bio', 'my reaction'
]

def add_sentiment_scores_to_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or df['text'].dropna().empty:
        df['compound'] = 0.0
        return df
    
    sample_texts = df['text'].dropna().astype(str).tolist()
    results = sentiment_pipeline(sample_texts, truncation=True)
    
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
    joker_keywords = [k.lower() for k in gemini_analysis.get("joker_keywords", []) if k]
    thanos_keywords = [k.lower() for k in gemini_analysis.get("thanos_keywords", []) if k]
    
    if df.empty or df['text'].dropna().empty: 
        return {"joker_score": 0, "thanos_score": 0}

    all_comments_lower = df['text'].dropna().astype(str).str.lower()
    total_comments = len(all_comments_lower)
    if total_comments == 0:
        return {"joker_score": 0, "thanos_score": 0}

    joker_count = 0
    thanos_count = 0

    if joker_keywords:
        joker_pattern = '|'.join(re.escape(k) for k in joker_keywords)
        joker_count = all_comments_lower.str.contains(joker_pattern, case=False, na=False, regex=True).sum()

    if thanos_keywords:
        thanos_pattern = '|'.join(re.escape(k) for k in thanos_keywords)
        thanos_count = all_comments_lower.str.contains(thanos_pattern, case=False, na=False, regex=True).sum()

    joker_score = round((joker_count / total_comments) * 100, 2)
    thanos_score = round((thanos_count / total_comments) * 100, 2)

    return {"joker_score": joker_score, "thanos_score": thanos_score}
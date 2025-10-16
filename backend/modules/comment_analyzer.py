# modules/comment_analyzer.py

import pandas as pd
import nltk
import text2emotion as te
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("Downloading NLTK model 'vader_lexicon'...")
    nltk.download('vader_lexicon')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK model 'stopwords'...")
    nltk.download('stopwords')
try:
    # Baris ini akan memperbaiki error 'LookupError: Resource punkt_tab not found'
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK model 'punkt'...")
    nltk.download('punkt')
# ===================================================================


# --- DEFINISI KATA KUNCI TEORETIS ---
INTERNET_SLANG = [
    'sigma', 'rizz', 'gyatt', 'skibidi', 'fanum tax', 'cringe', 
    'based', 'copium', 'pog', 'let him cook', 'no cap', 'fr fr'
]
ENGAGEMENT_PHRASES = [
    'like if', 'subscribe', 'comment below', 'what do you think', 
    'setuju gak', 'klik link', 'cek bio'
]

# --- FUNGSI-FUNGSI ANALISIS (Semua menerima DataFrame) ---

def analyze_sentiments(df: pd.DataFrame) -> dict:
    """Menganalisis sentimen dari DataFrame komentar."""
    if df.empty:
        return {'positive_percent': 0, 'neutral_percent': 0, 'negative_percent': 0}

    sid = SentimentIntensityAnalyzer()
    # Pastikan kolom 'text' adalah string untuk menghindari error
    df['sentiment'] = df['text'].astype(str).apply(sid.polarity_scores)
    df['compound'] = df['sentiment'].apply(lambda score: score['compound'])

    positive = len(df[df['compound'] > 0.05])
    neutral = len(df[(df['compound'] >= -0.05) & (df['compound'] <= 0.05)])
    negative = len(df[df['compound'] < -0.05])
    total = len(df)

    return {
        'positive_percent': round((positive / total) * 100, 2) if total > 0 else 0,
        'neutral_percent': round((neutral / total) * 100, 2) if total > 0 else 0,
        'negative_percent': round((negative / total) * 100, 2) if total > 0 else 0,
    }

def analyze_emotions(df: pd.DataFrame) -> dict:
    """Menganalisis emosi dari DataFrame komentar."""
    full_text = " ".join(df['text'].dropna().astype(str))
    if not full_text:
        return {'Happy': 0, 'Angry': 0, 'Surprise': 0, 'Sad': 0, 'Fear': 0}
    # Batasi panjang teks untuk menjaga performa
    return te.get_emotion(full_text[:50000])

def calculate_lexical_diversity(df: pd.DataFrame) -> float:
    """Menghitung skor keragaman leksikal (0-100)."""
    full_text = " ".join(df['text'].dropna().astype(str).str.lower())
    words = full_text.split()
    if not words:
        return 0.0
    unique_words = set(words)
    diversity_score = len(unique_words) / len(words)
    return round(diversity_score * 100, 2)

def calculate_simulation_score(df: pd.DataFrame) -> float:
    """Menghitung skor simulasi Baudrillard (0-100)."""
    full_text = " ".join(df['text'].dropna().astype(str).str.lower())
    words = full_text.split()
    if not words:
        return 0.0
    meme_count = sum(1 for word in words if word in INTERNET_SLANG)
    score = min(100.0, (meme_count / len(words)) * 1000 * 5)
    return round(score, 2)

def calculate_reinforcement_score(df: pd.DataFrame) -> float:
    """Menghitung skor penguatan perilaku Skinner (0-100)."""
    if df.empty:
        return 0.0
    engagement_count = sum(df['text'].str.contains(phrase, case=False, na=False, regex=False).sum() for phrase in ENGAGEMENT_PHRASES)
    total_comments = len(df)
    if total_comments == 0:
        return 0.0
    score = min(100.0, (engagement_count / total_comments) * 100 * 2)
    return round(score, 2)

def find_top_topics(df: pd.DataFrame, num_topics: int = 5, words_per_topic: int = 4) -> list[str]:
    """Menemukan topik utama dari DataFrame komentar."""
    if df.empty or len(df['text'].dropna()) < num_topics:
        return ["Tidak cukup data untuk analisis topik."]
        
    stop_words = nltk.corpus.stopwords.words('english')
    vectorizer = CountVectorizer(max_df=0.9, min_df=5, stop_words=stop_words)
    try:
        term_matrix = vectorizer.fit_transform(df['text'].dropna().astype(str))
    except ValueError:
        return ["Teks terlalu singkat untuk dianalisis."]
        
    if term_matrix.shape[1] == 0:
        return ["Tidak ada kata kunci yang signifikan setelah diproses."]

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(term_matrix)
    feature_names = vectorizer.get_feature_names_out()
    top_topics = []
    for component in lda.components_:
        top_words_indices = component.argsort()[:-words_per_topic - 1:-1]
        top_words = [feature_names[i] for i in top_words_indices]
        top_topics.append(", ".join(top_words))
    return top_topics

# --- FUNGSI SINTESIS (LOGIKA UTAMA) ---

def determine_complex_archetype(metrics: dict) -> dict:
    """Fungsi sintesis untuk menentukan arketipe berdasarkan semua metrik."""
    negativity = metrics.get('sentiment', {}).get('negative_percent', 0)
    anger = metrics.get('emotion', {}).get('Angry', 0.0) * 100
    simulation = metrics.get('simulation_score', 0)
    diversity = metrics.get('diversity_score', 0)

    # Logika untuk Joker: reaktif (negatif & marah), anarkis, hidup dalam simulasi meme
    joker_score = (negativity * 0.4) + (anger * 0.3) + (simulation * 0.3)
    
    # Logika untuk Thanos: fokus tinggi (keragaman rendah), tidak emosional, dan tidak negatif
    thanos_score = ((100 - diversity) * 0.6) + ((100 - (negativity + anger)) * 0.4)

    archetype = "Komunitas Seimbang/Netral"
    if joker_score > 55:
        archetype = "Arketipe Joker: Komunitas Reaktif & Hiper-Nyata"
    elif thanos_score > 65:
        archetype = "Arketipe Thanos: Komunitas Fokus & Seragam"
    
    return {
        "predicted_archetype": archetype,
        "details": f"Kecenderungan Joker: {round(joker_score, 2)}/100, Kecenderungan Thanos: {round(thanos_score, 2)}/100"
    }
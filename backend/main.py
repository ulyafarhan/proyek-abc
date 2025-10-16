# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from modules.youtube_fetcher import get_all_video_ids_from_channel, get_comments_from_videos
from modules.comment_analyzer import (
    analyze_sentiments,
    analyze_emotions,
    calculate_lexical_diversity,
    calculate_simulation_score,
    calculate_reinforcement_score,
    find_top_topics,
    determine_complex_archetype
)
import pandas as pd

app = FastAPI(
    title="Algorithmic Self Analyzer API",
    description="API untuk menganalisis komunitas YouTube berdasarkan paper 'The Algorithmic Self'.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "API is running!"}

@app.get("/analyze_channel/{channel_id}")
def analyze_channel(channel_id: str):
    video_ids = get_all_video_ids_from_channel(channel_id)
    if not video_ids:
        raise HTTPException(status_code=404, detail="Channel tidak ditemukan atau tidak memiliki video.")
    
    comments = get_comments_from_videos(video_ids)
    if not comments:
        raise HTTPException(status_code=404, detail="Tidak ada komentar yang bisa dianalisis.")

    # Gunakan DataFrame untuk efisiensi
    comments_df = pd.DataFrame(comments)

    # --- Lakukan semua analisis ---
    sentiment_summary = analyze_sentiments(comments_df)
    emotion_scores = analyze_emotions(comments_df)
    diversity_score = calculate_lexical_diversity(comments_df)
    simulation_score = calculate_simulation_score(comments_df)
    reinforcement_score = calculate_reinforcement_score(comments_df)
    top_topics = find_top_topics(comments_df)

    # Gabungkan metrik untuk fungsi sintesis
    all_metrics = {
        "sentiment": sentiment_summary,
        "emotion": emotion_scores,
        "diversity_score": diversity_score,
        "simulation_score": simulation_score,
    }
    archetype_analysis = determine_complex_archetype(all_metrics)

    # --- Susun respons JSON yang komprehensif ---
    return {
        "analysis_summary": {
            "channel_id": channel_id,
            "total_videos_scanned": len(video_ids),
            "total_comments_analyzed": len(comments),
        },
        "archetype_diagnosis": archetype_analysis,
        "linguistic_analysis": {
            "sentiment_summary": sentiment_summary,
            "emotion_distribution": emotion_scores,
            "main_topics": top_topics,
            "lexical_diversity_percent": diversity_score,
        },
        "theoretical_metrics": {
            "baudrillard_simulation_score": simulation_score,
            "skinner_reinforcement_score": reinforcement_score,
        }
    }
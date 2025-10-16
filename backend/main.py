from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from modules.youtube_fetcher import parse_youtube_input, get_video_ids_from_channel, get_comments_from_videos
from modules.gemini_analyzer import get_intelligent_analysis_from_gemini
from modules.comment_analyzer import (
    add_sentiment_scores_to_df,
    analyze_emotions_hf,
    calculate_lexical_diversity,
    calculate_reinforcement_score,
    calculate_archetype_scores_from_gemini
)
import pandas as pd

app = FastAPI(title="Algorithmic Self Analyzer API", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/analyze")
def analyze_youtube_target(target: str = Query(..., description="YouTube Channel ID, Video URL, atau Channel URL")):
    parsed_input = parse_youtube_input(target)
    if parsed_input["type"] == "unknown":
        raise HTTPException(status_code=400, detail="Input tidak valid.")

    video_ids = [parsed_input["id"]] if parsed_input["type"] == "video" else get_video_ids_from_channel(parsed_input["id"])
    if not video_ids:
        raise HTTPException(status_code=404, detail="Tidak ada video ditemukan.")
    
    comments = get_comments_from_videos(video_ids)
    if not comments:
        raise HTTPException(status_code=404, detail="Tidak ada komentar yang bisa dianalisis.")

    comments_df = pd.DataFrame(comments)

    # --- ALUR ANALISIS HIBRIDA TIGA TAHAP ---

    # Tahap 1 (Pra-Filter): Tambahkan skor sentimen ke setiap komentar (gratis)
    comments_df = add_sentiment_scores_to_df(comments_df)

    # Tahap 2 (Konsultasi Cerdas): Kirim sampel ke Gemini (1 API Call)
    gemini_analysis = get_intelligent_analysis_from_gemini(comments_df)

    # Tahap 3 (Skoring Lokal): Hitung skor berdasarkan "kecerdasan" Gemini (gratis)
    archetype_scores = calculate_archetype_scores_from_gemini(comments_df, gemini_analysis)
    
    # Lakukan analisis pendukung lainnya (gratis)
    emotion_scores = analyze_emotions_hf(comments_df)
    diversity_score = calculate_lexical_diversity(comments_df)
    reinforcement_score = calculate_reinforcement_score(comments_df)

    # Tentukan arketipe akhir
    archetype = "Komunitas Seimbang/Netral"
    if archetype_scores["joker_score"] > 5.0: # Ambang batas disesuaikan
        archetype = "Arketipe Joker: Komunitas Reaktif & Anarkis"
    elif archetype_scores["thanos_score"] > 5.0:
        archetype = "Arketipe Thanos: Komunitas Logis & Ekstrem"
    
    return {
        "analysis_summary": {
            "input_type": parsed_input["type"],
            "total_comments_analyzed": len(comments),
        },
        "archetype_diagnosis": {
            "predicted_archetype": archetype,
            "details": gemini_analysis.get("analysis_summary")
        },
        "gemini_context_analysis": {
            "community_vibe": gemini_analysis.get("community_vibe"),
            "joker_keywords_detected": gemini_analysis.get("joker_keywords"),
            "thanos_keywords_detected": gemini_analysis.get("thanos_keywords"),
        },
        "quantitative_metrics": {
            "joker_score": archetype_scores["joker_score"],
            "thanos_score": archetype_scores["thanos_score"],
            "skinner_reinforcement_score": reinforcement_score,
            "lexical_diversity_percent": diversity_score,
        },
        "emotion_distribution": emotion_scores
    }
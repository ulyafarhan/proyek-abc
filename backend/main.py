import models
import database
import schemas
import auth
from modules.youtube_fetcher import parse_youtube_input, get_video_ids_from_channel, get_comments_from_videos
from modules.gemini_analyzer import get_intelligent_analysis_from_gemini, get_brainrot_analysis
from modules.comment_analyzer import (
    add_sentiment_scores_to_df,
    analyze_emotions_hf,
    calculate_lexical_diversity,
    calculate_reinforcement_score,
    calculate_archetype_scores_from_gemini
)
from modules.anima_path_generator import generate_recovery_plan

# Buat tabel di database saat aplikasi pertama kali dijalankan
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="PsycheMap Anima API", version="2.0.0")

# Konfigurasi CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Pastikan ini sesuai dengan port frontend Anda
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cache = TTLCache(maxsize=100, ttl=3600)

# Dependensi untuk mendapatkan sesi database
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# === AUTHENTICATION ENDPOINTS ===
@app.post("/api/register", response_model=schemas.User)
def register_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = auth.get_user(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username sudah terdaftar")
    
    hashed_password = auth.get_password_hash(user.password)
    new_user = models.User(username=user.username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/api/token", response_model=schemas.Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = auth.authenticate_user(db, username=form_data.username, password=form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Username atau password salah",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = auth.create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/users/me", response_model=schemas.User)
def read_users_me(current_user: models.User = Depends(auth.get_current_active_user)):
    return current_user

# === ANALYSIS ENDPOINTS (AUTHENTICATED) ===
@app.get("/api/analyze_youtube")
def analyze_youtube_target(
    target: str = Query(..., description="YouTube Channel ID, Video URL, atau Channel URL"),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_active_user)
):
    if target in cache:
        print(f"Mengambil hasil dari cache untuk: {target}")
        return cache[target]

    print(f"Melakukan analisis penuh untuk: {target}")
    parsed_input = parse_youtube_input(target)
    if parsed_input["type"] == "unknown":
        raise HTTPException(status_code=400, detail="Input URL YouTube tidak valid.")

    video_ids = [parsed_input["id"]] if parsed_input["type"] == "video" else get_video_ids_from_channel(parsed_input["id"])
    if not video_ids:
        raise HTTPException(status_code=404, detail="Tidak ada video ditemukan.")
    
    comments = get_comments_from_videos(video_ids)
    if not comments:
        raise HTTPException(status_code=404, detail="Tidak ada komentar yang bisa dianalisis.")

    comments_df = pd.DataFrame(comments)
    comments_df = add_sentiment_scores_to_df(comments_df)
    gemini_analysis = get_intelligent_analysis_from_gemini(comments_df)
    archetype_scores = calculate_archetype_scores_from_gemini(comments_df, gemini_analysis)
    emotion_scores = analyze_emotions_hf(comments_df)
    diversity_score = calculate_lexical_diversity(comments_df)
    reinforcement_score = calculate_reinforcement_score(comments_df)
    
    sentiment_counts = comments_df['sentiment_label'].value_counts(normalize=True) * 100
    sentiment_aggregation = {
        'positive_percent': sentiment_counts.get('positive', 0),
        'negative_percent': sentiment_counts.get('negative', 0),
        'neutral_percent': sentiment_counts.get('neutral', 0)
    }

    archetype = "Komunitas Seimbang/Netral"
    if archetype_scores.get("joker_score", 0) > 50:
        archetype = "Arketipe Joker: Komunitas Reaktif & Anarkis"
    elif archetype_scores.get("thanos_score", 0) > 50:
        archetype = "Arketipe Thanos: Komunitas Logis & Ekstrem"
    
    final_result = {
        "analysis_summary": { "input_type": parsed_input["type"], "total_comments_analyzed": len(comments) },
        "archetype_diagnosis": { "predicted_archetype": archetype, "details": gemini_analysis.get("analysis_summary") },
        "gemini_context_analysis": { "community_vibe": gemini_analysis.get("community_vibe"), "joker_keywords_detected": gemini_analysis.get("joker_keywords"), "thanos_keywords_detected": gemini_analysis.get("thanos_keywords"), "main_themes": gemini_analysis.get("main_themes", []) },
        "quantitative_metrics": { "joker_score": archetype_scores.get("joker_score", 0), "thanos_score": archetype_scores.get("thanos_score", 0), "skinner_reinforcement_score": reinforcement_score, "lexical_diversity_percent": diversity_score },
        "emotion_distribution": emotion_scores,
        "sentiment_distribution": sentiment_aggregation
    }
    
    new_analysis = models.Analysis(analysis_type="community", result_json=json.dumps(final_result), owner_id=current_user.id)
    db.add(new_analysis)
    db.commit()
    
    cache[target] = final_result
    return final_result

@app.post("/api/analyze_behavior")
def analyze_user_behavior(
    activities: List[schemas.UserActivity], 
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_active_user)
):
    try:
        activities_dict = [activity.dict() for activity in activities]
        analysis_result = get_brainrot_analysis(activities_dict)
        
        new_analysis = models.Analysis(analysis_type="behavior", result_json=json.dumps(analysis_result), owner_id=current_user.id)
        db.add(new_analysis)
        db.commit()
        
        return analysis_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/journal", response_model=schemas.JournalEntry)
def create_journal_entry(
    entry: schemas.JournalEntryCreate, 
    db: Session = Depends(get_db), 
    current_user: models.User = Depends(auth.get_current_active_user)
):
    analysis_result = {"insight": "Analisis dari Gemini untuk entri jurnal ini akan muncul di sini.", "scores": [7, 6, 8, 5, 9]}
    
    db_entry = models.JournalEntry(content=entry.content, analysis_json=json.dumps(analysis_result), owner_id=current_user.id)
    db.add(db_entry)
    db.commit()
    db.refresh(db_entry)
    return db_entry

@app.get("/api/journal", response_model=List[schemas.JournalEntry])
def get_journal_entries(current_user: models.User = Depends(auth.get_current_active_user), db: Session = Depends(get_db)):
    return db.query(models.JournalEntry).filter(models.JournalEntry.owner_id == current_user.id).order_by(models.JournalEntry.timestamp.desc()).all()

@app.get("/api/anima_path")
def get_anima_path(current_user: models.User = Depends(auth.get_current_active_user)):
    user_profile = {"dominant_archetype": "Joker", "low_focus_score": True}
    plan = generate_recovery_plan(user_profile)
    return plan

@app.get("/api/dashboard_data")
def get_dashboard_data(current_user: models.User = Depends(auth.get_current_active_user), db: Session = Depends(get_db)):
    return {"message": f"Data dasbor untuk {current_user.username}"}

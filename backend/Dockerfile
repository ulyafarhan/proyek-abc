# Gunakan base image Python 3.11 (sesuai dengan file .pyc Anda)
FROM python:3.11-slim

# Set direktori kerja di dalam container
WORKDIR /app

# Salin file requirements terlebih dahulu untuk caching layer
COPY requirements.txt .

# Instal dependensi
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua kode aplikasi
COPY . .

# Muat model AI saat build (opsional tapi bagus)
# Ini akan "memanaskan" cache model sehingga startup server lebih cepat
RUN python -c "from modules.comment_analyzer import sentiment_pipeline, emotion_pipeline"

# Ekspos port yang digunakan FastAPI
EXPOSE 8000

# Perintah untuk menjalankan aplikasi saat container dimulai
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
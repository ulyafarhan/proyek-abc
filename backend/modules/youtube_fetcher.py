# modules/youtube_fetcher.py
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from config import YOUTUBE_API_KEY

try:
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
except Exception as e:
    youtube = None
    print(f"Error saat inisialisasi YouTube service: {e}")
    print("Pastikan YOUTUBE_API_KEY di file .env sudah benar.")

def get_all_video_ids_from_channel(channel_id: str) -> list[str]:
    """Mengambil semua ID video dari sebuah channel."""
    if not youtube:
        return []
    
    try:
        res = youtube.channels().list(id=channel_id, part='contentDetails').execute()
        if not res['items']:
            return []
        playlist_id = res['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        
        video_ids = []
        next_page_token = None
        
        while True:
            res = youtube.playlistItems().list(
                playlistId=playlist_id,
                part='contentDetails',
                maxResults=50,
                pageToken=next_page_token
            ).execute()
            
            video_ids.extend([item['contentDetails']['videoId'] for item in res['items']])
            next_page_token = res.get('nextPageToken')
            
            if next_page_token is None:
                break
                
        return video_ids
    except HttpError as e:
        print(f"Error HTTP saat mengambil video: {e}")
        return []

def get_comments_from_videos(video_ids: list[str], max_videos: int = 10, max_comments_per_video: int = 50) -> list[dict]:
    """Mengambil komentar dari daftar video, dengan batasan untuk efisiensi."""
    if not youtube:
        return []
    
    all_comments = []
    # Batasi jumlah video yang diproses agar demo tidak terlalu lama.
    # Hapus slicing [:max_videos] untuk memproses semua video.
    for video_id in video_ids[:max_videos]:
        try:
            next_page_token = None
            comment_count = 0
            while True:
                # Batasi total komentar yang diambil per video
                if comment_count >= max_comments_per_video:
                    break

                res = youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    maxResults=100, # Ambil 100 per halaman
                    pageToken=next_page_token,
                    textFormat='plainText'
                ).execute()
                
                for item in res['items']:
                    comment = item['snippet']['topLevelComment']['snippet']
                    all_comments.append({
                        'video_id': video_id,
                        'text': comment['textDisplay'],
                        'author': comment['authorDisplayName'],
                        'published_at': comment['publishedAt']
                    })
                    comment_count += 1
                
                next_page_token = res.get('nextPageToken')
                if next_page_token is None:
                    break
        except HttpError as e:
            # Sering terjadi jika komentar dinonaktifkan
            print(f"Tidak bisa mengambil komentar untuk video {video_id}: {e}")
            continue
            
    return all_comments
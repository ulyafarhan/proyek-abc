import re
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from config import YOUTUBE_API_KEY

try:
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
except Exception as e:
    youtube = None
    print(f"Error saat inisialisasi YouTube service: {e}")

def get_video_ids_from_channel(channel_id: str) -> list[str]:
    if not youtube: return []
    try:
        res = youtube.channels().list(id=channel_id, part='contentDetails').execute()
        if not res.get('items'): return []
        playlist_id = res['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        
        video_ids = []
        next_page_token = None
        while True:
            res = youtube.playlistItems().list(
                playlistId=playlist_id, part='contentDetails',
                maxResults=50, pageToken=next_page_token
            ).execute()
            video_ids.extend([item['contentDetails']['videoId'] for item in res.get('items', [])])
            next_page_token = res.get('nextPageToken')
            if next_page_token is None:
                break
        return video_ids
    except HttpError as e:
        print(f"Error HTTP saat mengambil video dari channel: {e}")
        return []

def get_comments_from_videos(video_ids: list[str], max_videos: int = 10, max_comments_per_video: int = 50) -> list[dict]:
    if not youtube: return []
    all_comments = []
    for video_id in video_ids[:max_videos]:
        try:
            next_page_token = None
            comment_count = 0
            while comment_count < max_comments_per_video:
                res = youtube.commentThreads().list(
                    part='snippet', videoId=video_id, maxResults=100,
                    pageToken=next_page_token, textFormat='plainText'
                ).execute()
                
                for item in res.get('items', []):
                    comment = item['snippet']['topLevelComment']['snippet']
                    all_comments.append({
                        'video_id': video_id,
                        'text': comment.get('textDisplay', ''),
                    })
                    comment_count += 1
                
                next_page_token = res.get('nextPageToken')
                if next_page_token is None:
                    break
        except HttpError as e:
            print(f"Tidak bisa mengambil komentar untuk video {video_id}: {e}")
            continue
    return all_comments

def parse_youtube_input(youtube_input: str) -> dict:
    video_id_pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11})'
    channel_id_pattern = r'(?:channel\/|c\/|@)([^\/\?&]+)'

    video_match = re.search(video_id_pattern, youtube_input)
    if video_match:
        return {"type": "video", "id": video_match.group(1)}

    channel_match = re.search(channel_id_pattern, youtube_input)
    if channel_match:
        try:
            search_response = youtube.search().list(
                q=channel_match.group(1), part='id', type='channel', maxResults=1
            ).execute()
            if search_response.get("items"):
                return {"type": "channel", "id": search_response['items'][0]['id']['channelId']}
        except HttpError as e:
            print(f"Gagal mengkonversi handle/username ke Channel ID: {e}")
    
    if youtube_input.startswith("UC"):
        return {"type": "channel", "id": youtube_input}
        
    return {"type": "unknown", "id": None}
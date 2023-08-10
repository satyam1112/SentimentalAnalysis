
from googleapiclient.discovery import build

# Replace with your own API key
api_key = 'AIzaSyB_d69XgLRxeoN__MfEMF93vi6ZqwQkJxo'

# Initialize the YouTube Data API
from urllib.parse import urlparse, parse_qs

def extract_video_id(url):
    parsed_url = urlparse(url)
    
    if parsed_url.netloc == 'www.youtube.com':
        if parsed_url.path == '/watch':
            video_id = parse_qs(parsed_url.query).get('v')
            if video_id:
                return video_id[0]
        elif parsed_url.path.startswith('/embed/'):
            return parsed_url.path.split('/')[-1]
    elif parsed_url.netloc == 'youtu.be':
        return parsed_url.path[1:]
    
    return None
youtube = build('youtube', 'v3', developerKey=api_key)

# Specify the video ID
url='https://www.youtube.com/watch?v=_KvtVk8Gk1A'
video_id = extract_video_id(url)
# https://youtu.be/7UVoCmolAPI

# Retrieve comments from the video
comments = []
nextPageToken = None
while True:
    response = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        textFormat='plainText',
        maxResults=100,
        pageToken=nextPageToken
    ).execute()

    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comments.append(comment)

    nextPageToken = response.get('nextPageToken')
    if not nextPageToken:
        break

# Save the comments to a file
with open('comments.txt', 'w', encoding='utf-8') as f:
    for comment in comments:
        f.write(comment + '\n')

print(f'Successfully scraped {len(comments)} comments and saved to "comments.txt".')

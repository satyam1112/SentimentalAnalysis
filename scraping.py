
from googleapiclient.discovery import build

# Replace with your own API key
api_key = 'AIzaSyB_d69XgLRxeoN__MfEMF93vi6ZqwQkJxo'

# Initialize the YouTube Data API
youtube = build('youtube', 'v3', developerKey=api_key)

# Specify the video ID
video_id = 'vIBHZ7FfWUA'

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

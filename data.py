from pytube import YouTube
from ytcomments import YTComments

# URL of the YouTube video
video_url = 'https://www.youtube.com/watch?v=p2zMXSXhZ9M'

# Download the video
yt = YouTube(video_url)
video_stream = yt.streams.get_highest_resolution()
video_stream.download(output_path='downloads', filename='video')

# Initialize YTComments with the video ID
comments = YTComments(yt.video_id)

# Get comments from the video
# all_comments = comments.get_comments()
desired_comment_count = 100
all_comments = []
for comment in comments.get_comments():
    all_comments.append(comment)
    if len(all_comments) >= desired_comment_count:
        break

# Save the comments to a file
with open('comments.txt', 'w', encoding='utf-8') as f:
    for comment in all_comments:
        f.write(comment + '\n')

print(f'Successfully scraped {len(all_comments)} comments and saved to "comments.txt".')

from transformers import AutoTokenizer,AutoConfig
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
from aiohttp import web
from scraping import extract_comment,extract_video_id
from aiohttp import web
from random import choice
from langdetect import detect

import re
def remove_non_english_sentences(text):
    sentences = text.split('\n')  # Split text into sentences (assuming each line is a sentence)
    english_sentences = [sentence for sentence in sentences if detect(sentence) == 'en']
    cleaned_text = '\n'.join(english_sentences)
    return cleaned_text
def clean_text(text):
    # Remove emojis
    text = re.sub(r'[\U0001F600-\U0001F6FF]', '', text)
    
    # Remove non-alphanumeric characters and extra spaces
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove comments with view/like/comment counts
    text = re.sub(r'\d+k\scomments|\d+milion\slikes|\d+b\sviews', '', text, flags=re.IGNORECASE)
    
    # Remove timestamps and irrelevant text
    text = re.sub(r'\d+:\d+\s(am|pm)|\d+/\d+/\d+\sح/س|master\s?piece|for\swhat\s?exactly', '', text, flags=re.IGNORECASE)
    
    return text
def predict(l):
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    # filename='comments.txt'
    print("List of comments\n",l)
    # lines = [a.strip() for a in open(filename).readlines()]
    result = [choice(l) for a in range(40)]
    print("\n\n\n")
    print("size\n",len(l))
    text="".join(result)
    text=clean_text(text)
    print("Clean text",text)
    text=remove_non_english_sentences(text)
    print("Comments text\n",text)
    # res={}
    # max_chunk_size = 512  # Maximum token limit
    # chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]

# Perform sentiment analysis on each chunk
#     sentiment_scores = []
#     for chunk in chunks:
#         encoded_input = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True)
#         output = model(**encoded_input)
#         scores = output.logits.softmax(dim=-1).detach().numpy()
#         sentiment_scores.extend(scores)

# # Calculate average sentiment scores across chunks
#     average_scores = np.mean(sentiment_scores, axis=0)
#     sorted_scores = sorted([(label, score) for label, score in zip(config.id2label.values(), average_scores)], key=lambda x: x[1], reverse=True)

# Display sentiment results
    # for i, (label, score) in enumerate(sorted_scores):
    #     res[label]=np.round(float(score), 4)
    #     print(f"{i+1}) {label}: {np.round(score, 4)}")

    encoded_input = tokenizer(text, return_tensors='pt',padding=True,truncation=True)
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()

    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    res={}
    for i in range(scores.shape[0]):
        l = config.id2label[ranking[i]]
        s = scores[ranking[i]]
        res[l]=np.round(float(s), 4)
        print(f"{i+1}) {l} {np.round(float(s), 4)}")
    return res
async def func(req):
    return web.Response(text="hello world")

async def func2(req):
    data = await req.text()
    id=extract_video_id(data)
    ls=extract_comment(id)
    res=predict(ls)
    return web.json_response(res)
    # return web.Response(text=f"Received: {data}") 



app=web.Application()
app.add_routes([web.get("/",func)])
app.router.add_route('POST', '/url', func2)
web.run_app(app,port=8989,host="127.0.0.1")



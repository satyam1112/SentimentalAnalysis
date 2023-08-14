from transformers import AutoTokenizer,AutoConfig
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
from aiohttp import web
from scraping import extract_comment,extract_video_id
from aiohttp import web
from random import choice
def predict():
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    filename='comments.txt'
    
    lines = [a.strip() for a in open(filename).readlines()]
    result = [choice(lines) for a in range(10)]
    print("\n\n\n")
    text="".join(result).lower()
    # print(text)
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
    extract_comment(id)
    res=predict()
    return web.json_response(res)
    # return web.Response(text=f"Received: {data}") 



app=web.Application()
app.add_routes([web.get("/",func)])
app.router.add_route('POST', '/url', func2)
web.run_app(app,port=8989,host="127.0.0.1")



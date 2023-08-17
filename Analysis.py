from transformers import AutoTokenizer,AutoConfig
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
from aiohttp import web
from scraping import extract_comment,extract_video_id
from aiohttp import web
from random import choice
from transformers import pipeline
from swagger_ui import api_doc
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
import tensorflow as tf


# from langdetect import detect

import re
# def remove_non_english_sentences(text):
#     sentences = text.split('\n')  # Split text into sentences (assuming each line is a sentence)
#     english_sentences = [sentence for sentence in sentences if detect(sentence) == 'en']
#     cleaned_text = '\n'.join(english_sentences)
#     return cleaned_text
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
    # print("Length \n",len(l))
    # print("List of comments\n",l)
    l=list(map(clean_text,l))
    # print("Length \n",len(l))
    print("After cleaning ,list of comments are\n",l)
    # result = [choice(l) for a in range(50)]
    print("\n\n\n")
    # print("size\n",len(l))
    text=" ".join(l)
    text=clean_text(text)
    # print("Res length\n",len(result))
    # text=remove_non_english_sentences(text)
    print("Comments text\n",text)
    res={}
    max_chunk_size = 512  # Maximum token limit
    chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]

# Perform sentiment analysis on each chunk
    sentiment_scores = []
    for chunk in chunks:
        encoded_input = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True)
        output = model(**encoded_input)
        scores = output.logits.softmax(dim=-1).detach().numpy()
        sentiment_scores.extend(scores)

# # Calculate average sentiment scores across chunks
    average_scores = np.mean(sentiment_scores, axis=0)
    sorted_scores = sorted([(label, score) for label, score in zip(config.id2label.values(), average_scores)], key=lambda x: x[1], reverse=True)

# Display sentiment results
    print("\n\n")
    print("length of comments are\n",len(l))

    for i, (label, score) in enumerate(sorted_scores):
        res[label]=np.round(float(score*100), 2)
        print(f"{i+1}) {label}: {np.round(score*100, 2)}")

    # encoded_input = tokenizer(text, return_tensors='pt',padding=True,truncation=True)
    # output = model(**encoded_input)
    # scores = output.logits[0].detach().numpy()

    # scores = softmax(scores)
    # ranking = np.argsort(scores,axis=-1)
    # ranking = ranking[::-1]
    # res={}
    # for i in range(scores.shape[0]):
    #     l = config.id2label[ranking[i]]
    #     s = scores[ranking[i]]
    #     percentage_score = np.round(float(s) * 100, 2)  # Convert to percentage and round to 2 decimal places
    #     res[l] = f"{percentage_score}%"
    #     print(f"{i+1}) {l} {percentage_score}%")
    return res
# async def func(req):
#     return web.Response(text="hello world")

async def sent_analysis(req):
    data = await req.text()
    id=extract_video_id(data)
    if id is None:
        res=predict([data])
    else:   
        ls=extract_comment(id)
        res=predict(ls)
    return web.json_response(res)
async def emotion_analysis(req):
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    data = await req.text()
    res=classifier(data)
    sorted_res = sorted(res[0], key=lambda x: x['score'], reverse=True)
    res={}
    for data in sorted_res:
        label = data['label']
        score = np.round(float(data['score']), 4)
        res[label]=score
        print(f"{label}    {score}")

    return web.json_response(res)

async def spam_ham(req):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = TFAutoModelForSequenceClassification.from_pretrained("my_model")
    data = await req.text()
    new_message_enc = tokenizer(data, padding=True, truncation=True)
    input_ids = tf.convert_to_tensor(new_message_enc['input_ids'])
    attention_mask = tf.convert_to_tensor(new_message_enc['attention_mask'])
    logits = model([input_ids, attention_mask])[0]
    probs = tf.nn.softmax(logits, axis=-1)
    predicted_label = tf.argmax(probs, axis=-1).numpy()[0]
    res={}
    if predicted_label == 1:
        print("Predicted Label: Spam")
        res["Predicted Label"]="Spam"
        print(f"Class Probabilities: Spam: {probs[0][1]:.4f}, Not Spam: {probs[0][0]:.4f}")
    else:
        res["Predicted Label"]="Not Spam"
        print("Predicted Label: Not Spam")
        print(f"Class Probabilities: Spam: {probs[0][1]:.4f}, Not Spam: {probs[0][0]:.4f}")
    res["Spam"]=f"{probs[0][1]:.4f}"
    res["Not Spam"]=f"{probs[0][0]:.4f}"
    
app=web.Application()
# app.add_routes([web.get("/",func)])
app.router.add_route('POST', '/sent_analysis', sent_analysis)
app.router.add_route('POST','/emotion',emotion_analysis)
app.router.add_route('POST','/spam_ham',spam_ham)
api_doc(app, config_path=r"/home/formbay/Desktop/SentimentalAnalysis/openapi.yml", url_prefix="/", title="API doc")
web.run_app(app,port=8989,host="127.0.0.1")




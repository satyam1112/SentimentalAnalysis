from transformers import AutoTokenizer,AutoConfig
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
text1 = """There was a time when the social media
services like Facebook used to just have two emotions associated with
each post, i.e You can like a post or you can leave the post without any
reaction and that basically signifies that you didn’t like it"""
filename='comments.txt'
from random import choice
lines = [a.strip() for a in open(filename).readlines()]
result = [choice(lines) for a in range(30)]
print("\n\n\n")
text="".join(result).lower()
print(text)
# text="If you can stay positive in a negative situation, you win"
# text = preprocess(text)
encoded_input = tokenizer(text, return_tensors='pt')
# # print(encoded_input)
output = model(**encoded_input)
# # print("Output ",output)
# # print(type(output))
scores = output[0][0].detach().numpy()
# print("Scores    ",scores)
scores = softmax(scores)
# print("Scores  ->   ",scores)
# # Print labels and scores
ranking = np.argsort(scores)
# # print("ranking  ->   ",ranking)

ranking = ranking[::-1]
# # print("ranking  ->   ",ranking)

for i in range(scores.shape[0]):
    l = config.id2label[ranking[i]]
    s = scores[ranking[i]]
    print(f"{i+1}) {l} {np.round(float(s), 4)}")
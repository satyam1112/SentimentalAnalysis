from transformers import pipeline
from transformers import pipeline
import numpy as np
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
res=classifier("")
sorted_res = sorted(res[0], key=lambda x: x['score'], reverse=True)

for data in sorted_res:
    label = data['label']
    score = np.round(float(data['score']), 4)
    print(f"{label}    {score}")
        


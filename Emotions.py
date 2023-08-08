from transformers import pipeline
from transformers import pipeline
import numpy as np
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
res=classifier("spirit lead me where my trust is without bordersdeath stranding music blissfullife is something we’ve been blessed,em pleno 2022...thank you for this, this is what i needed😡🤜😣🍩 😢the first one 😅✌️so amazing bro 😭nao conssigo parar de cantar 🙌🏼🙌🏼🙌🏼🙌🏼please do double take i know for fact u're going to make it fireи свет дыханьем полыхал.god's love letter for you...(worth watching) / jologsmehthis song is like walking in heaven while you’re highthank you for doing thisremember, the devil can never take away the gift god gave you at birth but he is strong enough to blind you from that gift. i love everyone one of you. open our eyes father and help us understand that it is not the size of faith but realize that its all about who our faith is in.❤❤❤❤and my faith will be made strongerwhen oceans risethank you ryanand my faith will be made strongerwe never know when we’ll perish.this made me cry abt all the sins i've committed i really want to change..i love this song fully, perfect to go to sleep to or just chill in the gospel 🤗😊✨✨💒in the presence of my saviour")
print("Output\n")
for r in res:
    for data in r:
        print(f"{data['label']}    {np.round(float(data['score']), 4)}")


        


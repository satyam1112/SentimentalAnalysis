from transformers import DistilBertTokenizerFast
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
df=pd.read_csv("dataset_spam_ham.csv",sep="\t",names=["label","message"])
X=list(df["message"])
Y=df["label"]
Y=list(pd.get_dummies(Y,drop_first=True)['spam'])
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=0)
tokenizer=DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
train_enc=tokenizer(x_train,padding=True,truncation=True)
test_enc=tokenizer(x_test,padding=True,truncation=True)



train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_enc),
    y_train
))

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_enc),
    y_test
))
batch_size = 10
num_epochs = 1
model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


model.fit(train_dataset.shuffle(1000).batch(batch_size),
          validation_data=test_dataset.batch(batch_size),
          epochs=num_epochs)

test_loss, test_accuracy = model.evaluate(test_dataset.batch(batch_size))
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)
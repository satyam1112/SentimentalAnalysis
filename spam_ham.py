import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import torch
import torch.nn as nn


from torch.utils.data import Dataset, DataLoader
import numpy as np
# import tensorflow as tf
# from tensorflow.keras import activations, optimizers, losses
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from transformers import DistilBertForSequenceClassification,Trainer,TrainingArguments
df=pd.read_csv("dataset_spam_ham.csv",sep="\t",names=["label","message"])
X=list(df["message"])
Y=df["label"]
Y=list(pd.get_dummies(Y,drop_first=True)['spam'])
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=0)
tokenizer=DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
train_enc=tokenizer(x_train,padding=True,truncation=True,return_tensors='pt')
test_enc=tokenizer(x_test,padding=True,truncation=True,return_tensors='pt')


# Create torch dataset


class Dataset(Dataset):
    def __init__(self, encodings, labels=None, max_length=128):
        self.encodings = encodings
        self.labels = labels
        self.max_length = max_length

    def __getitem__(self, idx):
        item = {
            key: val[idx][:self.max_length] for key, val in self.encodings.items()
        }
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels) if self.labels else len(self.encodings["input_ids"])



# def compute_metrics(p):    
#     pred, labels = p
#     pred = np.argmax(pred, axis=1)
#     accuracy = accuracy_score(y_true=labels, y_pred=pred)
#     recall = recall_score(y_true=labels, y_pred=pred)
#     precision = precision_score(y_true=labels, y_pred=pred)
#     f1 = f1_score(y_true=labels, y_pred=pred)
#     return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


train_data=Dataset(train_enc,y_train,max_length=128)
test_data=Dataset(test_enc,y_test,max_length=128)
model=DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",num_labels=2)


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    save_steps=500,
    logging_dir="./logs",
)
print("Shapes are \n\n")
print(train_enc["input_ids"].shape)
print(test_enc["input_ids"].shape)
print(len(y_train))
print(len(y_test))
print(len(train_data))
print(len(test_data))

# train_args=TrainingArguments(num_train_epochs=2,output_dir="./results",per_device_train_batch_size=16,eval_steps=100
#                             ,per_device_eval_batch_size=64,warmup_steps=500,weight_decay=0.01,
#                             logging_dir="./logs",logging_steps=10)

# # print(train_args)


# with train_args.strategy.scope():
#     model=TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",num_labels=1)

# loss=losses.SparseCategoricalCrossentropy(from_logits=True)
# model.compile(optimizer='Adam',metrics=['accuracy'],loss=loss)
# model.fit(train_data,epochs=2,batch_size=16)
# trainer=Trainer(model=model,args=train_args,train_dataset=train_data,eval_dataset=test_data,compute_metrics=compute_metrics)
# print("\n\n")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
)
trainer.train()
eval_results = trainer.evaluate()

print(eval_results)







# trainer.evaluate(test_data)


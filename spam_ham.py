import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import torch
# import tensorflow as tf
# from tensorflow.keras import activations, optimizers, losses

from transformers import DistilBertForSequenceClassification,Trainer,TrainingArguments
df=pd.read_csv("dataset_spam_ham.csv",sep="\t",names=["label","message"])
# print(df.head(10))
# print(df.shape)
X=list(df["message"])
Y=df["label"]
Y=list(pd.get_dummies(Y,drop_first=True)['spam'])
# print(Y)
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=0)
tokenizer=DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
train_enc=tokenizer(x_train,padding=True,truncation=True,return_tensors='tf')

test_enc=tokenizer(x_test,padding=True,truncation=True,return_tensors='tf')
# train_data=tf.data.Dataset.from_tensor_slices((dict(train_enc),y_train))
# # print(train_data)
# test_data=tf.data.Dataset.from_tensor_slices((dict(test_enc),y_test))
# print(test_data)

# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])



train_data=Dataset(train_enc,y_train)
test_data=Dataset(test_enc,y_test)

train_args=TrainingArguments(num_train_epochs=2,output_dir="./results",per_device_train_batch_size=16,eval_steps=100
                            ,per_device_eval_batch_size=64,warmup_steps=500,weight_decay=0.01,
                            logging_dir="./logs",logging_steps=10)

# # print(train_args)
model=DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",num_labels=2)


# with train_args.strategy.scope():
#     model=TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",num_labels=1)

# loss=losses.SparseCategoricalCrossentropy(from_logits=True)
# model.compile(optimizer='Adam',metrics=['accuracy'],loss=loss)
# model.fit(train_data,epochs=2,batch_size=16)
trainer=Trainer(model=model,args=train_args,train_dataset=train_data,eval_dataset=test_data)

# trainer.train()
# trainer.evaluate(test_data)


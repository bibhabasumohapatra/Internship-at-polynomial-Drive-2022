
import io
import torch
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from sklearn.model_selection import train_test_split


from dataset import *
from engine import *
from model import *

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, torkens[1:]))
    return data

def create_embedding_matrix(word_index, embedding_dict):
    
    embedding_matrix = np.zeros((len(word_index)+1, 300))
    
    for word, i in word_index.items():
        if word in embedding_dict:
            embedding_matrix[i] = embedding_dict[word]
    
    return embedding_matrix

df = pd.read_csv('../input/amazon-dataset-csv-generator/PolynomialInternshipDrive2022.csv')
df_train, df_valid = train_test_split(df, test_size = 0.2, shuffle=True)

print('Tokenizer ################ ')

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(df.review.values.tolist())
#save this tokenizer fit
joblib.dump(tokenizer, 'tf_tokenizer.pkl')
xtrain = tokenizer.texts_to_sequences(df_train.review.values)
xtest = tokenizer.texts_to_sequences(df_valid.review.vlaues)
xtrain = tf.keras.preprocessing.sequence.pad_sequences(
 xtrain, maxlen=128
 )
xtest = tf.keras.preprocessing.sequence.pad_sequences(
 xtest, maxlen=128
 )

train_dataset = Amazondataset(
 reviews=xtrain,
 targets=train_df.overall.values
 )

train_data_loader = torch.utils.data.DataLoader(
 train_dataset,
 batch_size=2,
 num_workers=0
 )

valid_dataset = Amazondataset(
 reviews=xtest,
 targets=test_df.overall.values
 )

valid_data_loader = torch.utils.data.DataLoader(
 valid_dataset,
 batch_size=2,
 num_workers=0
 )

embedding_dict = load_vectors("../input/crawl-300d-2M.vec")
 embedding_matrix = create_embedding_matrix(
 tokenizer.word_index, embedding_dict
 )

device = torch.device('cuda')
model = LSTM(embedding_matrix)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_accuracy = 0
early_stopping_counter = 0

for epoch in range(5):
    train(train_data_loader, model, optimizer, device)
    outputs, targets = engine.evaluate(
         valid_data_loader, model, device
         )
    
    
    accuracy = metrics.accuracy_score(targets, outputs)
    confusion_matrix = metrics.confusion_matrix(outputs, targets)

    sns.heatmap(confusion_matrix, annot=True)
    print(f'Epochs : {epoch} accuracy: {accuracy}')

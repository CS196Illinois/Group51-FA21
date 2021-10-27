import numpy as np
import torch
import matplotlib as plt
import pydub
import librosa
import csv
import torchaudio
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
#from torch.utils.data import DataLoader
#from torchvision import datasets, transforms
import torch.utils.data as data_utils

df = pd.read_csv('/Users/balajisampath/Desktop/features.csv')
features = df[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']]
labels = df['labels']
X = features
y = labels

def label2idx(cols):
    mapping = dict()
    for i, emotion in enumerate(cols):
        mapping[emotion] = i
    return mapping

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(df)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
CLASSES = df['labels'].nunique()
real_labels = label2idx(df['labels'].unique())
print(real_labels)
df['labels'] = df['labels'].apply(lambda x: real_labels[x])
print(df['labels'])
y = torch.tensor(df['labels']).long().to(device)


def df_to_tensor(df):
    return torch.tensor(df.values).float().to(device)

train = data_utils.TensorDataset(df_to_tensor(X), y)
train_loader = data_utils.DataLoader(train, batch_size=16, shuffle=True)

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(20, 10)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(10, CLASSES)

        # Define sigmoid activation and softmax output
        self.softmax = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)

        return x

model = Network().to(device)
print(model)

"""model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))"""
# Define the loss
criterion = nn.NLLLoss()

##########
# Logsoft + NLLLoss
# F.cross_entropy()
##########
# Optimizers require the parameters to optimize and a learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=0.003)
epochs = 200
for e in range(epochs):
    running_loss = 0
    for audios, labels in train_loader:
        # Flatten MNIST images into a 784 long vector
        # images = images.view(images.shape[0], -1)

        # Training pass
        optimizer.zero_grad()
        output = model(audios)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss / len(train_loader)}")



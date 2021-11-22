import torch
import pandas as pd
from torch import nn

# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms

import torch.utils.data as data_utils

df = pd.read_csv('/Users/madhav/Downloads/features.csv')
features = df[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']]
labels = df['labels']
X = features
y = labels

def label2idx(cols):
    mapping = dict()
    for i, emotion in enumerate(cols):
        mapping[emotion] = i
    return mapping

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


print(df)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
CLASSES = df['labels'].nunique()
real_labels = label2idx(df['labels'].unique())
print(real_labels)
df['labels'] = df['labels'].apply(lambda x: real_labels[x])
print(df['labels'])
y = torch.tensor(df['labels']).long().to(device)
print(type(y))
train_size = int(0.8 * len(X))
test_size = len(X) - train_size

def df_to_tensor(df):
    return torch.tensor(df.values).float().to(device)

train = data_utils.TensorDataset(df_to_tensor(X), y)

train_dataset, test_dataset = torch.utils.data.random_split(train, [train_size, test_size])

train_loader = data_utils.DataLoader(train_dataset, batch_size=16, shuffle=True)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


"""
OLD STUFF if we need later
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
"""


n_hidden = 128
Network = RNN(20, n_hidden, CLASSES)
model = Network

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
optimizer = tooptimizer = torch.optim.SGD(model.parameters(), lr=0.003)
epochs = 20
for e in range(epochs):
    running_loss = 0
    for audios, labels in train_loader:
        # Flatten MNIST images into a 784 long vector
        # images = images.view(images.shape[0], -1)

        # Training pass
        optimizer.zero_grad()
        hidden = model.initHidden()
        output = model(audios)
        loss = criterion(output, labels)


        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss / len(train_loader)}")
torch.optim.SGD(model.parameters(), lr=0.003)
epochs = 30
for e in range(epochs):
    running_loss = 0
    for audios, labels in train_loader:
        # Flatten MNIST images into a 784 long vector
        # images = images.view(images.shape[0], -1)

        # Training pass
        optimizer.zero_grad()
        hidden = model.initHidden()
        output = model(audios)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss / len(train_loader)}")

# Print about testing
print('Starting testing')

# Saving the model
save_path = './mlp.pth'
torch.save(model.state_dict(), save_path)

# Testing loop
correct, total = 0, 0
with torch.no_grad():
    # Iterate over the test data and generate predictions
    for i, data in enumerate(test_dataset, 0):
        # Get inputs
        features, labels = data

        # Generate outputs
        print(type(model))
        print(type(features))
        outputs = model(features)

        # Set total and correct
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Print accuracy
    print('Accuracy: %d %%' % (100 * correct / total))

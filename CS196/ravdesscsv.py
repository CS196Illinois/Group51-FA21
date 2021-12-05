import torch
import pandas as pd
from torch import nn
import torch.utils.data as data_utils

# Getting data from features.csv and extracting features and labels
df = pd.read_csv('/Users/balajisampath/Desktop/features.csv')
print(df)
features = df[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']]
labels = df['labels']
X = features
y = labels

# labels are in the form of string so mapping to integer value
def label2idx(cols):
    mapping = dict()
    for i, emotion in enumerate(cols):
        mapping[emotion] = i
    return mapping

device = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASSES = df['labels'].nunique()

# Only taking unique labels since categorical data
real_labels = label2idx(df['labels'].unique())
df['labels'] = df['labels'].apply(lambda x: real_labels[x])
y = torch.tensor(df['labels']).long().to(device)

# Converting data from Pandas dataframe to Torch tensor
def df_to_tensor(df):
    return torch.tensor(df.values).float().to(device)

train = data_utils.TensorDataset(df_to_tensor(X), y)

# Splitting data into training and testing data
train_size = int(0.8 * len(X))
test_size = len(X) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(train, [train_size, test_size])
train_loader = data_utils.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = data_utils.DataLoader(test_dataset, batch_size=16, shuffle=True)


# Creating the RNN class
class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.batch_size = 16
        self.hidden = self.init_hidden(16)

        # Defining the layers

        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim*20, output_size)

    def forward(self, x):
        # Passing in the input and hidden state into the model and obtaining outputs
        # [0.1, 0.44, 0.74, ...] size=20
        # [[0.1]
        #  [0.44]]   size = 20 * 1
        x = x.unsqueeze(2)
        out, self.hidden = self.rnn(x, self.hidden) # out size is (16,20,128) (16, 2560)

        with torch.no_grad():
            out = out.contiguous().view(self.batch_size, -1)
        # print(f'on line 67, out shape is {out.size()}')
        out = self.fc(out)

        return out, self.hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        return hidden


n_hidden = 128
model = RNN(1, CLASSES, n_hidden, 3).to(device)

# Define the loss
criterion = nn.CrossEntropyLoss()

# Optimizers require the parameters to optimize and a learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=0.003)
epochs = 50

for e in range(epochs):
    running_loss = 0
    for audios, labels in train_loader:
        optimizer.zero_grad()
        outputs, hidden = model(audios)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    else:
        print(f"Training loss: {loss / len(train_loader)}")

# Saving the model
save_path = './mlp.pth'
torch.save(model.state_dict(), save_path)

# Testing loop
correct, total = 0, 0
with torch.no_grad():
    # Iterate over the test data and generate predictions
    for audios, labels in test_loader:
        # Get inputs
        optimizer.zero_grad()
        outputs, hidden = model(audios)

        # Set total and correct
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Print accuracy
    print('Testing Accuracy: %d %%' % (100 * correct / total))


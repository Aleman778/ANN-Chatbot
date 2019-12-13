import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from cbgamma.datasets import AmazonReviews
from cbgamma.transforms import ToTfidf
    

def train():
    train_loss = 0
    for batch_nr, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        target = to_onehot(target)
        
        prediction = network(data)
        loss = loss_function(prediction, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
        print(
            '\rEpoch {} [{}/{}] - Loss: {}'.format(
                current_epoch+1, batch_nr+1, len(train_loader), loss
            ),
            end=''
        )
        train_loss /= len(train_loader)
        losses.append(train_loss)
        
    
def validate():
    validation_loss = 0
    for batch_nr, (data, target) in enumerate(validation_loader):
        data = data.to(device)
        target = target.to(device)
        target = to_onehot(target)
        
        prediction = network(data)
        loss = loss_function(prediction, target)
        loss.backward()
        validation_loss += loss.item()
        print(
            '\rEpoch {} [{}/{}] - Validation: {}'.format(
                current_epoch+1, batch_nr+1, len(validation_loader), loss
            ),
            end=''
        )
        validation_loss /= len(validation_loader)
        validation_losses.append(validation_loss)
        
    return validation_loss


def chat():
    inputs = ["this product is good",
              "this product is awful",
              "super good",
              "hate it but love it anyways"]
    for input in inputs:
        data = train_dataset.vectorizer([input])
        data = torch.from_numpy(np.array(data)).type(torch.FloatTensor)
        data = data.to(device)
        prediction = network(data)
        sentiment = "positive" if torch.argmax(prediction) == 1 else "negative"
        print("Text: {}\nPrediction: {}\nSentiment: {}\n".format(input, prediction, sentiment))


if __name__ == "__main__":
    vectorizer = ToTfidf()
    train_dataset = AmazonReviews('./', train=True, vectorizer=vectorizer, download=True, stopwords=True)
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=False)
    
    validation_dataset = AmazonReviews('./', train=False, vectorizer=vectorizer, download=True, stopwords=True)
    validation_loader = DataLoader(validation_dataset, batch_size=50, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("Training on:", device_name)
    
    network = nn.Sequential(
        nn.Linear(5199,2),
    )

    network.to(device)

    optimizer = optim.SGD(network.parameters(), lr=0.05)
    loss_function = nn.MSELoss()
    to_onehot = nn.Embedding(2, 2) 
    to_onehot.weight.data = torch.eye(2)
    to_onehot.to(device)
    epochs = 500
    current_epoch = 0
    losses = []
    validation_losses = []
    
    while current_epoch < epochs:
        train()
        loss = validate()
        current_epoch += 1

    print("\nDone training.")
    chat()

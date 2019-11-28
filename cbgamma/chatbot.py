
import torch
import torch.nn  as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data

class Model(nn.Module):
    """The Model is the representataion for the Artificial Neural Network.
    This model holds 

    This model is used by the Chatbot class to provide functionality
    such as training, validating etc.

    This model inherits torch.nn.Module which is a container
    for holding e.g. a network layer, these can be combined to 
    form a tree for layers
    """

    

    def __init__(self):
        """Initializes the neural network. Hard coded atm."""

        super(Model, self).__init__()
        self.linear = nn.Linear(7305, 2)
        self.loss = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.05)
        
    def forward(self, *input, **kwargs):
        F.relu(self.linear(*input))


        
class TrainingSession:
    """A Training session is data representation for holding
    the state of current training session e.g. epochs, losses etc.
    """
    def __init__(self, train_loader, val_loader):
        self.epoch = 0
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.losses = []
        self.val_losses = []

    def next_epoch(self):
        self.epoch += 1

    def print_status(self, batch_num, loss):
        print("\rEpoch {} [{}/{}] - Loss: {}".format(
                self.current_epoch+1, batch_num+1, self.batch_size, loss),
            end='')
        

class Chatbot:
    """The chatbot class has a neural network model object that
    can be trained, validated, tested and also used to chat with.
    """

    def __init__(self, model):
        self.model = model
        self.to_onehot = nn.Embedding(2, 2) 
        self.to_onehot.weight.data = torch.eye(2)
    

    def train(self, sess):
        """Runs the training for one epoch"""
        train_loss = 0
        for batch_num, (inputs, labels) in enumerate(sess.train_loader):
            labels = self.to_onehot(labels)
            prediction = self.model(inputs)
            loss = self.model.loss(prediction, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
            sess.print_status(batch_num)
            
        train_loss /= len(train_loader)
        losses.append(train_loss)
        sess.next_epoch()
        


    def validate(self, sess):
        validation_loss = 0
        for batch_num, (inputs, labels) in enumerate(sess.validation_loader):
            labels = to_onehot(labels)
            prediction = self.model(inputs)
            loss = self.model.loss(prediction, labels)
            validation_loss += loss.item()
            sess.print_status()
            
        validation_loss /= len(validation_loader)
        validation_losses.append(validation_loss)
        return validation_loss


import torch
import torch.nn  as nn
import torch.optim as optim
import torch.nn.functional as F


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
        self.optimizer = optim.SGD(self.parameters)
        
    def forward():
        F.relu(self.linear)
        

class Dataset:

    def __init__(inputs, labels, batch_size):
        """Initializes a dataset containing inputs and labels
        with batching of specific size"""
        
        self.inputs = inputs
        self.labels = labels
        self.batch_size = batch_size
        

        
class TrainingSession:

    def __init__(dataset):
        self.epoch = 0
        self.dataset = 0
        self.losses = []


    def train(num_epochs):
        self.max_epochs = self.epoch + num_epochs
        while self.epoch < self.max_epochs:
            
        self.curr_epoch++
        

    def print_status(batch_num, loss):
        print("\rEpoch {} [{}/{}] - Loss: {}".format(
                self.current_epoch+1, batch_num+1, self.batch_size, loss),
            end='')
        
    

class Chatbot:
    """The chatbot class has a neural network model object that
    can be trained, validated, tested and also used to chat with.
    """

    def __init__(self, model):
        self.model = model
    

    def train(self, sess):
        """Runs the training for one epoch"""
        for 
        train_loss = 0
        for batch_nr, (images, labels) in enumerate(train_loader):
            labels = to_onehot(labels)
            images = images.view(-1,784)
            prediction = network(images)
            loss = loss_function(prediction, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
            
            
        train_loss /= len(train_loader)
        losses.append(train_loss)
        


    def validate(epoch):
        validation_loss = 0
        for batch_nr, (images, labels) in enumerate(validation_loader):
            labels = to_onehot(labels)
            images = images.view(-1,784)
            prediction = network(images)
            loss = loss_function(prediction, labels)
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

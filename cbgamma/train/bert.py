import matplotlib.pyplot as plt
import transformers.optimization as optim
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torchvision.transforms import ToTensor
from torchvision.datasets.vision import StandardTransform

# Include logging information.
import logging
logging.basicConfig(level=logging.INFO)


class BertTransform:
    """Tokenizes the input sequence in the dataset using the BertTokenizer.
    The BERT model takes in tensor of indexed tokens."""
    
    def __init__(self, max_len, pretrained_weights):
        """Creates a BERT transformer that tokenizes the input text for
        use in the BERT model. This requires pretrained weights."""
        # The pretrained BERT tokenizer.
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        self.max_len = max_len;
        

    def __call__(self, dataset):
        tokens = []
        for text in dataset:
            padded_text = text + " <pad>"*(self.max_len)
            
            # Splits the input text into separate tokens.
            tokenized_text = self.tokenizer.tokenize(padded_text)

            # Prepends a classifier token and appends a separator token.
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text[0:self.max_len])

            # Converts the textual tokens into indices.
            indexed_tokens = self.tokenizer.build_inputs_with_special_tokens(indexed_tokens)

            # Appends the indexed tokens to the output tokens for entire dataset
            tokens.append(indexed_tokens)
        return torch.tensor(tokens)

    
class BertSentiment:
    """Bert Sentment class is used to train the BERT encoder model
    with an appended classification layer to perform a sentiment analysis."""

    def __init__(self, dataset, batch_size=32):
        """Creates a new model for sentiment analysis using BERT."""
        # The pretrained weights to use.
        pretrained_weights = 'bert-large-uncased'

        # Create trainsformer to convert text to indexed tokens.
        transformer = BertTransform(62, pretrained_weights)

        # Setup the train loader
        train_dataset = dataset('./', train=True, vectorizer=transformer, download=True)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        # Setup the validation loader
        val_dataset = dataset('./', train=False, vectorizer=transformer, download=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Retrive the CUDA device if available otherwise use CPU instead
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        print("Training on:", device_name)
        
        # Loads the pretrained BERT model with classifcation layer 
        self.model = BertForSequenceClassification.from_pretrained(pretrained_weights, num_labels=2)
        self.model.to(self.device)

        # Set the learning rate
        self.lr = 1e-5
        
        # Set the optimizer and scheduler
        training_steps = len(train_dataset) / batch_size
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, correct_bias=False)
        self.scheduler = optim.get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0.1, num_training_steps=training_steps)

        # Maximum gradient norm (used for gradient clipping)
        self.max_grad_norm = 1.0
        
        
    
    def train(self, epoch):
        train_loss = 0
        for batch_num, (data, targets) in enumerate(self.train_loader):
            data = data.to(self.device)
            targets = targets.to(self.device)
            loss, predicitons = self.model(input_ids=data, labels=targets)
            loss.backward();
            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            train_loss += loss.item() / len(self.train_loader)
            print(
                '\rEpoch {} [{}/{}] - Loss: {}'.format(
                    epoch+1, batch_num+1, len(train_loader), loss
                ),
                end=''
            )
        return train_loss

            
    def validate(self, epoch):
        val_loss = 0
        for batch_num, (data, target) in enumerate(self.val_loader):
            data = data.to(self.device);
            target = target.to(self.device);
            loss, predicitons = self.model(input_ids=data, lables=targets);
                        

    def run(self, max_epochs):
        train_losses = []
        val_losses = []
        for epoch in range(max_epochs):
            loss = self.train(epoch)
            train_losses.insert(loss);

        plt.plot(range(1,max_epochs+1), train_losses)
        plt.legend(["Training"])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

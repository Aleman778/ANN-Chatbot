import cbgamma.datasets as ds
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

import logging
logging.basicConfig(level=logging.INFO)

pretrained_weights = 'bert-large-uncased'


class BertTransform:
    """Tokenizes the input sequence in the dataset using the BertTokenizer.
    The BERT model takes in tensor of indexed tokens."""
    
    def __init__(self):
        # The pretrained BERT tokenizer.
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        

    def __call__(self, dataset):
        tokens = []
        for text in dataset:
            # Splits the input text into separate tokens.
            tokenized_text = self.tokenizer.tokenize(text)

            # Prepends a classifier token and appends a separator token.
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

            # Converts the textual tokens into indices.
            indexed_tokens = self.tokenizer.build_inputs_with_special_tokens(indexed_tokens)

            # Appends the indexed tokens to the output tokens for entire dataset
            tokens.append(torch.tensor(indexed_tokens));
        return tokens


class BertSentiment:
    """Bert Sentment class is used to train the BERT encoder model
    with an appended classification layer to perform a sentiment analysis."""

    def __init__(dataset):
        # Create trainsformer to convert text to indexed tokens.
        transformer = BertTransform();

        # Setup the train loader
        train_dataset = dataset('./', train=True, vectorizer=transformer, download=True)
        self.train_loader = DataLoader(train_dataset, batch_size=50, shuffle=False)

        # Setup the validation loader
        val_dataset = dataset('./', train=False, vectorizer=transformer, download=True)
        self.val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False)

        # Loads the pretrained BERT model with classifcation layer 
        self.model = BertForSequenceClassification.from_pretrained(pretrained_weights);

        # Retrive the CUDA device if available otherwise use CPU instead.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        print("Training on:", device_name)
        
    
    def train():
        
    

    def valid():
        
        
        

if __name__ == "__main__":
    bert = BertSentiment(ds.AmazonReviews);
    

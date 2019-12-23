import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer

__all__ = ["ToTfidf", "BertTransform", "DataToTensor"]


class ToTfidf(TfidfVectorizer):
    """Simple wrapper class around the Tfidf Vectorizer.
    This is used to turn the coprus into a vector format."""
    
    def __init__(self, analyzer='word', ngram_range=(1,2), max_features=50000, max_df=0.5, use_idf=True, norm='l2'):
        self.word_vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_features=max_features,
            max_df=max_df,
            use_idf=use_idf,
            norm=norm
        )
        self.fit = True

    def __call__(self, data):
        if self.fit:
            data = self.word_vectorizer.fit_transform(data)
        else:
            data = self.word_vectorizer.transform(data)
        self.fit = False
        return torch.from_numpy(np.array(data.todense())).type(torch.FloatTensor)



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
        return tokens

    
class DataToTensor:
    """Similar to ToTensor transformer but only
    transforms the data and leaves the target unchanged."""
    def __call__(self, data, target):
        return torch.tensor(data), target
    

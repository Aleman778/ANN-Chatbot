import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


__all__ = ["ToTfidf"]


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


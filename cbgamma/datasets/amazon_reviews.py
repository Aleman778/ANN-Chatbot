import os
import os.path
import torch
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_url, makedir_exist_ok
from sklearn.model_selection import train_test_split
from cbgamma.transforms import ToTfidf

class AmazonReviews(datasets.VisionDataset):
    """Dataset containing amazon reviews with positive or negative output."""
    
    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - negative', '1 - positive']

    resource = "https://drive.google.com/uc?export=download&id=1BvKLnZU3A8d-JSR3o1-KZeKSHaDqlepN"

    def __init__(self, root, train=True, transforms=None, vectorizer=None, download=False, stopwords=False):
        """Initializes a dataset containing inputs and labels
        with batching of specific size"""
        super(AmazonReviews, self).__init__(root, transforms=transforms)
        self.vectorizer = vectorizer
        self.stopwords = stopwords

        if download:
            self.download()
        
        if not self.check_exists():
            raise RuntimeError("Dataset not found. You can download it by setting `download=True`")

        data_file = self.training_file if train else self.test_file
        self.data, self.targets, self.vectorizer = torch.load(os.path.join(self.processed_folder, data_file))
        
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        data = self.data[index]
        target = self.targets[index]
        if self.transforms:
            return self.transforms(data, target)
        else:
            return data, target
        

    def preprocess_pandas(self, data, columns):
        """Preprocess the data removing any sensitive information e.g. emails, ip-addresses and numbers. Also makes the sentences lowercase so that """
        print('Processing...')
        data['Sentence'] = data['Sentence'].str.lower()
        data['Sentence'] = data['Sentence'].replace('[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)
        data['Sentence'] = data['Sentence'].replace('((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', regex=True)
        data['Sentence'] = data['Sentence'].replace('[^\w\s]','', regex=True)
        data['Sentence'] = data['Sentence'].replace('\d', '', regex=True)
        if self.stopwords:
            result = pd.DataFrame(columns=columns)
            for index, row in data.iterrows():
                word_tokens = word_tokenize(row['Sentence'])
                filtered_sent = [w for w in word_tokens if not w in stopwords.words('english')]
                result = result.append({
                    "index": row['index'],
                    "Class": row['Class'],
                    "Sentence": " ".join(filtered_sent[0:])
                }, ignore_index=True)
            return result;
        else:
            return data;

    def download(self):
        """Downloads the amazon reviews dataset from the internet and preprocess it."""
        if self.check_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        filename = 'amazon_cells_labelled.txt'
        download_url(self.resource, self.raw_folder, filename=filename)
        
        print('Processing...')
        
        train_data, test_data, train_targets, test_targets = self.read_csv(filename)

        # Converts array data into pytorch tensors
        train_tensor  = self.vectorizer(train_data)
        train_targets = torch.from_numpy(np.array(train_targets)).long()
        test_tensor   = self.vectorizer(test_data)
        test_targets  = torch.from_numpy(np.array(test_targets)).long()
        
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save((train_tensor, train_targets, self.vectorizer), f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save((test_tensor, test_targets, self.vectorizer), f)
            
        print('Done!')

    def read_csv(self, filename):
        data = pd.read_csv(os.path.join(self.raw_folder, filename), delimiter='\t', header=None);
        data.columns = ['Sentence', 'Class']
        data['index'] = data.index;
        columns = ['index', 'Class', 'Sentence']
        data = self.preprocess_pandas(data, columns)
        return train_test_split(
            data['Sentence'].values.astype('U'),
            data['Class'].values.astype('int32'),
            test_size=0.10,
            random_state=0,
            shuffle=True
        )

    @property
    def raw_folder(self):
        """The processed folder where stored pt files are located."""
        return os.path.join(self.root, self.__class__.__name__, 'raw')

        
    @property
    def processed_folder(self):
        """The processed folder where stored pt files are located."""
        return os.path.join(self.root, self.__class__.__name__, 'processed')

                
    def check_exists(self):
        """Checks if the training.pt and test.pt files exists."""
        return (os.path.exists(os.path.join(self.processed_folder,self.training_file)) and
            os.path.exists(os.path.join(self.processed_folder, self.test_file)))


if __name__ == "__main__":
    vectorizer = ToTfidf()
    dataset = AmazonReviews('./', train=True, vectorizer=vectorizer, download=True)
    print(len(dataset))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False);
    for batch_num, (data, targets) in enumerate(train_loader):
        print("batch_num {}:\n\tdata:{}, (len={})\n\ttargets:{}, (len={})\n\n"
              .format(batch_num, data, len(data[0]), targets, len(targets)));

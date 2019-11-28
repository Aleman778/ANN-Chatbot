import torch.utils.data as data
import csv


class CSVDataset(data.Dataset):

    training_file = 'training.pt'
    test_file = 'test.pt'
    
    """Simple dataset created by parsing a comma separated
    values (CSV) dataset. Simple preprocessing is done to 
    remove unrelated data e.g. email addresses, stopwords etc."""
    def __init__(self, root, filename, train=True, delimiter=','):
        self.filename = filename
        self.train = train
        self.delimiter = delimiter

        check_exist()


    def preprocess()
        
        
    def read(self):
        with open(self.filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=self.delimiter)
            for row in csv_reader:
                

    def write(self):
        print("Not implemented")

    

        
    def check_exists():
        


if __name__ == "__main__":
    dataset = CSVDataset("amazon_cells_labelled.txt", train=True, delimiter='\t')
    dataset.read()
    

from dataset import CSVDataset


class AmazonReviews(CSVDataset):
    """Dataset containing amazon reviews with positive or negative output."""
    
    def __init__(self, train=False):
        """Initializes a dataset containing inputs and labels
        with batching of specific size"""
        super(self, "amazon_cells_labelled.txt", train, '\t').__init__()
        
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)
        
    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

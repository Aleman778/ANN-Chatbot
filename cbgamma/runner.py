import cbgamma.visualize.dynplot as plt
import cbgamma.train.tfidf as network
import numpy as np



###
### Main entry point for the application.
###


def main():
    """The main function is called automatically when this file is executed."""

    vectorizer = datasets.TfidfTransform()
    train_dataset = datasets.AmazonReviews('./', train=True, vectorizer=vectorizer, download=True)
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=False)
    
    validation_dataset = datasets.AmazonReviews('./', train=False, vectorizer=vectorizer, download=True)
    validation_loader = DataLoader(validation_dataset, batch_size=50, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("Training on", device_name)
    
    network = nn.Sequential(
        nn.Linear(5199,2),
    )

    network.to(device)

    optimizer = optim.SGD(network.parameters(), lr=0.05)
    loss_function = nn.MSELoss()
    to_onehot = nn.Embedding(2, 2) 
    to_onehot.weight.data = torch.eye(2)
    to_onehot.to(device)
    current_epoch = 0
    losses = []
    validation_losses = []

    max_epochs = 1500
    epochs = range(1,max_epochs + 1)
    losses = []

    plt.plot(epochs, losses)
    
    while current_epoch < max_epochs:
        plt.update()
        train()
        loss = validate()
        current_epoch += 1

    print("\nDone training.")
    

if __name__ == '__main__':
    main()

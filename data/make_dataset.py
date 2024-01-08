import torch
from tests import _PATH_DATA

def mnist(data_path):
    """Return train and test dataloaders for MNIST."""
    train_data, train_labels = [ ], [ ]
    for i in range(5):
        train_data.append(torch.load(f"{data_path}/raw/corruptmnist/train_images_{i}.pt"))
        train_labels.append(torch.load(f"{data_path}/raw/corruptmnist/train_target_{i}.pt"))

    train_data = torch.cat(train_data, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    test_data = torch.load(f"{data_path}/raw/corruptmnist/test_images.pt")
    test_labels = torch.load(f"{data_path}/raw/corruptmnist/test_target.pt")

    #print(train_data.shape)
    #print(train_labels.shape)
    #print(test_data.shape)
    #print(test_labels.shape)

    train_data = train_data.unsqueeze(1)
    test_data = test_data.unsqueeze(1)
    
    train = torch.utils.data.TensorDataset(train_data, train_labels)
    print(len(train[:][0]))

    return (
        torch.utils.data.TensorDataset(train_data, train_labels), 
        torch.utils.data.TensorDataset(test_data, test_labels)
    )
    
    
if __name__ == "__main__":
    mnist(data_path=_PATH_DATA)
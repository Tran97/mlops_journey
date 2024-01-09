from make_dataset import mnist
from tests import _PATH_DATA
import os.path
import pytest

@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data():
    dataset = mnist(_PATH_DATA)
    train_tensor, test_tensor = dataset
    train_data = train_tensor[:][0]
    test_data = test_tensor[:][0]
    N_train = len(train_data)
    N_test = len(test_data)
    assert len(dataset[0]) == N_train, "Dataset did not have the correct number of samples"
    assert len(dataset[1]) == N_test, "Dataset did not have the correct number of samples"
    assert all([data_point.shape == (1,28,28) for data_point in train_data]), "Datapoints does not have the correct shape"
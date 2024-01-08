from models.model import myawesomemodel
import torch
import pytest

@pytest.mark.parametrize("test_input,expected", [(torch.randn(1,1,28,28), (1,11)), (torch.randn(2,1,28,28), (2,10))])
def test_model(test_input, expected):
    output = myawesomemodel(test_input)
    assert output.shape == expected
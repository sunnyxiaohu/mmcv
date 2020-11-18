import torch

from mmcv.cnn.bricks import HSigmoid
from mmcv.utils import TORCH_VERSION


def test_hsigmoid():
    act = HSigmoid()
    input_shape = torch.Size([1, 3, 64, 64])
    input = torch.randn(input_shape)
    expected_output = torch.min(
        torch.max((input + 1) / 2, torch.zeros(input_shape)),
        torch.ones(input_shape))
    if TORCH_VERSION == 'parrots':
        input = input.cuda()
        act = act.cuda()
        output = act(input)
        expected_output = expected_output.cuda()
    # test output shape
    assert output.shape == expected_output.shape
    # test output value
    assert torch.equal(output, expected_output)

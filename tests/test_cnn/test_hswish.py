import torch
from torch.nn.functional import relu6

from mmcv.cnn.bricks import HSwish
from mmcv.utils import TORCH_VERSION


def test_hswish():
    # test inplace
    act = HSwish(inplace=True)
    assert act.act.inplace
    act = HSwish()
    assert not act.act.inplace

    input = torch.randn(1, 3, 64, 64)
    expected_output = input * relu6(input + 3) / 6
    if TORCH_VERSION == 'parrots':
        input = input.cuda()
        expected_output = expected_output.cuda()
    output = act(input)
    # test output shape
    assert output.shape == expected_output.shape
    # test output value
    assert torch.equal(output, expected_output)

from models.model import load_model

import torch

network = load_model('utils/defaults.yaml')
x = torch.randn(1, 3, 256, 256)
result, paramsDict, handDictList, otherInfo = network(x)

print()
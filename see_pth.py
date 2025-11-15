import argparse
import torch
from models.model import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, default='misc/model/config.yaml')
parser.add_argument("--model", type=str, default='output/model/exp/hpds_eaa_ppp_best.pth')
parser.add_argument("--search", type=str, default='encoder.resnet.layer1')
opt = parser.parse_args()

network = load_model('utils/defaults.yaml')

# 获取当前模型的状态字典
state = network.state_dict()

# 打印检查点的键
# print("检查点中的键:")
# for key in state.keys():
#     if opt.search in key:
#         print(key)

network = load_model(opt.cfg)
# 打印模型期望的键
print("\n模型期望的键:")
for key in network.state_dict().keys():
    if opt.search in key:
        print(key)
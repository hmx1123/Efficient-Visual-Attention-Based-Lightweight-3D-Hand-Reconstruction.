from models.model import load_model

import torch

network = load_model('misc/model/config.yaml')

# 获取当前模型的状态字典
model_state = network.state_dict()
# 加载保存的状态字典
saved_state = torch.load('output/model/exp/hpds_eaa_pp_best.pth')

# print("当前模型参数:")
# for key in model_state.keys():
#     print(f"  {key}: {model_state[key].shape}")

# print("\n保存的模型参数:")
# for key in saved_state.keys():
#     print(f"  {key}: {saved_state[key].shape}")

# 找出不匹配的键
missing_keys = [key for key in model_state.keys() if key not in saved_state]
unexpected_keys = [key for key in saved_state.keys() if key not in model_state]

print(f"\n缺失的键: {missing_keys}")
print(f"意外的键: {unexpected_keys}")
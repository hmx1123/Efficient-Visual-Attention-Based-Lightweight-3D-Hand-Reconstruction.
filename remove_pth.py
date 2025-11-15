import torch
from collections import OrderedDict

def remove_keys_from_state_dict(input_path, output_path, keywords_to_remove):
    """
    从状态字典中移除包含特定关键词的键
    
    参数:
        input_path: 输入模型文件路径
        output_path: 输出模型文件路径
        keywords_to_remove: 要移除的关键词列表
    """
    # 加载原始状态字典
    state_dict = torch.load(input_path, map_location='cpu')
    
    # 创建新的状态字典，过滤掉包含关键词的键
    new_state_dict = OrderedDict()
    removed_keys = []
    
    for key, value in state_dict.items():
        # 检查键是否包含任何要移除的关键词
        if key in keywords_to_remove:
            removed_keys.append(key)
            continue
        new_state_dict[key] = value
    
    # 保存过滤后的状态字典
    torch.save(new_state_dict, output_path)
    
    # 打印被移除的键
    print(f"移除了 {len(removed_keys)} 个键:")
    for key in removed_keys:
        print(f"  - {key}")
    
    return new_state_dict

# 使用示例
keywords = ['encoder.conv_for_anchors.weight', 'encoder.conv_for_anchors.bias']  # 要移除的关键词列表
remove_keys_from_state_dict('output/model/exp/1.pth', 'output/model/exp/1_new.pth', keywords)
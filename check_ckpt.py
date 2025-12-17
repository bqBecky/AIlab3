import torch

# 加载.ckpt文件中的参数
ckpt_path = "model.ckpt"  # 你的.ckpt文件路径
checkpoint = torch.load(ckpt_path, map_location="cpu")

# 打印参数的键名（反映模型的层结构）
print("模型参数键名列表：")
for key in checkpoint.keys():
    print(key)
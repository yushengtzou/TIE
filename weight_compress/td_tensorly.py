"""
TT-Decomposition of MobileNetV2 FC layer weights using Python Package Tensorly.
Requires: torch, torchvision, numpy, tensorly, tensorly-torch
Author: Yu-Sheng Tzou
Date: 2025.12.13
"""

import torch as tn
import tensorly as tl
from tensorly.decomposition import tensor_train
from tensorly import tt_to_tensor

# ---------------------------------資料前處理--------------------------------------------

tl.set_backend('pytorch')

# 讀取 FC layer 權重
# with open("./weights/mobilenet_classifier_1.bin", "rb") as f:
#     weights = f.read()
#     weights = tn.frombuffer(weights, dtype=tn.float32).clone().reshape([1280, 1000])  # [out, in]
#     print(weights.shape)
#     print(weights.dtype)
#     print(weights.device)

# 使用 Pytorch 讀取 FC layer 權重並存為 tensor object
weights = tn.from_file(
    './weights/mobilenet_classifier_1.bin',
    shared=False, 
    size=1280*1000, 
    dtype=tn.float32).clone().reshape(1280,1000)

print('weights.shape: ', weights.shape)
print('weights.dtype: ', weights.dtype)
print('weights.device: ', weights.device)
print('weights: ', weights)

# 將矩陣張量化：1280 -> (8, 20, 8)、1000 -> (10, 10, 10)
tensor_high_order = weights.reshape(8, 20, 8, 10, 10, 10)
# 定義 TT-Rank：6 維張量需要 7 個 rank 數值，頭尾必須為 1，這裡中間的 rank 決定了壓縮率與精度
rank_list = [1, 16, 32, 32, 32, 16, 1]

# ---------------------------------張量分解--------------------------------------------

print('Start Block-TT decomposition')
# Block-TT Decomposition to a tensorized matrix into TT forma
factors = tensor_train(tensor_high_order, rank=rank_list) # 分解
# 檢查壓縮後的 Core
for i, f in enumerate(factors):
    print(f"Core {i} shape: {f.shape}")

# ---------------------------------張量重建--------------------------------------------

reconstructed_tensor = tt_to_tensor(factors)
weights_rec = reconstructed_tensor.reshape(1280, 1000)

# ---------------------------------結果分析--------------------------------------------

# 計算重建誤差
reconstruction_error = tn.linalg.norm(weights - weights_rec) / tn.linalg.norm(weights)
print(f'Reconstruction error: {reconstruction_error:.4f}')

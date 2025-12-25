"""
TT-Decomposition of MobileNetV2 FC layer weights using Python Package Tensorly.
Requires: torch, torchvision, numpy, tensorly, tensorly-torch
Author: Yu-Sheng Tzou
Date: 2025.12.13
"""

# import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch as tn
import tltorch

# ---------------------------------資料前處理--------------------------------------------

# 讀取 FC layer 權重並存為 tensor object
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
print('---------------------------------')
print('[讀取 FC layer 權重]')
print(weights.shape)
print(weights.dtype)
print(weights.device)
print('---------------------------------')

# ---------------------------------張量分解--------------------------------------------

# Reference: https://tensorly.org/torch/dev/user_guide/factorized_tensors.html#tensorized-tensors
tensorized_row_shape = [8, 20, 8]
tensorized_column_shape = [10, 10, 10]
rank = [1, 25, 25, 1]

# To Tensorized Matrix into TTM Format
print('[張量分解]')
tensorized_weights = tltorch.TensorizedTensor.from_matrix(
    weights, 
    tensorized_row_shape, 
    tensorized_column_shape, 
    rank,
    factorization='blocktt'
)

print('Tensorized Weights Type:', type(tensorized_weights))
print('Tensorized Weights:', tensorized_weights)

# ---------------------------------結果分析：層權重數值相對誤差--------------------------------------------

# 重建並計算誤差
weights_rec = tensorized_weights.to_matrix()
reconstruction_error = tn.linalg.norm(weights - weights_rec) / tn.linalg.norm(weights)
print('---------------------------------')
print('[結果分析1：層權重數值相對誤差]')
print(f'Reconstruction error with rank {rank}: {reconstruction_error:.4f}')

# ---------------------------------結果分析：壓縮比計算--------------------------------------------

# 原始參數總數 (1280 * 1000)
params_orig = weights.numel() 

# TTM (BlockTT) 核心參數總數
# tensorized_weights.factors 包含了所有的 TT-cores
params_tt = sum(p.numel() for p in tensorized_weights.factors)

# 計算指標
compression_ratio = params_orig / params_tt  # 壓縮比 (幾倍)
storage_percentage = (params_tt / params_orig) * 100  # 佔原本的百分比

print('---------------------------------')
print('[結果分析2：壓縮比計算]')
print(f"原始參數數量: {params_orig:,}")
print(f"壓縮後參數數量: {params_tt:,}")
print(f"壓縮比 (Compression Ratio): {compression_ratio:.2f}x")
print(f"現在的參數只佔原本的: {storage_percentage:.2f}%")
print('---------------------------------')
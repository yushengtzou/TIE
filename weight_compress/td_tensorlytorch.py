"""
TT-Decomposition of MobileNetV2 FC layer weights using Python Package Tensorly.
Requires: torch, torchvision, numpy, tensorly, tensorly-torch
Author: Yu-Sheng Tzou
Date: 2025.12.13
"""


# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch as tn
import tltorch
import pandas as pd
import os
import glob

# ---------------------------------資料前處理--------------------------------------------

# 創建 output 目錄
os.makedirs('layers_weights_tensor_cores', exist_ok=True)


# 讀取 FC layer 權重並存為 tensor object
# with open("./weights/mobilenet_classifier_1.bin", "rb") as f:
#     weights = f.read()
#     weights = tn.frombuffer(weights, dtype=tn.float32).clone().reshape([1280, 1000])  # [out, in]
#     print(weights.shape)
#     print(weights.dtype)
#     print(weights.device)

# 讀取全部 weight 檔案並排序
weight_files = glob.glob('./weights/mobilenet_*.bin')

def sort_key(filepath):
    """Sort by: feature_0 to feature_18, then classifier. Within features: conv_0, conv_1, conv_2"""
    filename = os.path.basename(filepath).replace('mobilenet_', '').replace('.bin', '')
    
    if filename.startswith('classifier'):
        return (999, 0, 0)  # Classifier comes last
    
    # Parse feature_X_conv_Y_Z format
    parts = filename.split('_')
    feature_num = int(parts[1])  # feature number (0-18)
    
    # Find conv number - look for 'conv' followed by number
    conv_num = 0  # default for layers without conv
    for i, part in enumerate(parts):
        if part == 'conv' and i + 1 < len(parts):
            conv_num = int(parts[i + 1])  # Take the number after 'conv'
            break
    
    return (feature_num, conv_num, 0)

# Sort files logically
weight_files.sort(key=sort_key)

print(f"Found {len(weight_files)} weight files to process (sorted)")
print('Processing order:')

# 張量分解每一層的權重
for i, f in enumerate(weight_files):
    layer_name = os.path.basename(f).replace('.bin', '').replace('mobilenet_', '')
    print(f"  {i+1:2d}. {layer_name}")
    
    # Load layer info to get shape
    info_file = f.replace('.bin', '_info.txt')
    with open(info_file, 'r') as file_handler:
        lines = file_handler.readlines()
        shape_line = [line for line in lines if 'Shape:' in line][0]
        shape = eval(shape_line.split(':')[1].strip())  # [1000, 1280] etc.
    
    # 使用 Pytorch 讀取該層權重並存為 tensor object
    if (len(shape) == 4):
        total_elements = shape[0] * shape[1] * shape[2] * shape[3]
    elif (len(shape) == 2):
        total_elements = shape[0] * shape[1]
    weights = tn.from_file(
        f, 
        shared=False, 
        size=total_elements, 
        dtype=tn.float32).clone().reshape(shape)
    
    print(f'\n--- 處理 {layer_name} ---')
    print(f'Shape: {weights.shape}')
    print(f'dtype: {weights.dtype}')
    print(f'device: {weights.device}')
    print('---------------------------------')
 
    # Reference: https://tensorly.org/torch/dev/user_guide/factorized_tensors.html#tensorized-tensors
    # ---------------------------------張量分解--------------------------------------------
    
    # 確認是否為 Conv layer
    is_conv = len(shape) == 4  # Conv layers have 4D weights [out, in, h, w]
    
    if is_conv:
        # For Conv layers - you might need Tucker decomposition instead
        print(f"Skipping Conv layer {layer_name}")
        continue
    
    # For FC layers - apply TT decomposition
    if shape == [1000, 1280]:
        tensorized_row_shape = [10, 10, 10]  # 10*10*10 = 1000 
        tensorized_column_shape = [8, 20, 8]    # 8*20*8 = 1280
        rank = [1, 25, 25, 1]
    else:
        print(f"Skipping layer {layer_name} with shape {shape} (not configured)")
        continue

    # To Tensorized Matrix into TTM Format and Decompose It Using Block-TT
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
 
    # ---------------------------------儲存：層權重張量核--------------------------------------------

    # 確認用：印出每一個張量核
    # for i, factor in enumerate(tensorized_weights.factors):
    #     print(f'Core {i} shape: {factor.shape}')
    #     print(factor)
  
    # 以 .pt 格式儲存張量核
    core_filename = f"layers_weights_tensor_cores/{layer_name}_cores.pt"
    tn.save(tensorized_weights.factors, core_filename)

    # ---------------------------------結果分析：層權重數值相對誤差--------------------------------------------

    # 重建並計算誤差
    weights_rec = tensorized_weights.to_matrix()
    reconstruction_error = tn.linalg.norm(weights - weights_rec) / tn.linalg.norm(weights)
    print('---------------------------------')
    print('[結果分析1：層權重數值相對誤差]')
    print(f'Reconstruction error with rank {rank}: {reconstruction_error:.4f}')

    # ---------------------------------結果分析：壓縮比計算--------------------------------------------

    # 原始參數總數 (1000 * 1280)
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


print(f'\n ✓ 已處理 {len(weight_files)} layers')


# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

# MobileNetV2 Accuracy(%) Verification on CIFAR-10 of Tensor-Train Decomposition Algorithmic Inferences
# Reference: Gong, Y., Yin, M., Huang, L., Xiao, J., Sui, Y., Deng, C., & Yuan, B. (2023, June). ETTE: Efficient tensor-train-based computing engine for deep neural networks. In Proceedings of the 50th Annual International Symposium on Computer Architecture (pp. 1-13).





# Input Parameters Preparation for ETTE Algo. 1, Algo. 2 - C Model (C++) 
# Reference: Gong, Y., Yin, M., Huang, L., Xiao, J., Sui, Y., Deng, C., & Yuan, B. (2023, June). ETTE: Efficient tensor-train-based computing engine for deep neural networks. In Proceedings of the 50th Annual International Symposium on Computer Architecture (pp. 1-13).
# ---------------二次分解 (Secondary SVD)：對每個 Gn 進行 SVD，將其解耦為獨立的 A 與 B 序列---------------





# ---------------全域重排 (Inter-core Reordering)：將所有的消除核 A 移至運算鏈的最前端，所有的擴展核 B 移至最後端---------------






# ---------------尺寸排序 (Heuristic Ordering)：根據核心尺寸按降序排列 A 群組，以最大化 FLOPs 縮減效果---------------


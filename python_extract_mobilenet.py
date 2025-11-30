#!/usr/bin/env python3
"""
Extract MobileNet convolutional layer weights for TT-decomposition testing
Requires: pip install torch torchvision numpy
"""

import torch
import torchvision.models as models
import numpy as np
import os

def extract_conv_weights(model, layer_name):
    """Extract weights from a specific convolutional layer"""
    for name, module in model.named_modules():
        if name == layer_name and isinstance(module, torch.nn.Conv2d):
            weights = module.weight.data.cpu().numpy()
            return weights
    raise ValueError(f"Layer {layer_name} not found")

def save_tensor_for_cpp(weights, filename_prefix):
    """Save tensor in binary format for C++ and metadata"""
    shape = weights.shape
    
    # Save binary (float32)
    weights_flat = weights.flatten().astype(np.float32)
    with open(f'{filename_prefix}.bin', 'wb') as f:
        f.write(weights_flat.tobytes())
    
    # Save metadata
    with open(f'{filename_prefix}_info.txt', 'w') as f:
        f.write(f'Shape: {list(shape)}\n')
        f.write(f'Total elements: {weights.size}\n')
        f.write(f'Min: {weights.min():.6f}\n')
        f.write(f'Max: {weights.max():.6f}\n')
        f.write(f'Mean: {weights.mean():.6f}\n')
        f.write(f'Std: {weights.std():.6f}\n')
    
    print(f"Saved: {filename_prefix}.bin")
    print(f"  Shape: {shape}")
    print(f"  Size: {weights.size} parameters ({weights.nbytes} bytes)")
    return shape

def main():
    print("=" * 60)
    print("MobileNet Weight Extraction for TT-Decomposition")
    print("=" * 60)
    print()
    
    # Create output directory
    os.makedirs('test_data', exist_ok=True)
    
    # Load pretrained MobileNetV2
    print("Loading MobileNetV2 (pretrained on ImageNet)...")
    model = models.mobilenet_v2(pretrained=True)
    model.eval()
    print("✓ Model loaded\n")
    
    # Print available conv layers
    print("Available Convolutional Layers:")
    print("-" * 60)
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            weight_shape = module.weight.shape
            num_params = np.prod(weight_shape)
            conv_layers.append((name, weight_shape, num_params))
            print(f"  {name:40s} {str(weight_shape):20s} {num_params:>10,} params")
    print()
    
    # Select interesting layers for testing
    # Sorted from small to large to show when TT-decomposition works best
    test_cases = [
        # Tiny layers (< 1K params) - won't compress
        ("features.1.conv.0.0", "Depthwise 3×3 (32×1×3×3 = 288 params)"),
        ("features.0.0", "First conv (32×3×3×3 = 864 params)"),
        ("features.2.conv.1.0", "Depthwise 3×3 (96×1×3×3 = 864 params)"),
        
        # Small layers (1K-10K params) - marginal compression
        ("features.8.conv.1.0", "Depthwise 3×3 (384×1×3×3 = 3.5K params)"),
        
        # Medium layers (10K-100K params) - moderate compression
        ("features.12.conv.0.0", "Pointwise (576×96×1×1 = 55K params)"),
        
        # Large layers (100K-500K params) - good compression!
        ("features.17.conv.0.0", "Pointwise (960×160×1×1 = 153K params)"),
        ("features.17.conv.2", "Pointwise (320×960×1×1 = 307K params)"),
        ("features.18.0", "Final conv (1280×320×1×1 = 409K params) - LARGEST!"),
    ]
    
    print("Extracting Selected Layers:")
    print("-" * 60)
    
    for layer_name, description in test_cases:
        try:
            print(f"\n{description}")
            print(f"  Layer: {layer_name}")
            
            weights = extract_conv_weights(model, layer_name)
            
            # Save for C++
            filename = f"test_data/mobilenet_{layer_name.replace('.', '_')}"
            shape = save_tensor_for_cpp(weights, filename)
            
            # Estimate TT-decomposition compression potential
            # Assuming rank r for each dimension
            r_max = min(16, min(shape))  # Conservative rank
            
            original_size = np.prod(shape)
            if len(shape) == 4:
                # (out_ch, in_ch, h, w) -> TT format
                tt_size_estimate = (
                    shape[0] * r_max +           # Core 1
                    r_max * shape[1] * r_max +   # Core 2
                    r_max * shape[2] * r_max +   # Core 3
                    r_max * shape[3]             # Core 4
                )
                compression_ratio = original_size / tt_size_estimate
                print(f"  Estimated TT compression: {compression_ratio:.1f}× (with rank {r_max})")
            
        except ValueError as e:
            print(f"  ✗ Error: {e}")
    
    print()
    print("=" * 60)
    print("Extraction Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Run C++ comparison: ./build/cmodel/tt_matlab_compare")
    print("  2. Compare with MATLAB TT-Toolbox")
    print("  3. Measure actual compression ratios")

if __name__ == "__main__":
    main()


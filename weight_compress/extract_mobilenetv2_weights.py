"""
Extract MobileNetV2 weights for Tensor decomposition.
Saves both convolutional and fully connected layer weights.
Requires: pip install torch torchvision numpy
Author: Yu-Sheng Tzou
Date: 2025.12.13
"""

import torch
import torchvision.models as models
import numpy as np
import os

def extract_layer_weights(model, layer_name):
    """Extract weights from Conv2d or Linear layer"""
    for name, module in model.named_modules():
        if name == layer_name:
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                weights = module.weight.data.cpu().numpy()
                layer_type = 'Conv' if isinstance(module, torch.nn.Conv2d) else 'FC'
                return weights, layer_type
    raise ValueError(f"Layer {layer_name} not found or not Conv2d/Linear")

def save_tensor_for_matlab(weights, filename_prefix, layer_type):
    """
    PyTorch stores weights in row-major (C) order: [out, in, h, w]
    """
    shape = weights.shape    
   
    # Save binary (float32, row-major)
    with open(f'{filename_prefix}.bin', 'wb') as f:
        f.write(weights.tobytes())
    
    # Save metadata for Python script
    with open(f'{filename_prefix}_info.txt', 'w') as f:
        f.write(f'Layer Type: {layer_type}\n')
        f.write(f'Shape: {list(shape)}\n')
        f.write(f'Total Parameters: {weights.size}\n')
        f.write(f'Size (KB): {weights.nbytes / 1024:.2f}\n')
        f.write(f'Min: {weights.min():.6f}\n')
        f.write(f'Max: {weights.max():.6f}\n')
        f.write(f'Mean: {weights.mean():.6f}\n')
        f.write(f'Std: {weights.std():.6f}\n')
        f.write(f'Data Order: Row-major (PytorchTT-compatible)\n')
    
    print(f"  Saved: {filename_prefix}.bin")
    print(f"  Type: {layer_type}")
    print(f"  Shape: {shape}")
    print(f"  Size: {weights.size:,} parameters ({weights.nbytes / 1024:.2f} KB)")
    
    return shape

def main():
    print("=" * 70)
    print("MobileNetV2 Weight Extraction for PytorchTT Tensor Decomposition")
    print("=" * 70)
    print()
    
    # Create output directory
    os.makedirs('weights', exist_ok=True)
    
    # Load pretrained MobileNetV2
    print("Loading MobileNetV2 (pretrained on ImageNet)...")
    model = models.mobilenet_v2(pretrained=True)
    model.eval()
    print("✓ Model loaded")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Scan all layers
    print("Available Layers:")
    print("-" * 70)
    all_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            weight_shape = module.weight.shape
            num_params = np.prod(weight_shape)
            layer_type = 'Conv2d'
            all_layers.append((name, layer_type, weight_shape, num_params))
            print(f"  {name:45s} Conv2d  {str(weight_shape):25s} {num_params:>10,} params")
        elif isinstance(module, torch.nn.Linear):
            weight_shape = module.weight.shape
            num_params = np.prod(weight_shape)
            layer_type = 'Linear'
            all_layers.append((name, layer_type, weight_shape, num_params))
            print(f"  {name:45s} Linear  {str(weight_shape):25s} {num_params:>10,} params")
    
    print()
    
    print("Extracting ALL Layers:")
    print("=" * 70)
    
    extracted_count = 0
    conv_count = 0
    fc_count = 0
    
    for layer_name, layer_type, weight_shape, num_params in all_layers:
        try:
            print(f"\n[{extracted_count + 1}/{len(all_layers)}] {layer_type} Layer")
            print(f"  Name: {layer_name}")
            
            weights, ltype = extract_layer_weights(model, layer_name)
            
            # Save for MATLAB
            filename = f"weights/mobilenet_{layer_name.replace('.', '_')}"
            shape = save_tensor_for_matlab(weights, filename, ltype)
            
            if ltype == 'Conv':
                conv_count += 1
            else:
                fc_count += 1
            
            extracted_count += 1
            
        except ValueError as e:
            print(f"  ✗ Skipped: {e}")
    
    print()
    print("=" * 70)
    print(f"Extraction Complete! ({extracted_count}/{len(all_layers)} layers)")
    print("=" * 70)
    print()
    print(f"  • Convolutional layers: {conv_count} (will use Tucker decomposition)")
    print(f"  • Fully connected layers: {fc_count} (will use TT decomposition)")
    print()

if __name__ == "__main__":
    main()


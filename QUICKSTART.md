# Quick Start: Compare C++ with MATLAB

## One-Command Comparison

```bash
chmod +x run_comparison.sh
./run_comparison.sh
```

This will:
1. ✅ Generate test data in MATLAB
2. ✅ Build C++ comparison program
3. ✅ Run C++ TT-decomposition
4. ✅ Compare results in MATLAB
5. ✅ Display summary

## Manual Steps

If automatic script fails, run manually:

### In MATLAB:
```matlab
cd /Users/yusheng/Dropbox/A01：工作/tt_decompose/tt_decompose_code
mkdir test_data
matlab_generate_test
```

### In Terminal:
```bash
cd /Users/yusheng/Dropbox/A01：工作/tt_decompose/tt_decompose_code
mkdir -p build && cd build
cmake ..
make tt_matlab_compare
./cmodel/tt_matlab_compare
cd ..
```

### Back in MATLAB:
```matlab
matlab_compare_results
```

## Expected Output

```
=== Test 1: Small 2×2×2 Tensor ===
  MATLAB ranks: [1 2 2 1]
  Core 1: Error = 1.234e-05
  Core 2: Error = 2.345e-05
  Status: ✓ Ranks match!

=== Test 2: Medium 3×4×5 Tensor ===
  ...

=== Test 3: Image 8×8×8×8×3 Tensor ===
  MATLAB storage: 5234 elements
  C++ storage: 5450 elements
  Storage ratio: 1.04
  Status: ✓ Both achieve similar compression!
```

## Files Generated

- `test_data/tensor_*.bin` - Input tensors
- `test_data/matlab_core_*.bin` - MATLAB TT-cores
- `test_data/cpp_core_*.bin` - C++ TT-cores
- `test_data/*.mat` - MATLAB workspace

## Troubleshooting

**Error: MATLAB not found**
```bash
# Add MATLAB to PATH
export PATH="/Applications/MATLAB_R2023b.app/bin:$PATH"
```

**Error: TT-Toolbox not found**
```matlab
% In MATLAB
addpath('/path/to/TT-Toolbox')
cd /path/to/TT-Toolbox
setup
```

**Error: Build failed**
```bash
# Check dependencies
cmake ..  # Should show SystemC, GTest found
```

See `MATLAB_COMPARISON_GUIDE.md` for details.


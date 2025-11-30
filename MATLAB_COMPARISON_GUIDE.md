# MATLAB vs C++ Comparison Guide

## Overview

This guide explains how to validate the C++ TT-decomposition implementation against the MATLAB TT-Toolbox reference.

## Prerequisites

1. **MATLAB** with TT-Toolbox installed
2. **C++ build environment** (see main README.md)
3. **TT-Toolbox** cloned in parent directory

## Directory Structure

```
tt_decompose/
├── TT-Toolbox/                    # MATLAB reference implementation
├── tt_decompose_code/             # C++ implementation
│   ├── test_data/                 # Generated test data (created automatically)
│   ├── matlab_generate_test.m     # Step 1: Generate test data in MATLAB
│   ├── cmodel/tt_matlab_compare   # Step 2: Run C++ comparison
│   └── matlab_compare_results.m   # Step 3: Analyze results in MATLAB
```

## Workflow

### Step 1: Generate Test Data (MATLAB)

In MATLAB, navigate to `tt_decompose_code/` and run:

```matlab
cd /Users/yusheng/Dropbox/A01：工作/tt_decompose/tt_decompose_code

% Create test_data directory if needed
mkdir test_data

% Generate test data and MATLAB reference results
matlab_generate_test
```

**Output:**
- `test_data/tensor_*.bin` - Input tensors (binary float32)
- `test_data/matlab_result_*.mat` - MATLAB TT-decomposition results
- `test_data/matlab_core_*_*.bin` - MATLAB TT-cores (binary float32)
- `test_data/test_metadata.txt` - Summary information

### Step 2: Run C++ Comparison

In terminal, build and run the comparison program:

```bash
cd tt_decompose_code
cd build  # or create: mkdir build && cd build

# Build
cmake ..
make tt_matlab_compare

# Run comparison
./cmodel/tt_matlab_compare
```

**Output:**
- Console output showing C++ decomposition results
- `test_data/cpp_core_*_*.bin` - C++ TT-cores for MATLAB comparison

### Step 3: Analyze Results (MATLAB)

Back in MATLAB:

```matlab
% Compare C++ results with MATLAB
matlab_compare_results
```

**Output:**
- Core-by-core error comparison
- Rank comparison
- Storage comparison
- Summary of differences

## Test Cases

### Test 1: Small 2×2×2 Tensor

**Purpose:** Basic correctness check
- **Input:** Structured 2×2×2 tensor
- **Expected:** Near-identical ranks and cores
- **Tolerance:** < 1e-4 relative error

### Test 2: Medium 3×4×5 Tensor

**Purpose:** Validate multi-dimensional decomposition
- **Input:** Structured 3×4×5 tensor
- **Expected:** Similar ranks (may differ by 1-2)
- **Tolerance:** < 1e-3 relative error

### Test 3: Image 8×8×8×8×3 Tensor

**Purpose:** Real-world data (image-like)
- **Input:** Random 8×8×8×8×3 tensor (reshaped 64×64×3 image)
- **Expected:** Similar compression ratios
- **Tolerance:** Both achieve < 1e-3 reconstruction error

## Understanding Differences

### Why Results May Differ

1. **Different SVD Algorithms:**
   - **MATLAB:** Exact SVD (QR + Jacobi iterations)
   - **C++:** Randomized SVD (fixed iterations, approximate)

2. **Non-Unique Decomposition:**
   - TT-decomposition is not unique
   - Different algorithms may find different (but equally valid) cores

3. **Numerical Precision:**
   - MATLAB uses LAPACK (highly optimized)
   - C++ uses custom implementation (portable)

### What Should Match

✅ **Compression ratio** (storage reduction)
✅ **Reconstruction error** (within same tolerance)
✅ **Rank bounds** (C++ ranks ≤ MATLAB ranks + oversampling)
✅ **Computational complexity** (both O(n×r²) operations)

❌ **Exact core values** (different decompositions)
❌ **Exact ranks** (randomized may select different ranks)

## Expected Output Example

```
=== Test 1: Small 2×2×2 Tensor ===
  MATLAB ranks: [1 2 2 1]
  C++ ranks:    [1 2 2 1]
  Core 1: Error = 1.234e-05
  Core 2: Error = 2.345e-05
  Core 3: Error = 1.123e-05
  Status: ✓ Ranks match!

=== Test 3: Image 8×8×8×8×3 Tensor ===
  MATLAB ranks: [1 8 18 15 8 1]
  MATLAB storage: 5234 elements
  MATLAB error: 9.876e-04
  
  C++ storage: 5450 elements
  Storage ratio (C++/MATLAB): 1.04
  Status: ✓ C++ decomposition completed
  Note: Randomized SVD may have different ranks than exact SVD
```

## Troubleshooting

### Error: "Cannot open file"

**Solution:** Run Step 1 (MATLAB) first to generate test data.

### Error: "TT-Toolbox not found"

**Solution:** 
```matlab
addpath('/path/to/TT-Toolbox');
cd /path/to/TT-Toolbox
setup
cd /path/to/tt_decompose_code
```

### Build Error: Missing chrono

**Solution:** Add to `tt_matlab_compare.cpp`:
```cpp
#include <chrono>
```

### Large Error in Comparison

**Possible causes:**
1. Different random seeds (expected)
2. Tolerance mismatch (check eps parameter)
3. Bug in C++ implementation (check against simple test)

## Performance Comparison

You can benchmark both implementations:

**MATLAB:**
```matlab
tic; tt_tensor(tensor_img, 1e-3); t_matlab = toc;
fprintf('MATLAB time: %.3f sec\n', t_matlab);
```

**C++:** Timing is printed automatically in output.

**Expected:** C++ should be slower initially (no BLAS/LAPACK optimization), but has predictable latency (hardware benefit).

## Next Steps

After validation:

1. **Optimize C++ implementation:**
   - Add BLAS/Eigen for matrix multiply
   - Implement fixed-point arithmetic
   - Profile and optimize hotspots

2. **Create SystemC model:**
   - Use validated C++ as reference
   - Add cycle-accurate timing
   - Prepare for hardware synthesis

3. **Benchmark on target data:**
   - Test on your specific application
   - Measure compression ratios
   - Tune rank selection

---

**Questions?** Check `RSVD_HARDWARE_GUIDE.md` for algorithm details.


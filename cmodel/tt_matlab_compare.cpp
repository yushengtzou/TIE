#include "tt_rsvd.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>

using namespace tt_rsvd;

// Load binary float data from MATLAB
std::vector<float> load_binary(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg() / sizeof(float);
    file.seekg(0, std::ios::beg);
    
    std::vector<float> data(size);
    file.read(reinterpret_cast<char*>(data.data()), size * sizeof(float));
    file.close();
    
    return data;
}

// Load MATLAB tensor and convert from column-major to row-major
std::vector<float> load_matlab_tensor(const std::string& filename, 
                                       const std::vector<int>& shape) {
    auto matlab_data = load_binary(filename);
    
    // MATLAB uses column-major (Fortran) order
    // C++ expects row-major (C) order
    // Need to transpose the indices
    
    std::vector<float> cpp_data(matlab_data.size());
    
    int d = shape.size();
    if (d == 3) {
        // For 3D tensor: MATLAB(i,j,k) -> C++[i,j,k]
        // MATLAB linear: i + j*n1 + k*n1*n2
        // C++ linear: i*n2*n3 + j*n3 + k
        int n1 = shape[0], n2 = shape[1], n3 = shape[2];
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                for (int k = 0; k < n3; k++) {
                    int matlab_idx = i + j*n1 + k*n1*n2;
                    int cpp_idx = i*n2*n3 + j*n3 + k;
                    cpp_data[cpp_idx] = matlab_data[matlab_idx];
                }
            }
        }
    } else if (d == 4) {
        // For 4D tensor: MATLAB(i1,i2,i3,i4) -> C++[i1,i2,i3,i4]
        // Typical for Conv weights: (out_ch, in_ch, kernel_h, kernel_w)
        int n1 = shape[0], n2 = shape[1], n3 = shape[2], n4 = shape[3];
        for (int i1 = 0; i1 < n1; i1++) {
            for (int i2 = 0; i2 < n2; i2++) {
                for (int i3 = 0; i3 < n3; i3++) {
                    for (int i4 = 0; i4 < n4; i4++) {
                        int matlab_idx = i1 + i2*n1 + i3*n1*n2 + i4*n1*n2*n3;
                        int cpp_idx = i1*n2*n3*n4 + i2*n3*n4 + i3*n4 + i4;
                        cpp_data[cpp_idx] = matlab_data[matlab_idx];
                    }
                }
            }
        }
    } else if (d == 5) {
        // For 5D tensor: MATLAB(i1,i2,i3,i4,i5) -> C++[i1,i2,i3,i4,i5]
        int n1 = shape[0], n2 = shape[1], n3 = shape[2], n4 = shape[3], n5 = shape[4];
        for (int i1 = 0; i1 < n1; i1++) {
            for (int i2 = 0; i2 < n2; i2++) {
                for (int i3 = 0; i3 < n3; i3++) {
                    for (int i4 = 0; i4 < n4; i4++) {
                        for (int i5 = 0; i5 < n5; i5++) {
                            int matlab_idx = i1 + i2*n1 + i3*n1*n2 + i4*n1*n2*n3 + i5*n1*n2*n3*n4;
                            int cpp_idx = i1*n2*n3*n4*n5 + i2*n3*n4*n5 + i3*n4*n5 + i4*n5 + i5;
                            cpp_data[cpp_idx] = matlab_data[matlab_idx];
                        }
                    }
                }
            }
        }
    } else {
        // For other dimensions, use generic transpose
        // TODO: implement for arbitrary dimensions
        cpp_data = matlab_data;
    }
    
    return cpp_data;
}

// Save binary float data for MATLAB
void save_binary(const std::string& filename, const std::vector<float>& data) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot create file: " + filename);
    }
    
    file.write(reinterpret_cast<const char*>(data.data()), 
               data.size() * sizeof(float));
    file.close();
}

// Save TT-core in MATLAB-compatible 3D format (r_prev, mode_size, r_next)
void save_core_for_matlab(const std::string& filename, const TTCore& core) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot create file: " + filename);
    }
    
    // C++ stores as Matrix(r_prev*mode_size, r_next)
    // MATLAB needs (r_prev, mode_size, r_next) in column-major order
    // MATLAB iterates: r_prev fastest, then mode_size, then r_next
    
    std::vector<float> matlab_data(core.r_prev * core.mode_size * core.r_next);
    
    for (int r_next = 0; r_next < core.r_next; r_next++) {
        for (int mode = 0; mode < core.mode_size; mode++) {
            for (int r_prev = 0; r_prev < core.r_prev; r_prev++) {
                // C++ index: (r_prev * mode_size + mode, r_next)
                int cpp_idx = r_prev * core.mode_size + mode;
                float value = core.core(cpp_idx, r_next);
                
                // MATLAB index: r_prev + mode*r_prev + r_next*r_prev*mode_size
                int matlab_idx = r_prev + mode * core.r_prev + r_next * core.r_prev * core.mode_size;
                matlab_data[matlab_idx] = value;
            }
        }
    }
    
    file.write(reinterpret_cast<const char*>(matlab_data.data()), 
               matlab_data.size() * sizeof(float));
    file.close();
}

// Compute relative error
double relative_error(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Vector size mismatch");
    }
    
    double diff_norm = 0.0;
    double a_norm = 0.0;
    
    for (size_t i = 0; i < a.size(); i++) {
        double diff = a[i] - b[i];
        diff_norm += diff * diff;
        a_norm += a[i] * a[i];
    }
    
    return std::sqrt(diff_norm / a_norm);
}

// Test Case 1: Small 2×2×2 tensor
void test_small_tensor() {
    std::cout << "=== Test 1: Small 2×2×2 Tensor ===\n";
    
    // Load MATLAB input (convert from column-major to row-major)
    std::vector<int> shape = {2, 2, 2};
    auto tensor = load_matlab_tensor("../test_data/tensor_small.bin", shape);
    std::vector<int> max_ranks = {8, 8};  // High ranks for exact decomposition
    
    std::cout << "  Loaded tensor: " << tensor.size() << " elements\n";
    std::cout << "  Shape: [2, 2, 2]\n";
    
    // C++ TT decomposition
    auto cores = tensor_train_decompose(tensor, shape, max_ranks);
    
    // Reconstruct and compute error
    auto reconstructed = reconstruct_from_cores(cores, shape);
    double rel_error = compute_relative_error(tensor, reconstructed);
    
    std::cout << "  C++ TT-cores: " << cores.size() << "\n";
    for (size_t k = 0; k < cores.size(); k++) {
        std::cout << "    Core " << k << ": " 
                  << cores[k].r_prev << " × " 
                  << cores[k].mode_size << " × " 
                  << cores[k].r_next << "\n";
    }
    std::cout << "  Reconstruction error: " << std::scientific << rel_error << std::fixed << "\n";
    
    // Save C++ cores for MATLAB comparison
    for (size_t k = 0; k < cores.size(); k++) {
        std::string filename = "../test_data/cpp_core_small_" + 
                               std::to_string(k+1) + ".bin";
        save_core_for_matlab(filename, cores[k]);
    }
    
    std::cout << "  C++ cores saved for MATLAB comparison\n\n";
}

// Test Case 2: 3×4×5 tensor
void test_medium_tensor() {
    std::cout << "=== Test 2: Medium 3×4×5 Tensor ===\n";
    
    // Load MATLAB input (convert from column-major to row-major)
    std::vector<int> shape = {3, 4, 5};
    auto tensor = load_matlab_tensor("../test_data/tensor_345.bin", shape);
    std::vector<int> max_ranks = {10, 10};
    
    std::cout << "  Loaded tensor: " << tensor.size() << " elements\n";
    std::cout << "  Shape: [3, 4, 5]\n";
    
    // C++ TT decomposition
    auto cores = tensor_train_decompose(tensor, shape, max_ranks);
    
    // Reconstruct and compute error
    auto reconstructed = reconstruct_from_cores(cores, shape);
    double rel_error = compute_relative_error(tensor, reconstructed);
    
    std::cout << "  C++ TT-cores: " << cores.size() << "\n";
    for (size_t k = 0; k < cores.size(); k++) {
        std::cout << "    Core " << k << ": " 
                  << cores[k].r_prev << " × " 
                  << cores[k].mode_size << " × " 
                  << cores[k].r_next << "\n";
    }
    std::cout << "  Reconstruction error: " << std::scientific << rel_error << std::fixed << "\n";
    
    // Save C++ cores
    for (size_t k = 0; k < cores.size(); k++) {
        std::string filename = "../test_data/cpp_core_345_" + 
                               std::to_string(k+1) + ".bin";
        save_core_for_matlab(filename, cores[k]);
    }
    
    std::cout << "  C++ cores saved for MATLAB comparison\n\n";
}

// Test Case 3: Image-like 8×8×8×8×3 tensor
void test_image_tensor() {
    std::cout << "=== Test 3: Image-like 8×8×8×8×3 Tensor ===\n";
    
    // Load MATLAB input (convert from column-major to row-major)
    std::vector<int> shape = {8, 8, 8, 8, 3};
    auto tensor = load_matlab_tensor("../test_data/tensor_img.bin", shape);
    std::vector<int> max_ranks = {20, 20, 20, 20};
    
    std::cout << "  Loaded tensor: " << tensor.size() << " elements\n";
    std::cout << "  Shape: [8, 8, 8, 8, 3]\n";
    
    // C++ TT decomposition
    auto start = std::chrono::high_resolution_clock::now();
    auto cores = tensor_train_decompose(tensor, shape, max_ranks);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Reconstruct and compute error
    auto reconstructed = reconstruct_from_cores(cores, shape);
    double rel_error = compute_relative_error(tensor, reconstructed);
    
    std::cout << "  C++ TT-cores: " << cores.size() << "\n";
    int total_storage = 0;
    for (size_t k = 0; k < cores.size(); k++) {
        int storage = cores[k].r_prev * cores[k].mode_size * cores[k].r_next;
        total_storage += storage;
        std::cout << "    Core " << k << ": " 
                  << cores[k].r_prev << " × " 
                  << cores[k].mode_size << " × " 
                  << cores[k].r_next 
                  << " (" << storage << " elements)\n";
    }
    
    std::cout << "  Total storage: " << total_storage 
              << " elements (original: " << tensor.size() << ")\n";
    std::cout << "  Compression ratio: " 
              << (float)tensor.size() / total_storage << "x\n";
    std::cout << "  Reconstruction error: " << std::scientific << rel_error << std::fixed << "\n";
    std::cout << "  Computation time: " << duration.count() << " ms\n";
    
    // Save C++ cores
    for (size_t k = 0; k < cores.size(); k++) {
        std::string filename = "../test_data/cpp_core_img_" + 
                               std::to_string(k+1) + ".bin";
        save_core_for_matlab(filename, cores[k]);
    }
    
    std::cout << "  C++ cores saved for MATLAB comparison\n\n";
}

// Generic function to test MobileNet layers
void test_mobilenet_layer(const std::string& layer_name, 
                           const std::string& description,
                           const std::vector<int>& shape,
                           int test_num) {
    std::cout << "=== Test " << test_num << ": " << description << " ===\n";
    
    std::string bin_file = "../test_data/mobilenet_" + layer_name + ".bin";
    
    // Check if file exists
    std::ifstream check(bin_file);
    if (!check.good()) {
        std::cout << "  ⚠ File not found: " << bin_file << "\n";
        std::cout << "  Skipping...\n\n";
        return;
    }
    check.close();
    
    auto tensor = load_matlab_tensor(bin_file, shape);
    std::vector<int> max_ranks(shape.size() - 1, 32);  // Rank 32 for all
    
    std::cout << "  Loaded tensor: " << tensor.size() << " elements\n";
    std::cout << "  Shape: [";
    for (size_t i = 0; i < shape.size(); i++) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    auto cores = tensor_train_decompose(tensor, shape, max_ranks);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Reconstruct and compute error
    auto reconstructed = reconstruct_from_cores(cores, shape);
    double rel_error = compute_relative_error(tensor, reconstructed);
    
    std::cout << "  C++ TT-cores: " << cores.size() << "\n";
    int total_storage = 0;
    for (size_t k = 0; k < cores.size(); k++) {
        int storage = cores[k].r_prev * cores[k].mode_size * cores[k].r_next;
        total_storage += storage;
        std::cout << "    Core " << k << ": " 
                  << cores[k].r_prev << " × " 
                  << cores[k].mode_size << " × " 
                  << cores[k].r_next 
                  << " (" << storage << " elements)\n";
    }
    
    std::cout << "  Original: " << tensor.size() << " params (" 
              << tensor.size() * 4.0 / 1024 << " KB)\n";
    std::cout << "  Compressed: " << total_storage << " params (" 
              << total_storage * 4.0 / 1024 << " KB)\n";
    
    // Color-coded compression result
    float compression = (float)tensor.size() / total_storage;
    if (compression < 1.0) {
        std::cout << "  Compression: " << compression << "× ⚠️  EXPANSION (BAD!)\n";
    } else if (compression < 1.5) {
        std::cout << "  Compression: " << compression << "× ⚠️  MARGINAL\n";
    } else if (compression < 3.0) {
        std::cout << "  Compression: " << compression << "× ✓ GOOD\n";
    } else {
        std::cout << "  Compression: " << compression << "× ✅ EXCELLENT\n";
    }
    
    std::cout << "  Reconstruction error: " << std::scientific << rel_error;
    if (rel_error < 1e-10) {
        std::cout << " (near-perfect)";
    } else if (rel_error < 1e-3) {
        std::cout << " (very good)";
    } else if (rel_error < 0.01) {
        std::cout << " (acceptable)";
    } else {
        std::cout << " ⚠️  HIGH ERROR!";
    }
    std::cout << std::fixed << "\n";
    
    std::cout << "  Computation time: " << duration.count() << " ms\n";
    
    // Save cores
    for (size_t k = 0; k < cores.size(); k++) {
        std::string filename = "../test_data/cpp_core_mobilenet_" + 
                               layer_name + "_" + 
                               std::to_string(k+1) + ".bin";
        save_core_for_matlab(filename, cores[k]);
    }
    
    std::cout << "  C++ cores saved\n\n";
}

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════╗\n";
    std::cout << "║  TT-Decompose: MATLAB vs C++ Comparison   ║\n";
    std::cout << "╚════════════════════════════════════════════╝\n";
    std::cout << "\n";
    
    try {
        // Basic validation tests
        test_small_tensor();
        test_medium_tensor();
        test_image_tensor();
        
        // MobileNet weight compression tests (sorted by size)
        std::cout << "╔════════════════════════════════════════════╗\n";
        std::cout << "║       MobileNet Weight Compression         ║\n";
        std::cout << "║     (Sorted Small → Large to Show Trend)   ║\n";
        std::cout << "╚════════════════════════════════════════════╝\n";
        std::cout << "\n";
        
        // Test all 8 layers in order (small to large)
        test_mobilenet_layer("features_1_conv_0_0", "Depthwise 32×1×3×3 (288 params)", {32, 1, 3, 3}, 4);
        test_mobilenet_layer("features_0_0", "First Conv 32×3×3×3 (864 params)", {32, 3, 3, 3}, 5);
        test_mobilenet_layer("features_2_conv_1_0", "Depthwise 96×1×3×3 (864 params)", {96, 1, 3, 3}, 6);
        test_mobilenet_layer("features_8_conv_1_0", "Depthwise 384×1×3×3 (3.5K params)", {384, 1, 3, 3}, 7);
        test_mobilenet_layer("features_12_conv_0_0", "Pointwise 576×96×1×1 (55K params)", {576, 96, 1, 1}, 8);
        test_mobilenet_layer("features_17_conv_0_0", "Pointwise 960×160×1×1 (153K params)", {960, 160, 1, 1}, 9);
        test_mobilenet_layer("features_17_conv_2", "Pointwise 320×960×1×1 (307K params)", {320, 960, 1, 1}, 10);
        test_mobilenet_layer("features_18_0", "Final Conv 1280×320×1×1 (409K params) - LARGEST!", {1280, 320, 1, 1}, 11);
        
        std::cout << "=== All tests completed ===\n";
        std::cout << "Run 'matlab_compare_results.m' in MATLAB to see comparison\n";
        std::cout << "Run 'matlab_mobilenet_test.m' for MobileNet-specific analysis\n\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}


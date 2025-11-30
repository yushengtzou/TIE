#include "tt_rsvd.h"
#include <iostream>
#include <iomanip>

using namespace tt_rsvd;

// Simple test: Can we decompose and reconstruct a 2×2×2 tensor?
int main() {
    std::cout << "=== Debug: TT-Decomposition Test ===\n\n";
    
    // Create simple 2×2×2 tensor
    std::vector<float> tensor = {
        1, 2,  // [0,:,:]
        3, 4,
        5, 6,  // [1,:,:]
        7, 8
    };
    
    std::cout << "Original tensor (2×2×2):\n";
    for (int i = 0; i < 2; i++) {
        std::cout << "[:,:," << i << "] =\n";
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                int idx = i * 4 + j * 2 + k;
                std::cout << std::setw(6) << tensor[idx] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    
    // Try standard TT-SVD (should work for this simple case)
    std::vector<int> shape = {2, 2, 2};
    std::vector<int> max_ranks = {8, 8};  // High ranks for exact
    
    auto cores = tensor_train_decompose(tensor, shape, max_ranks);
    
    std::cout << "TT-Cores generated:\n";
    for (size_t i = 0; i < cores.size(); i++) {
        std::cout << "  Core " << i << ": " 
                  << cores[i].r_prev << " × " 
                  << cores[i].mode_size << " × " 
                  << cores[i].r_next << "\n";
    }
    std::cout << "\n";
    
    // Print first core
    std::cout << "Core 0 values:\n";
    for (int i = 0; i < cores[0].r_prev; i++) {
        for (int j = 0; j < cores[0].mode_size; j++) {
            for (int r = 0; r < cores[0].r_next; r++) {
                int idx = i * cores[0].mode_size + j;
                std::cout << "  G0[" << i << "," << j << "," << r << "] = " 
                          << cores[0].core(idx, r) << "\n";
            }
        }
    }
    
    // Reconstruct
    std::cout << "\nReconstruction:\n";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                float sum = 0.0f;
                for (int r1 = 0; r1 < cores[0].r_next; r1++) {
                    for (int r2 = 0; r2 < cores[1].r_next; r2++) {
                        float g0 = cores[0].core(i, r1);
                        float g1 = cores[1].core(r1 * cores[1].mode_size + j, r2);
                        float g2 = cores[2].core(r2 * cores[2].mode_size + k, 0);
                        sum += g0 * g1 * g2;
                    }
                }
                std::cout << "  T[" << i << "," << j << "," << k << "] = " 
                          << sum << " (expected: " << tensor[i*4 + j*2 + k] << ")\n";
            }
        }
    }
    
    return 0;
}


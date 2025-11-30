#include "tt_rsvd.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>

namespace tt_rsvd {

// Generate random Gaussian matrix (hardware: use LFSR instead)
Matrix random_matrix(int rows, int cols, unsigned seed) {
    Matrix result(rows, cols);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result(i, j) = dist(gen);
        }
    }
    return result;
}

// Matrix multiplication: C = A * B
Matrix matmul(const Matrix& A, const Matrix& B) {
    if (A.cols != B.rows) {
        throw std::runtime_error("Matrix dimension mismatch");
    }
    
    Matrix C(A.rows, B.cols);
    
    // Simple implementation (hardware: use systolic array)
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < B.cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < A.cols; k++) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
    return C;
}

// QR decomposition using Modified Gram-Schmidt
void qr_decompose(const Matrix& A, Matrix& Q, Matrix& R) {
    int m = A.rows;
    int n = A.cols;
    
    Q = Matrix(m, n);
    R = Matrix(n, n);
    
    // Copy A to Q
    Q.data = A.data;
    
    // Modified Gram-Schmidt (hardware-friendly, sequential)
    for (int j = 0; j < n; j++) {
        // Compute R(j,j) = ||Q(:,j)||
        float norm = 0.0f;
        for (int i = 0; i < m; i++) {
            norm += Q(i, j) * Q(i, j);
        }
        R(j, j) = std::sqrt(norm);
        
        // Normalize Q(:,j)
        if (R(j, j) > 1e-10f) {
            for (int i = 0; i < m; i++) {
                Q(i, j) /= R(j, j);
            }
        }
        
        // Orthogonalize remaining columns
        for (int k = j + 1; k < n; k++) {
            // R(j,k) = Q(:,j)^T * Q(:,k)
            R(j, k) = 0.0f;
            for (int i = 0; i < m; i++) {
                R(j, k) += Q(i, j) * Q(i, k);
            }
            
            // Q(:,k) = Q(:,k) - R(j,k) * Q(:,j)
            for (int i = 0; i < m; i++) {
                Q(i, k) -= R(j, k) * Q(i, j);
            }
        }
    }
}

// Small SVD using power iteration (for k×k matrices)
SVDResult power_iteration_svd(const Matrix& A, int num_iterations) {
    int k = std::min(A.rows, A.cols);
    SVDResult result(A.rows, A.cols, k);
    
    Matrix A_work = A;
    
    // Find each singular value/vector pair
    for (int i = 0; i < k; i++) {
        // Random initial vector
        Matrix v = random_matrix(A_work.cols, 1, 42 + i);
        
        // Power iteration
        for (int iter = 0; iter < num_iterations; iter++) {
            // u = A * v
            Matrix u = matmul(A_work, v);
            
            // Normalize u
            float u_norm = 0.0f;
            for (int j = 0; j < u.rows; j++) {
                u_norm += u(j, 0) * u(j, 0);
            }
            u_norm = std::sqrt(u_norm);
            
            if (u_norm < 1e-10f) break;
            
            for (int j = 0; j < u.rows; j++) {
                u(j, 0) /= u_norm;
            }
            
            // v = A^T * u (transpose multiply)
            Matrix v_new(A_work.cols, 1);
            for (int j = 0; j < A_work.cols; j++) {
                float sum = 0.0f;
                for (int m = 0; m < A_work.rows; m++) {
                    sum += A_work(m, j) * u(m, 0);
                }
                v_new(j, 0) = sum;
            }
            
            // Normalize v
            float v_norm = 0.0f;
            for (int j = 0; j < v_new.rows; j++) {
                v_norm += v_new(j, 0) * v_new(j, 0);
            }
            v_norm = std::sqrt(v_norm);
            
            if (v_norm < 1e-10f) break;
            
            for (int j = 0; j < v_new.rows; j++) {
                v_new(j, 0) /= v_norm;
            }
            
            v = v_new;
        }
        
        // Compute singular value
        Matrix u = matmul(A_work, v);
        float sigma = 0.0f;
        for (int j = 0; j < u.rows; j++) {
            sigma += u(j, 0) * u(j, 0);
        }
        sigma = std::sqrt(sigma);
        
        // Store results
        result.S[i] = sigma;
        for (int j = 0; j < A_work.rows; j++) {
            result.U(j, i) = (sigma > 1e-10f) ? u(j, 0) / sigma : 0.0f;
        }
        for (int j = 0; j < A_work.cols; j++) {
            result.V(j, i) = v(j, 0);
        }
        
        // Deflate: A = A - sigma * u * v^T
        for (int m = 0; m < A_work.rows; m++) {
            for (int n = 0; n < A_work.cols; n++) {
                A_work(m, n) -= sigma * result.U(m, i) * result.V(n, i);
            }
        }
    }
    
    return result;
}

// Randomized SVD: Find rank-k approximation of A
SVDResult randomized_svd(const Matrix& A, int rank, int oversampling) {
    int m = A.rows;
    int n = A.cols;
    int l = std::min(rank + oversampling, std::min(m, n));
    
    // Step 1: Generate random matrix Ω (n × l)
    Matrix Omega = random_matrix(n, l, 12345);
    
    // Step 2: Y = A × Ω (Random projection)
    Matrix Y = matmul(A, Omega);
    
    // Step 3: Q = QR(Y) (Orthogonalization)
    Matrix Q, R;
    qr_decompose(Y, Q, R);
    
    // Step 4: B = Q^T × A (Form small matrix)
    Matrix B(l, n);
    for (int i = 0; i < l; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < m; k++) {
                sum += Q(k, i) * A(k, j);
            }
            B(i, j) = sum;
        }
    }
    
    // Step 5: SVD of small matrix B
    SVDResult small_svd = power_iteration_svd(B, 10);
    
    // Step 6: U = Q × Ũ (Recover full U)
    SVDResult result(m, n, rank);
    Matrix U_full = matmul(Q, small_svd.U);
    
    // Keep only top 'rank' components
    for (int i = 0; i < rank; i++) {
        result.S[i] = small_svd.S[i];
        for (int j = 0; j < m; j++) {
            result.U(j, i) = U_full(j, i);
        }
        for (int j = 0; j < n; j++) {
            result.V(j, i) = small_svd.V(j, i);
        }
    }
    
    return result;
}

// Tensor-Train decomposition using Randomized SVD
std::vector<TTCore> tensor_train_decompose(
    const std::vector<float>& tensor,
    const std::vector<int>& shape,
    const std::vector<int>& max_ranks
) {
    std::vector<TTCore> cores;
    int d = shape.size();  // Number of dimensions
    
    if (d == 0) return cores;
    
    // Current working tensor
    std::vector<float> C = tensor;
    int r_prev = 1;
    
    for (int k = 0; k < d - 1; k++) {
        int n_k = shape[k];
        int m = r_prev * n_k;
        
        // Compute remaining size
        int remaining = 1;
        for (int i = k + 1; i < d; i++) {
            remaining *= shape[i];
        }
        
        // Reshape C into matrix (m × remaining)
        Matrix A(m, remaining);
        for (int i = 0; i < m * remaining; i++) {
            A.data[i] = C[i];
        }
        
        // Randomized SVD
        int rank = std::min(max_ranks[k], std::min(m, remaining));
        SVDResult svd = randomized_svd(A, rank, 5);
        
        // Form TT-core
        TTCore core(r_prev, n_k, rank);
        for (int i = 0; i < r_prev; i++) {
            for (int j = 0; j < n_k; j++) {
                for (int r = 0; r < rank; r++) {
                    core.core(i * n_k + j, r) = svd.U(i * n_k + j, r);
                }
            }
        }
        cores.push_back(core);
        
        // Update C = Σ × V^T for next iteration
        C.resize(rank * remaining);
        for (int i = 0; i < rank; i++) {
            for (int j = 0; j < remaining; j++) {
                C[i * remaining + j] = svd.S[i] * svd.V(j, i);
            }
        }
        
        r_prev = rank;
    }
    
    // Last core
    int n_d = shape[d - 1];
    TTCore last_core(r_prev, n_d, 1);
    for (int i = 0; i < r_prev * n_d; i++) {
        last_core.core(i, 0) = C[i];
    }
    cores.push_back(last_core);
    
    return cores;
}

// Reconstruct tensor from TT-cores
std::vector<float> reconstruct_from_cores(
    const std::vector<TTCore>& cores,
    const std::vector<int>& shape
) {
    if (cores.empty() || shape.empty()) {
        return std::vector<float>();
    }
    
    int d = cores.size();
    
    // Start with the first core
    // First core: (1, n0, r1) -> reshape to (n0, r1)
    int n0 = cores[0].mode_size;
    int r1 = cores[0].r_next;
    Matrix result(n0, r1);
    
    for (int i = 0; i < n0; i++) {
        for (int j = 0; j < r1; j++) {
            result(i, j) = cores[0].core(i, j);
        }
    }
    
    // Contract with remaining cores
    for (int k = 1; k < d; k++) {
        int n_k = cores[k].mode_size;
        int r_prev = cores[k].r_prev;
        int r_next = cores[k].r_next;
        
        // Current result shape: (n0*n1*...*n_{k-1}, r_prev)
        int left_size = result.rows;
        
        // Core k shape: (r_prev, n_k, r_next)
        // Contraction: result[i, r] * core[r, n, r'] -> new_result[i*n_k + n, r']
        Matrix new_result(left_size * n_k, r_next);
        
        for (int i = 0; i < left_size; i++) {
            for (int n = 0; n < n_k; n++) {
                for (int r_out = 0; r_out < r_next; r_out++) {
                    float sum = 0.0f;
                    for (int r_in = 0; r_in < r_prev; r_in++) {
                        // result(i, r_in) * core(r_in * n_k + n, r_out)
                        sum += result(i, r_in) * cores[k].core(r_in * n_k + n, r_out);
                    }
                    new_result(i * n_k + n, r_out) = sum;
                }
            }
        }
        
        result = new_result;
    }
    
    // The last core should have r_next = 1, so result is (total_size, 1)
    // Flatten to 1D vector
    std::vector<float> reconstructed(result.rows);
    for (int i = 0; i < result.rows; i++) {
        reconstructed[i] = result(i, 0);
    }
    
    return reconstructed;
}

// Compute relative reconstruction error
double compute_relative_error(
    const std::vector<float>& original,
    const std::vector<float>& reconstructed
) {
    if (original.size() != reconstructed.size()) {
        throw std::runtime_error("Size mismatch in error computation");
    }
    
    double diff_norm_sq = 0.0;
    double orig_norm_sq = 0.0;
    
    for (size_t i = 0; i < original.size(); i++) {
        double diff = original[i] - reconstructed[i];
        diff_norm_sq += diff * diff;
        orig_norm_sq += original[i] * original[i];
    }
    
    if (orig_norm_sq < 1e-20) {
        return 0.0;  // Original is zero
    }
    
    return std::sqrt(diff_norm_sq / orig_norm_sq);
}

} // namespace tt_rsvd


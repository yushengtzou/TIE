#pragma once
#include <vector>
#include <cmath>
#include <random>

// Matrix utilities
struct Matrix {
    std::vector<float> data;
    int rows, cols;
    
    Matrix() : rows(0), cols(0) {}
    Matrix(int r, int c) : data(r * c, 0.0f), rows(r), cols(c) {}
    
    float& operator()(int i, int j) { return data[i * cols + j]; }
    const float& operator()(int i, int j) const { return data[i * cols + j]; }
};

// TT-Core structure
struct TTCore {
    Matrix core;
    int r_prev, mode_size, r_next;
    
    TTCore(int r1, int n, int r2) 
        : core(r1 * n, r2), r_prev(r1), mode_size(n), r_next(r2) {}
};

// Randomized SVD result
struct SVDResult {
    Matrix U;  // Left singular vectors
    std::vector<float> S;  // Singular values
    Matrix V;  // Right singular vectors
    
    SVDResult(int m, int n, int k) 
        : U(m, k), S(k), V(n, k) {}
};

// Core algorithms
namespace tt_rsvd {

// Generate random Gaussian matrix
Matrix random_matrix(int rows, int cols, unsigned seed = 42);

// Matrix multiplication: C = A * B
Matrix matmul(const Matrix& A, const Matrix& B);

// QR decomposition using Gram-Schmidt
void qr_decompose(const Matrix& A, Matrix& Q, Matrix& R);

// Small SVD using power iteration (for k×k matrices)
SVDResult power_iteration_svd(const Matrix& A, int num_iterations = 10);

// Randomized SVD: Find rank-k approximation of A
SVDResult randomized_svd(const Matrix& A, int rank, int oversampling = 5);

// Tensor-Train decomposition using Randomized SVD
std::vector<TTCore> tensor_train_decompose(
    const std::vector<float>& tensor,
    const std::vector<int>& shape,
    const std::vector<int>& max_ranks
);

// Reconstruct tensor from TT-cores
std::vector<float> reconstruct_from_cores(
    const std::vector<TTCore>& cores,
    const std::vector<int>& shape
);

// Compute relative reconstruction error
double compute_relative_error(
    const std::vector<float>& original,
    const std::vector<float>& reconstructed
);

} // namespace tt_rsvd


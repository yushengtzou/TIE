#include "tt_rsvd.h"
#include <gtest/gtest.h>
#include <iostream>
#include <iomanip>

using namespace tt_rsvd;

// Test random matrix generation
TEST(TTRSVDTest, RandomMatrix) {
    Matrix A = random_matrix(3, 4, 42);
    EXPECT_EQ(A.rows, 3);
    EXPECT_EQ(A.cols, 4);
    
    // Check not all zeros
    bool has_nonzero = false;
    for (auto val : A.data) {
        if (std::abs(val) > 1e-6f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

// Test matrix multiplication
TEST(TTRSVDTest, MatMul) {
    Matrix A(2, 3);
    A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3;
    A(1, 0) = 4; A(1, 1) = 5; A(1, 2) = 6;
    
    Matrix B(3, 2);
    B(0, 0) = 7; B(0, 1) = 8;
    B(1, 0) = 9; B(1, 1) = 10;
    B(2, 0) = 11; B(2, 1) = 12;
    
    Matrix C = matmul(A, B);
    
    EXPECT_EQ(C.rows, 2);
    EXPECT_EQ(C.cols, 2);
    
    // Expected: [[58, 64], [139, 154]]
    EXPECT_NEAR(C(0, 0), 58.0f, 1e-4f);
    EXPECT_NEAR(C(0, 1), 64.0f, 1e-4f);
    EXPECT_NEAR(C(1, 0), 139.0f, 1e-4f);
    EXPECT_NEAR(C(1, 1), 154.0f, 1e-4f);
}

// Test QR decomposition
TEST(TTRSVDTest, QRDecompose) {
    Matrix A(3, 2);
    A(0, 0) = 1; A(0, 1) = 2;
    A(1, 0) = 3; A(1, 1) = 4;
    A(2, 0) = 5; A(2, 1) = 6;
    
    Matrix Q, R;
    qr_decompose(A, Q, R);
    
    // Check Q is orthogonal: Q^T * Q = I
    Matrix QtQ(2, 2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) {
                sum += Q(k, i) * Q(k, j);
            }
            QtQ(i, j) = sum;
        }
    }
    
    // Q^T * Q should be identity
    EXPECT_NEAR(QtQ(0, 0), 1.0f, 1e-4f);
    EXPECT_NEAR(QtQ(1, 1), 1.0f, 1e-4f);
    EXPECT_NEAR(QtQ(0, 1), 0.0f, 1e-4f);
    EXPECT_NEAR(QtQ(1, 0), 0.0f, 1e-4f);
    
    // Check A = Q * R
    Matrix QR = matmul(Q, R);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            EXPECT_NEAR(QR(i, j), A(i, j), 1e-3f);
        }
    }
}

// Test randomized SVD
TEST(TTRSVDTest, RandomizedSVD) {
    // Create a simple low-rank matrix
    Matrix A(4, 3);
    A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3;
    A(1, 0) = 2; A(1, 1) = 4; A(1, 2) = 6;
    A(2, 0) = 3; A(2, 1) = 6; A(2, 2) = 9;
    A(3, 0) = 4; A(3, 1) = 8; A(3, 2) = 12;
    
    SVDResult svd = randomized_svd(A, 2, 3);
    
    EXPECT_EQ(svd.U.rows, 4);
    EXPECT_EQ(svd.U.cols, 2);
    EXPECT_EQ(svd.V.rows, 3);
    EXPECT_EQ(svd.V.cols, 2);
    EXPECT_EQ(svd.S.size(), 2);
    
    // First singular value should be dominant (rank-1 matrix)
    EXPECT_GT(svd.S[0], 10.0f);
    if (svd.S.size() > 1) {
        EXPECT_LT(svd.S[1], 1.0f);  // Second should be small (low-rank)
    }
    
    std::cout << "Singular values: ";
    for (auto s : svd.S) {
        std::cout << s << " ";
    }
    std::cout << std::endl;
}

// Test tensor-train decomposition
TEST(TTRSVDTest, TensorTrainDecompose) {
    // Create a simple 2×2×2 tensor
    std::vector<float> tensor = {
        1, 2,  // [0,:,:]
        3, 4,
        5, 6,  // [1,:,:]
        7, 8
    };
    
    std::vector<int> shape = {2, 2, 2};
    std::vector<int> max_ranks = {2, 2};
    
    auto cores = tensor_train_decompose(tensor, shape, max_ranks);
    
    EXPECT_EQ(cores.size(), 3);
    
    // Check dimensions
    EXPECT_EQ(cores[0].r_prev, 1);
    EXPECT_EQ(cores[0].mode_size, 2);
    
    EXPECT_EQ(cores[2].r_next, 1);
    EXPECT_EQ(cores[2].mode_size, 2);
    
    std::cout << "TT decomposition successful!" << std::endl;
    std::cout << "Core 0: " << cores[0].r_prev << " × " << cores[0].mode_size 
              << " × " << cores[0].r_next << std::endl;
    std::cout << "Core 1: " << cores[1].r_prev << " × " << cores[1].mode_size 
              << " × " << cores[1].r_next << std::endl;
    std::cout << "Core 2: " << cores[2].r_prev << " × " << cores[2].mode_size 
              << " × " << cores[2].r_next << std::endl;
}

// Test larger tensor
TEST(TTRSVDTest, LargerTensor) {
    // Create a 3×4×5 tensor with some structure
    std::vector<int> shape = {3, 4, 5};
    std::vector<float> tensor(3 * 4 * 5);
    
    // Fill with structured data
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 5; k++) {
                tensor[i * 20 + j * 5 + k] = (i + 1) * (j + 1) * (k + 1);
            }
        }
    }
    
    std::vector<int> max_ranks = {3, 4};
    
    auto cores = tensor_train_decompose(tensor, shape, max_ranks);
    
    EXPECT_EQ(cores.size(), 3);
    
    std::cout << "Large tensor TT decomposition:" << std::endl;
    for (size_t i = 0; i < cores.size(); i++) {
        std::cout << "Core " << i << ": " << cores[i].r_prev << " × " 
                  << cores[i].mode_size << " × " << cores[i].r_next << std::endl;
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


#include <iostream>
#include "compact_inference_scheme.h"
#include <vector>

Matrix compact_inference_scheme(
    const Matrix& X, // 原始輸入數據 X
    const std::vector<Matrix>& G, // d 個張量核心矩陣 G1...Gd
    int d,                        // 維度個數
    const int m[], 
    const int n[], 
    const int r[]
) {





}

Matrix transform(const Matrix& V, int h, const int m[], const int n[], const int r[]) {




}

Matrix matMul(const Matrix& A, const Matrix& B) {
    // 矩陣乘法前提：A 的列數必須等於 B 的行數
    if (A.cols != B.rows) {
        throw std::invalid_argument("Dimension mismatch for matrix multiplication.");
    }

    Matrix C;
    C.constructor(A.rows, B.cols);

    // 初始化 result 為 0
    std::fill(C.data, C.data + (C.rows * C.cols), 0.0);

    for (int i = 0; i < A.rows; ++i) {
        for (int k = 0; k < A.cols; ++k) {
            for (int j = 0; j < B.cols; ++j) {
                // 標準公式：C[i][j] += A[i][k] * B[k][j]
                C.data[i * C.cols + j] += A.data[i * A.cols + k] * B.data[k * B.cols + j];
            }
        }
    }
    return C;
}

Matrix transpose(const Matrix& A) {
    Matrix result;
    result.constructor(A.rows, A.cols); 
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < A.cols; ++j) {
            // 原本的 (i, j) 放到新矩陣的 (j, i)
            result.data[j * A.rows + i] = A.data[i * A.cols + j];
        }
    }
    return result;
}

void reshape(std::vector<Matrix>& X, int new_rows, int new_cols) {
    
}



#include <iostream>
#include "compact_inference_scheme.h"

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

Matrix matMul(const Matrix& G_tilde, const Matrix& V_prime) {
    
}

void transpose(Matrix& V) {
    for (int i = 0; i < V.rows; ++i) {
        for (int j = i + 1; j < V.cols; ++j) {
            std::swap(V.data[i * V.cols + j], V.data[j * V.cols + i]);
        }
    }
}

void reshape(Matrix& V, int new_rows, int new_cols) {
    
}



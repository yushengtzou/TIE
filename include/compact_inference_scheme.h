#pragma once

struct Matrix {
    int rows;
    int cols;
    int* data; // 使用一維陣列模擬二維，計算位址較快
};

Matrix compact_inference_scheme(
    const Matrix& X, // 原始輸入數據 X
    const std::vector<Matrix>& G, // d 個張量核心矩陣 G1...Gd
    int d,                        // 維度個數
    const int m[], 
    const int n[], 
    const int r[]
);

Matrix transform(const Matrix& V, int h, const int m[], const int n[], const int r[]);

Matrix matMul(const Matrix& G_tilde, const Matrix& V_prime); // V_h = Gh * V'h+1

void transpose(Matrix& V);

void reshape(Matrix& V, int new_rows, int new_cols);


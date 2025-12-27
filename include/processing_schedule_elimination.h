#pragma once
#include <vector>

struct Matrix {
    int constructor(int r, int c) {
        data = new int[r * c];
        rows = r;
        cols = c;
    };
    int rows;
    int cols;
    int* data; // 使用一維陣列模擬二維，計算位址較快
};

Matrix processing_schedule_elimination(
    const Matrix& X, // 原始輸入數據 X
    const std::vector<Matrix>& G, // d 個張量核心矩陣 G1...Gd
    int d,                        // 維度個數
    const int m[], 
    const int n[], 
    const int r[]
);

Matrix transform(const Matrix& A, int h, const int m[], const int n[], const int r[]);

Matrix matMul(const Matrix& A, const Matrix& B); // V_h = Gh * V'h+1

Matrix transpose(Matrix& A);

Matrix reshape(std::vector<Matrix>& X, int new_rows, int new_cols);


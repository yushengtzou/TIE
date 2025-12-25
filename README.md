TT-Edge: A Hardware-Software Co-Design for AI Accelerator with Verification on FPGA (ULX3S ECP5 12F)
======


# 1. 專案概述 (Project Overview)
本專案旨在驗證 TT-Edge (Tensor-Train Decomposition on Edge AI) 論文中提出的 TT-MAC 融合單元架構。但因為目標硬體 Lattice ECP5-12F 資源極度有限（僅約 12 個 DSP 乘法器），本專案的驗證重點從大規模並行加速（Massive Parallelism）轉向極致的資源效率 (Resource Efficiency) 與頻寬節省 (Bandwidth Saving)。

| 驗證項目      | 說明 |
| ----------- | ----------- |
| TT-MAC 融合      | 驗證 TT 解碼邏輯與 MAC 運算的單一流水線融合 (Single-Pipeline Fusion)，最大化資料重用。|
| INT8 精度   | 驗證 TT-MAC 在 INT8 定點化下的數值精度，確保量化誤差在 $1\%$ 以下。   |
| 資源/頻寬效率   | 證明壓縮後的 $G_k$ 核心儲存所需的 BRAM 佔用，遠低於傳統權重 $W$ 的佔用。 |
| 實體驗證   | 將精簡版 ($1 \times 1$ 或 $2 \times 2$ PE) 的 TT-MAC 部署到 ULX3S 上運行。|


# 2. 三層次 C Model 驗證策略 (Three-Tier Verification Strategy)
本專案採用分層次的 C Model 進行驗證，以確保從數學正確性到硬體行為的完整性。

| 目錄      | 驗證層次 | 目的 |
| ----------- | ----------- | ----------- |
| cmodel      | 橋樑級 C Model (Quantization Bridge) | 負責執行 $\text{FP32} \to \text{INT8/INT4}$ 定點化與數值精度驗證。 |
| systemc      | 架構級 SystemC Model |  實現 TT-MAC 融合單元的時序和資料流邏輯。用於 HLS 綜合前，計算精確的時脈週期數 (Latency) 和 $\text{Speedup}$。 |
| verilog      | 硬體目標 RTL (Target RTL) | 由 SystemC 經 HLS 工具或手動轉換生成的 Verilog/SystemVerilog RTL，最終部署到 ECP5-12F。 |

# 3. Directory Structure

The repository contains the following directories:

- cmodel: The Quantization Bridge software model for INT8 accuracy validation.
- docker: The Dockerfile and script to build the Docker container for a consistent build environment (includes Verilator and SystemC).
- include: Common header files defining data types and interfaces for cmodel, systemc, and Verilator wrappers.
- systemc: The Architecture Model implementing the single-PE, highly pipelined TT-MAC unit and its testbench for timing simulation.
- verilog: The generated or hand-coded SystemVerilog/RTL for the TT-MAC unit, targeting the ECP5-12F synthesis flow.
- vtuber: The directory for the utilities of C++ verilog datatype, and adapter for Verilator.

# 4. Installation

## Prerequisites

Before you can use Docker to build and run the application, you need to have the following installed on your system:
Docker: [Installation instructions for Docker](https://docs.docker.com/get-docker/)

Or you can follow the steps in Dockerfile, and install the required packages.

## Build Instructions

To build the repository using CMake, follow these steps:

1. Open a terminal and navigate to the root directory of the repository.
2. Create a build directory (e.g., build/) for out-of-source builds:

```bash
mkdir build
cd build
```
3. Generate the build files using CMake:
```bash
cmake ..
```

4. Build the repository using the generated Makefile:
```
make
make test
```

# 5. Contributing

We welcome contributions to improve the program. If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch: git checkout -b feature-branch
3. Make your changes and commit them: git commit -m "Description of changes"
4. Push your changes to your forked repository: git push origin feature-branch
5. Create a pull request on GitHub.

# 6. License

This program is released under the [MIT license](LICENSE). Please review the license file for more information.

# 7. Credits

1. [yodalee](https://github.com/yodalee)
2. [johnjohnlin](https://github.com/johnjohnlin)

# References
Hyunseok Kwak, et al. (2025). TT-Edge: A Hardware-Software Co-Design for Energy-Efficient Tensor-Train Decomposition on Edge AI. arXiv:2511.13738.

[Oseledets, I. V. (2011). Tensor-train decomposition. SIAM Journal on Scientific Computing, 33(5), 2295-2317.](https://users.math.msu.edu/users/iwenmark/Teaching/CMSE890/TENSOR_oseledets2011.pdf).
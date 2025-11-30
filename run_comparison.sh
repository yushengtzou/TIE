#!/bin/bash
# Automated MATLAB vs C++ Comparison Workflow

set -e  # Exit on error

echo "╔════════════════════════════════════════════╗"
echo "║  TT-Decompose: MATLAB vs C++ Comparison   ║"
echo "╚════════════════════════════════════════════╝"
echo ""

# Check if MATLAB is available
if ! command -v matlab &> /dev/null; then
    echo "❌ Error: MATLAB not found!"
    echo "   Please install MATLAB or add it to PATH"
    exit 1
fi

# Create test_data directory
mkdir -p test_data

# Step 1: Generate test data in MATLAB
echo "Step 1: Generating test data in MATLAB..."
echo "--------------------------------------"
matlab -nodisplay -nosplash -r "try; matlab_generate_test; catch ME; disp(ME.message); end; quit;" 2>&1 | grep -v "^$"

if [ ! -f "test_data/tensor_small.bin" ]; then
    echo "❌ Error: MATLAB test generation failed!"
    exit 1
fi
echo "✅ Test data generated successfully"
echo ""

# Step 2: Build C++ comparison program
echo "Step 2: Building C++ comparison program..."
echo "--------------------------------------"
if [ ! -d "build" ]; then
    mkdir build
fi

cd build
cmake .. > /dev/null 2>&1
make tt_matlab_compare

if [ ! -f "cmodel/tt_matlab_compare" ]; then
    echo "❌ Error: C++ build failed!"
    cd ..
    exit 1
fi
echo "✅ C++ program built successfully"
echo ""

# Step 3: Run C++ comparison
echo "Step 3: Running C++ decomposition..."
echo "--------------------------------------"
./cmodel/tt_matlab_compare
cd ..

if [ ! -f "test_data/cpp_core_small_1.bin" ]; then
    echo "❌ Error: C++ execution failed!"
    exit 1
fi
echo "✅ C++ decomposition completed"
echo ""

# Step 4: Compare results in MATLAB
echo "Step 4: Comparing results in MATLAB..."
echo "--------------------------------------"
matlab -nodisplay -nosplash -r "try; matlab_compare_results; catch ME; disp(ME.message); end; quit;" 2>&1 | grep -v "^$"

echo ""
echo "╔════════════════════════════════════════════╗"
echo "║          Comparison Complete!              ║"
echo "╚════════════════════════════════════════════╝"
echo ""
echo "Results saved in test_data/"
echo ""


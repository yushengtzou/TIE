#!/bin/bash
# Compile LaTeX report

echo "Compiling LaTeX report..."

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "Error: pdflatex not found!"
    echo "Please install LaTeX: brew install --cask mactex"
    exit 1
fi

# Compile (run twice for references)
pdflatex report.tex
pdflatex report.tex

# Cleanup auxiliary files
rm -f report.aux report.log report.out

echo "✓ Report compiled successfully: report.pdf"


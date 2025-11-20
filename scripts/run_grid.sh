#!/bin/bash
# Run grid of alpha values and methods

# Default values
CONTENT=""
STYLE1=""
STYLE2=""
ALPHAS="0.0,0.25,0.5,0.75,1.0"
OUTPUT_DIR="../data/outputs/grid"
METHODS="pca-joint,gram-linear"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --content)
            CONTENT="$2"
            shift 2
            ;;
        --style1)
            STYLE1="$2"
            shift 2
            ;;
        --style2)
            STYLE2="$2"
            shift 2
            ;;
        --alphas)
            ALPHAS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --methods)
            METHODS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$CONTENT" ] || [ -z "$STYLE1" ] || [ -z "$STYLE2" ]; then
    echo "Usage: $0 --content <path> --style1 <path> --style2 <path> [--alphas <comma-separated>] [--output <dir>] [--methods <comma-separated>]"
    exit 1
fi

# Run Python script
cd "$(dirname "$0")/../src" || exit 1
python -c "
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments import run_alpha_grid

alphas = [float(x.strip()) for x in '$ALPHAS'.split(',')]
methods = [x.strip() for x in '$METHODS'.split(',')]

df, grid_paths = run_alpha_grid(
    '$CONTENT', '$STYLE1', '$STYLE2',
    alphas, methods, '$OUTPUT_DIR'
)

print(f'Grids saved to: $OUTPUT_DIR')
print(f'Metrics CSV: {os.path.join(\"$OUTPUT_DIR\", \"metrics_summary.csv\")}')
"


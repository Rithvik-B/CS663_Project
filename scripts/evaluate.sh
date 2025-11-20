#!/bin/bash
# Evaluate metrics across output directory

# Default values
GEN_DIR="../data/outputs"
CONTENT_DIR="../data/content_examples"
STYLES_DIR="../data/style_examples"
OUTPUT_CSV="../results/metrics_summary.csv"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gen_dir)
            GEN_DIR="$2"
            shift 2
            ;;
        --content_dir)
            CONTENT_DIR="$2"
            shift 2
            ;;
        --styles_dir)
            STYLES_DIR="$2"
            shift 2
            ;;
        --out_csv)
            OUTPUT_CSV="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run Python script
cd "$(dirname "$0")/../src" || exit 1
python -c "
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics import evaluate_directory

df = evaluate_directory(
    '$GEN_DIR',
    '$CONTENT_DIR',
    '$STYLES_DIR',
    '$OUTPUT_CSV'
)

print(f'Evaluation complete. Metrics saved to: $OUTPUT_CSV')
print(f'Processed {len(df)} images')
"


#!/bin/bash
# Run a single PCA mixed style transfer

# Default values
CONTENT=""
STYLE1=""
STYLE2=""
ALPHA=0.5
OUTPUT=""
ITERS=1000
METHOD="joint"

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
        --alpha)
            ALPHA="$2"
            shift 2
            ;;
        --out)
            OUTPUT="$2"
            shift 2
            ;;
        --iters)
            ITERS="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
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
    echo "Usage: $0 --content <path> --style1 <path> --style2 <path> [--alpha <float>] [--out <path>] [--iters <int>] [--method <method>]"
    exit 1
fi

# Set default output if not provided
if [ -z "$OUTPUT" ]; then
    CONTENT_NAME=$(basename "$CONTENT" | cut -d. -f1)
    STYLE1_NAME=$(basename "$STYLE1" | cut -d. -f1)
    STYLE2_NAME=$(basename "$STYLE2" | cut -d. -f1)
    OUTPUT="../data/outputs/${CONTENT_NAME}_${STYLE1_NAME}_${STYLE2_NAME}_alpha${ALPHA}.jpg"
fi

# Run Python script
cd "$(dirname "$0")/../src" || exit 1
python -c "
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pca_gatys import pca_gatys_style_transfer
from config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config['iterations'] = $ITERS
config['alpha'] = $ALPHA

result, metrics = pca_gatys_style_transfer(
    '$CONTENT', '$STYLE1', '$STYLE2',
    alpha=$ALPHA,
    mixing_method='$METHOD',
    output_path='$OUTPUT',
    config=config
)

print(f'Result saved to: $OUTPUT')
print(f'Final loss: {metrics[\"total_loss\"][-1]:.4f}')
"


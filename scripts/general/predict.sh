#!/bin/bash
# Usage: bash scripts/general/predict.sh <corpus_train> <model_size> <parse_type>

if [ "$#" -ne 3 ]; then
    echo "Sử dụng: $0 <corpus_train> <model_size> <parse_type>"
    echo "Ví dụ: bash scripts/general/predict.sh rstdt 7b bottom_up"
    exit 1
fi

corpus=$1        # rstdt hoặc gum
model_size=$2    # 7b
parse_type=$3    # bottom_up hoặc top_down

# Prefix của tác giả trên Hugging Face
HF_PREFIX="arumaeawa"

# Tự động mapping sang Model ID trên Hugging Face
SPAN_ID="${HF_PREFIX}/${corpus}-${model_size}-span"
TOP_DOWN_ID="${HF_PREFIX}/${corpus}-${model_size}-top_down"
NUC_ID="${HF_PREFIX}/${corpus}-${model_size}-nuc"
REL_ID="${HF_PREFIX}/${corpus}-${model_size}-rel_with_nuc"

export PYTHONPATH=$PYTHONPATH:$(pwd)

python src/predict.py \
    --base_model_name "meta-llama/Llama-2-${model_size}-hf" \
    --span_lora_params "$SPAN_ID" \
    --top_down_lora_params "$TOP_DOWN_ID" \
    --nuc_lora_params "$NUC_ID" \
    --rel_with_nuc_lora_params "$REL_ID" \
    --parse_type "$parse_type" \
    --rel_type "rel_with_nuc" \
    --corpus "$corpus" \
    --save_dir_name "hf_test_${corpus}_${parse_type}"
#!/bin/bash --
set -exuo pipefail

ORIGINAL_FILE="adv/attack_type:0_model_type:0_1.adv"

# Clean previous results
rm -rf \
    adv/raw_result.zip \
    attack_type:0_model_type:0_1 \
    adv/extracted_data.json \
    senstive/raw_result.zip

cp $ORIGINAL_FILE adv/raw_result.zip

unzip adv/raw_result.zip

python -c "import pickle; import json; print(json.dumps(pickle.load(open('attack_type:0_model_type:0_1/data.pkl', 'rb')), indent=2))" > adv/extracted_data.json

python measure_senstive.py --attack 0 --data 0

ORIGINAL_FILE_SENS="senstive/attack_type:0_model_type:0.sen"

cp $ORIGINAL_FILE_SENS senstive/raw_result.zip

unzip senstive/raw_result.zip

python -c "import pickle; print(pickle.load(open('attack_type:0_model_type:0/data.pkl', 'rb')))" > senstive/extracted_data.json

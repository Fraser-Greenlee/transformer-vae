# Train an auto-encoder to compress code strings.

export TRAIN_FILE=250k_rndm_zinc_drugs_clean.txt
# Use MODEL_PATH to load a previous run
export MODEL_PATH=

export MODEL_NAME=zinc_drugs
#export MODEL_PATH=python_state_changes
export T5_MODEL_NAME=t5-large

python transformer_vae.py \
    --project_name="T5-VAE" \
    --output_dir=$MODEL_NAME \
    --model_path=$MODEL_PATH \
    --t5_model_name=t5-large \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --per_device_train_batch_size 5 \
    --gradient_accumulation_steps 16 \
    --save_total_limit 1 \
    --save_steps 625 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --overwrite_output_dir \
    --ae_latent_size 1000 \
    --set_seq_size 60 \

# needs an effective batch size >200

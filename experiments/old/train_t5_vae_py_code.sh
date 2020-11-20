# Train an auto-encoder to compress code strings.

export TRAIN_FILE=python_assign.txt
# Use MODEL_PATH to load a previous run
export MODEL_PATH=

export MODEL_NAME=python_assignments
# export MODEL_PATH=python_assignments/checkpoint-3250
export T5_MODEL_NAME=t5-large

python transformer_vae.py \
    --project_name="T5-VAE" \
    --output_dir=$MODEL_NAME \
    --model_path=$MODEL_PATH \
    --t5_model_name=t5-large \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --per_device_train_batch_size 28 \
    --gradient_accumulation_steps 7 \
    --save_total_limit 1 \
    --save_steps 625 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --overwrite_output_dir \
    --ae_latent_size 1000 \
    --set_seq_size 20 \

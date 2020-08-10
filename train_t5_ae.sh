# Train an auto-encoder to compress code strings.

export TRAIN_FILE=simple_state_changes.txt
export MODEL_PATH=

export MODEL_NAME=python_state_changes
#export MODEL_PATH=state_ae__base_infoVAE
export T5_MODEL_NAME=t5-large

python t5_ae.py \
    --project_name=diff_interp \
    --output_dir=$MODEL_NAME \
    --model_path=$MODEL_PATH \
    --t5_model_name=t5-large \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --per_device_train_batch_size 10 \
    --gradient_accumulation_steps 40 \
    --save_total_limit 1 \
    --save_steps 625 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --overwrite_output_dir \
    --ae_latent_size 1000 \
    --target_seq state \
    --set_seq_size 60 \

# needs an effective batch size >200

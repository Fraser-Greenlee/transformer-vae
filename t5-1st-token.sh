# Originally ran on 16GiB GPU
WANDB_PROJECT="transformer-vae-tests" WANDB_WATCH=false python -c "from transformer_vae.train import main; main()" \
    --output_dir=output \
    --run_name="news t5 1st token" \
    --do_train \
    --do_eval \
    --dataset_name="Fraser/news-category-dataset" \
    --text_column=headline \
    --per_device_train_batch_size 10 \
    --gradient_accumulation_steps 3 \
    --n_previous_latent_codes 0 \
    --transformer_type t5 \
    --encoder_model 1st-token \
    --set_seq_size 40 \
    --logging_steps 10 \
    --overwrite_output_dir \
    --save_total_limit 3 \
    --save_steps 1000 \
    --mlm_probability 0 \

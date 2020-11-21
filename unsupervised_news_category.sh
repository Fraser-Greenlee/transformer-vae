# Originally ran on 16GiB GPU
WANDB_PROJECT="T5-VAE tests" WANDB_WATCH=false python -c "from transformer_vae.train import main; main()" \
    --output_dir=output \
    --run_name="news-category-dataset test" \
    --do_train \
    --dataset_name="Fraser/news-category-dataset" \
    --text_column=headline \
    --per_device_train_batch_size 10 \
    --gradient_accumulation_steps 3 \
    --n_previous_latent_codes 0 \
    --transformer_type funnel-t5 \
    --set_seq_size 45 \
    --encoded_seq_size 12 \
    --logging_steps 100 \
    --overwrite_output_dir \
    --save_total_limit 3 \
    --save_steps 1000 \
    --mlm_probability 0 \

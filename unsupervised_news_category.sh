# Originally ran on 16GiB GPU
WANDB_PROJECT="T5-VAE tests" WANDB_WATCH=false python -c "from t5_vae.train import main; main()" \
    --output_dir=output \
    --run_name="news-category-dataset test" \
    --do_train \
    --dataset_name="Fraser/news-category-dataset" \
    --text_column=headline \
    --per_device_train_batch_size 20 \
    --gradient_accumulation_steps 3 \
    --n_previous_latent_codes 3 \
    --set_seq_size 45 \
    --t5_model_name "t5-large" \

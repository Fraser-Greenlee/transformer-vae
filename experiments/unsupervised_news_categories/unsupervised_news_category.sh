# Originally ran on 16GiB GPU
WANDB_PROJECT="T5-VAE"; WANDB_WATCH=false; python -m t5_vae \
    --output_dir=unsupervised_news_category \
    --do_train \
    --dataset_name="Fraser/news-category-dataset" \
    --text_column=headline \
    --per_device_train_batch_size 20 \
    --gradient_accumulation_steps 3 \
    --n_previous_latent_codes 3 \
    --set_seq_size 45 \

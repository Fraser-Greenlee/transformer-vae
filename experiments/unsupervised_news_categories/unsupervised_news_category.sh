WANDB_PROJECT="T5-VAE"; WANDB_WATCH=false; python -m t5_vae \
    --output_dir=unsupervised_news_category \
    --do_train \
    --dataset_name="Fraser/news-category-dataset" \
    --text_column=headline \

# Originally ran on 16GiB GPU
WANDB_PROJECT="transformer-vae-tests" TOKENIZERS_PARALLELISM=false python -c "from transformer_vae.train import main; main()" \
    --output_dir=output \
    --run_name="news t5 funnel" \
    --do_train \
    --evaluation_strategy steps \
    --dataset_name="Fraser/news-category-dataset" \
    --text_column=headline \
    --per_device_train_batch_size 75 \
    --transformer_type funnel-t5 \
    --latent_size 100 \
    --set_seq_size 45 \
    --logging_steps 100 \
    --eval_steps 500 \
    --overwrite_output_dir \
    --save_total_limit 3 \
    --save_steps 1000 \

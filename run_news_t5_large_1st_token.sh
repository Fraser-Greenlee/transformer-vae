# Originally ran on 16GiB GPU
WANDB_PROJECT="transformer-vae-tests" TOKENIZERS_PARALLELISM=false python -c "from transformer_vae.train import main; main()" \
    --output_dir=output \
    --run_name="news t5 1st token" \
    --do_train \
    --evaluation_strategy steps \
    --dataset_name="Fraser/news-category-dataset" \
    --text_column=headline \
    --per_device_train_batch_size 10 \
    --transformer_type t5 \
    --transformer_name t5-large \
    --encoder_model 1st-token \
    --decoder_model single-token \
    --latent_size 100 \
    --logging_steps 100 \
    --eval_steps 500 \
    --overwrite_output_dir \
    --save_total_limit 3 \
    --save_steps 1000 \


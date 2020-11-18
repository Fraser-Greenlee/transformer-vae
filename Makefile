format:
	black -l 120 -t py37 .

test:
	WANDB_DISABLED=true python -m pytest -s -v ./tests/

install-dev:
	pip install -e .[test]

run-sample:
	WANDB_PROJECT="T5-VAE"; WANDB_WATCH=false; python -m t5_vae \
	--output_dir=poet \
	--do_train \
	--huggingface_dataset=poems \

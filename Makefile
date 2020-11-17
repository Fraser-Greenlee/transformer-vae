format:
	black -l 120 -t py37 .

test:
	WANDB_DISABLED=true python -m pytest -s -v ./tests/

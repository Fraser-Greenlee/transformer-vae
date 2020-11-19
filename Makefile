format:
	black -l 120 -t py37 .

setup-env:
	# virtualenv t5_vae_env -p python3

activate:
	# source t5_vae_env/bin/activate

test:
	WANDB_DISABLED=true python -m pytest -s -v ./tests/

install-dev:
	pip uninstall -y t5_vae
	pip install -e .[test]

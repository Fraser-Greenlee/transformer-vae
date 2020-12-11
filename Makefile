format:
	black -l 120 -t py37 .

setup-env:
	# virtualenv t5_vae_env -p python3

activate:
	# source t5_vae_env/bin/activate

test:
	python -m pytest -s -v ./tests/

test-one-case:
	# python -m pytest -s -v ./tests/test_train.py::TrainTests::test_train_txt

install-dev:
	pip uninstall -y transformer_vae
	pip install -e .[test]

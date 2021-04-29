format:
	black -l 120 -t py37 .

reset-venv:
	rm -fr venv/
	virtualenv venv

install:
	pip install -r requirements.txt

test:
	black --check -l 120 -t py37 .
	python -m pytest -s -v ./tests/

test-one-case:
	# python -m pytest -s -v ./tests/test_train.py::TrainTests::test_train_txt

install-dev:
	pip install -r requirements-dev.txt
	pip install -r requirements.txt

publish:
	python setup.py sdist bdist_wheel
	twine upload --repository pypi dist/*

baselines:
	!cd transformer-vae; python run_experiment.py batch_small grad_accumulation_small semantics funnel_t5 full_30_tkn 30Seq eval --run_name News Headlines Baseline
	!cd transformer-vae; python run_experiment.py batch_small grad_accumulation_small syntax funnel_t5 full_30_tkn 30Seq eval

custom_tokenizers:
	python transformer_vae/tokenizer_train.py --dataset Fraser/python-lines
	python transformer_vae/tokenizer_train.py --dataset Fraser/mnist-text
	python transformer_vae/tokenizer_train.py --dataset Fraser/mnist-text-small

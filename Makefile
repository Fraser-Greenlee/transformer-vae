format:
	black -l 120 -t py37 .

test:
	python -m pytest -s -v ./tests/

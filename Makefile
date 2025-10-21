all: install train test lint #clean

install:
	pip install --upgrade pip
	pip install -r requirements.txt

train:
	python src/train.py

test:
	pytest tests/ -v

lint:
	flake8 src/ --max-line-length=100 --ignore=E501,W503

clean:
	rm -rf mlruns/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} +
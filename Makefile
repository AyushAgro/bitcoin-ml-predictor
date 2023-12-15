.PHONY: help install test clean lint format setup run demo fetch train predict analyze plot

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy

test: ## Run tests
	pytest tests/ -v

lint: ## Run linting
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format: ## Format code with black
	black .

clean: ## Clean up cache files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name ".DS_Store" -delete
	rm -f *.csv *.png *.jpg *.jpeg

run: ## Run the main application
	python bitcoin_predictor.py

demo: ## Run complete demo
	python cli.py demo

fetch: ## Fetch Bitcoin data
	python cli.py fetch-data

train: ## Train the model
	python cli.py train

predict: ## Make predictions
	python cli.py predict

analyze: ## Analyze Bitcoin data
	python cli.py analyze

plot: ## Plot Bitcoin data
	python cli.py plot

build: ## Build the package
	python setup.py sdist bdist_wheel

install-package: ## Install the package in development mode
	pip install -e .

notebook: ## Convert notebook to Python
	jupyter nbconvert --to python notebook.ipynb

run-notebook: ## Run the original notebook
	jupyter notebook notebook.ipynb 
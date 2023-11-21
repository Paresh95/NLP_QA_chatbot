init:
	curl -sSL https://install.python-poetry.org | python3 -
	poetry env use 3.10.5
	poetry shell
	poetry install

get-files:
	python run src/get_model_artifacts.py

run_api:
	python run src/run_api.py

docker-build:
	 docker build -t my_qa_app:latest .

docker-run:
	docker run -p 8000:8000 my_qa_app:latest

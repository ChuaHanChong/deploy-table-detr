#############
# ENVIRONMENT
#############
CONDA = conda
ENV_NAME = table-detr
ENV_PATH = $(shell $(CONDA) info --base)/envs/$(ENV_NAME)
PYTHON = $(ENV_PATH)/bin/python

.PHONY: env
env: $(ENV_PATH)/touchfile
$(ENV_PATH)/touchfile: environment.yml
	test -d $(ENV_PATH) || $(CONDA) env create -f environment.yml -q
	$(CONDA) env update -f environment.yml --prune
	$(CONDA) install -n $(ENV_NAME) -c conda-forge pre-commit flake8 pep8-naming black pydocstyle mypy -y
	$(CONDA) install -n $(ENV_NAME) -c conda-forge jupyter_core ipywidgets -y
	$(ENV_PATH)/bin/pre-commit install
	touch $(ENV_PATH)/touchfile


#############
# DEVELOPMENT
#############
.PHONY: run-dev-api
run-dev-api: env
	$(ENV_PATH)/bin/uvicorn src.app.api:app --host 0.0.0.0 --port 8000 --reload --reload-dir src 

.PHONY: run-inference
run-inference: env
	$(PYTHON) main.py --image-path $(filepath)


#############
# PRODUCTION
#############
DOCKER_IMAGE_NAME = table-detr-serving

.PHONY: run-prod-api
run-prod-api: env
	$(ENV_PATH)/bin/gunicorn src.app.api:app -c src/app/gunicorn.py -k uvicorn.workers.UvicornWorker

.PHONY: docker-build
docker-build:
	docker build -t $(DOCKER_IMAGE_NAME):latest -f Dockerfile .

.PHONY: docker-run
docker-run:
	docker run -p 8000:8000 --name $(DOCKER_IMAGE_NAME) $(DOCKER_IMAGE_NAME):latest


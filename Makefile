.PHONY: clean-pyc clean-build docs clean
define BROWSER_PYSCRIPT
import os, webbrowser, sys
try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT
BROWSER := python -c "$$BROWSER_PYSCRIPT"

PYTHON_VERSION = 3.8
PROJECT_NAME = ideas_annotation

ifeq (, $(shell which conda))
	HAS_CONDA=False
else
	HAS_CONDA=True
endif

#################################################################################
# CODE related, Clean, format, lint, documentation 						   		#
#################################################################################

## Remove all build, test, coverage and Python artefacts
clean: clean-build clean-pyc clean-test
## Remove build artefacts
clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

## Remove Python file artefacts
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

## Remove test and coverage artefacts
clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/

## Format and style code with flake8 and black (should be run inside environment)
lint:
	isort --lbt 1 --ls ideas_annotation tests
	black ideas_annotation tests --include '.py' --exclude '.ipynb'
	black ideas_annotation tests --include '.ipynb' --ipynb
	flake8 ideas_annotation tests

## Run tests quickly with the default Python (should be run inside environment)
test:
	python setup.py test

## Generate Sphinx HTML documentation, including API docs (should be run inside environment)
docs:
	rm -f docs/ideas_annotation.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ ideas_annotation
	$(MAKE) -C docs clean
	$(MAKE) -C docs singlehtml
	$(BROWSER) docs/_build/html/index.html

## Live reload/preview Sphinx HTML documentation, including API docs
servedocs: docs
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs singlehtml' -R -D .

## Install all source code as package (should be run inside environment)
install-as-pkg: clean
	python setup.py install

#################################################################################
# DATA and ENVIRONMENT related 													#
#################################################################################

## Setup project: environment, git, etc
setup: setup-environment setup-git

## Install git pre-commit hooks
setup-git:
ifeq (True, $(HAS_CONDA))
	conda run --name $(PROJECT_NAME) pre-commit install
else
	source venv/bin/activate; pre-commit install
endif

## Create and set up python environment (conda or virtualenv) and install dependencies
setup-environment:
ifeq (True, $(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
	conda create -y --name $(PROJECT_NAME) python=$(PYTHON_VERSION) pip
	conda run --name $(PROJECT_NAME) pip install -r requirements.txt -r requirements-dev.txt
	@echo ">>> New conda env created. Activate with: conda activate $(PROJECT_NAME)"
else
	python -m pip install -q virtualenv
	@echo ">>> Installing virtualenv if not already installed."
	virtualenv venv --python=python3
	source venv/bin/activate; pip install -r requirements.txt -r requirements-dev.txt
	@echo ">>> New virtualenv created. Activate with: source venv/bin/activate"
endif

# ToDo :=
# - Release command to GitLab CI/CD

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')

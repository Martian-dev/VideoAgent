# montage - cut a finished video out of one montage.
#
#   make setup    build .venv and install (~30 seconds)
#   make models   download the whisper.cpp model (~0.5 GB)
#   make doctor   verify this machine can run it
#
# First time:  make setup && make models && make doctor

VENV := .venv
PY   := $(VENV)/bin/python
UV   := $(shell command -v uv 2>/dev/null)

.PHONY: help setup models doctor check-uv clean

help:
	@sed -n '2,8p' $(MAKEFILE_LIST) | sed 's/^# \?//'

check-uv:
ifndef UV
	$(error uv not found. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh)
endif

setup: check-uv
	@echo "==> creating $(VENV)"
	@uv venv --python 3.11 --allow-existing $(VENV)
	@echo "==> installing"
	@uv pip install --python $(PY) -r requirements.txt
	@test -f .env || cp .env.example .env
	@echo
	@echo "Done. Put META_AI_API_KEY in .env, then: make models && make doctor"

models:
	@test -x $(PY) || { echo "No $(VENV) yet - run 'make setup' first."; exit 1; }
	@$(PY) scripts/fetch_models.py

doctor:
	@test -x $(PY) || { echo "No $(VENV) yet - run 'make setup' first."; exit 1; }
	@$(PY) -m montage.cli doctor --api

clean:
	rm -rf $(VENV) .montage

# Batteries-included VideoAgent (lean: editing + understanding).
#
#   make setup    build .venv and install everything (PROFILE=cpu|cuda|auto)
#   make models   download model weights (~6 GB, resumable)
#   make doctor   verify the machine can actually run it
#   make run      start VideoAgent
#
# Full first-time install:  make setup && make models && make doctor

PY_VERSION := 3.11
VENV       := .venv
# Torch flavour: cpu, cuda, or auto (cuda when an NVIDIA GPU is visible).
# The deployment target is CPU-only, so `make setup PROFILE=cpu` is the norm.
PROFILE    ?= auto
TORCH_REQS := $(strip $(if $(filter cpu,$(PROFILE)),requirements-torch-cpu.txt,\
              $(if $(filter cuda,$(PROFILE)),requirements-torch-cuda.txt,\
              $(if $(shell command -v nvidia-smi 2>/dev/null),requirements-torch-cuda.txt,requirements-torch-cpu.txt))))
# The CUDA wheels are large (nvidia-cudnn alone is ~700 MB) and pypi.nvidia.com
# is often slow; the default 30s timeout gives up on them.
export UV_HTTP_TIMEOUT := 300
PY         := $(VENV)/bin/python
UV         := $(shell command -v uv 2>/dev/null)

.PHONY: help setup models doctor run check-uv clean-venv

help:
	@sed -n '2,9p' $(MAKEFILE_LIST) | sed 's/^# \?//'

check-uv:
ifndef UV
	$(error uv not found. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh)
endif

setup: check-uv
	@echo "==> creating $(VENV) on python $(PY_VERSION)"
	@uv venv --python $(PY_VERSION) --allow-existing $(VENV)
	@echo "==> installing torch from $(TORCH_REQS)"
	@uv pip install --python $(PY) -r $(TORCH_REQS)
	@echo "==> installing lean requirements"
	@uv pip install --python $(PY) -r requirements-lean.txt
	@echo "==> installing vendored packages"
	@uv pip install --python $(PY) --no-deps -e tools/ImageBind
	@uv pip install --python $(PY) --no-deps -e tools/audio-preprocess
	@echo "==> applying dependency compatibility patches"
	@$(PY) scripts/patch_deps.py
	@test -f .env || cp .env.example .env
	@echo
	@echo "Done. Put META_AI_API_KEY in .env, then: make models && make doctor"

models:
	@test -x $(PY) || { echo "No $(VENV) yet - run 'make setup' first."; exit 1; }
	@$(PY) scripts/fetch_models.py

doctor:
	@test -x $(PY) || { echo "No $(VENV) yet - run 'make setup' first."; exit 1; }
	@$(PY) scripts/doctor.py --api

run:
	@test -x $(PY) || { echo "No $(VENV) yet - run 'make setup' first."; exit 1; }
	@$(PY) main.py

clean-venv:
	rm -rf $(VENV)

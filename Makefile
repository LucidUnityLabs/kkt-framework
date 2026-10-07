PYTHON ?= python3.13
VENV_PYTHON ?= .venv/bin/python

.PHONY: venv lock install test numerics diagnostic all-inputs reproducibility
venv:
	$(PYTHON) -m venv .venv
lock:
	$(PYTHON) tools/lock_environment.py
install:
	$(VENV_PYTHON) -m pip install --no-index --find-links vendor/wheels --only-binary=:all: --require-hashes -r requirements.lock
	$(VENV_PYTHON) -m pip check
test:
	$(VENV_PYTHON) -m pytest -q
numerics:
	$(VENV_PYTHON) tools/build.py --suite numerics --out build/numerics
diagnostic:
	$(VENV_PYTHON) tools/build.py --suite diagnostic --out build/diagnostic
all-inputs:
	$(VENV_PYTHON) tools/build.py --suite all --out build/all
reproducibility:
	$(VENV_PYTHON) tools/build.py --suite numerics --out build/repro-a
	$(VENV_PYTHON) tools/build.py --suite numerics --out build/repro-b
	$(VENV_PYTHON) tools/compare_builds.py build/repro-a build/repro-b
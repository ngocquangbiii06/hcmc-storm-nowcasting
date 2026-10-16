# Linux/macOS/WSL/Git-Bash convenience targets. On plain Windows run the commands directly.
START ?= 2025-08-01
END   ?= 2025-08-31

.PHONY: install test lint segments himawari radar align build build-sat report pipeline dagster train docker

install:
	pip install -e ".[dev,orchestration,ml,analysis]"

test:
	pytest

lint:
	ruff check src tests

segments:
	stormcast segments

himawari:
	stormcast himawari-download --start $(START) --end $(END)
	stormcast himawari-process  --start $(START) --end $(END)

radar:
	stormcast radar-index
	stormcast radar-process --start $(START) --end $(END)

align:
	stormcast align --start $(START) --end $(END)

build-sat:
	stormcast build-sat

build:
	stormcast build

report:
	stormcast report

pipeline: himawari build-sat report

dagster:
	dagster dev -f src/stormcast/orchestration/dagster_defs.py

train:
	python -m stormcast.ml.train_nowcast --config configs/default.yaml

docker:
	docker build -t stormcast .

.PHONY: install test lint demo baseline clean

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

# Runs the whole example with no API key and no network.
demo:
	cd examples/support-bot && evalgate run -s evalgate.yaml --no-report

# Shows the gate catching a real regression.
demo-regression:
	cd examples/support-bot && \
	  sed -i.bak 's|prompts/v2.txt|prompts/v1.txt|' evalgate.yaml && \
	  (evalgate run -s evalgate.yaml --gate --no-report; echo "exit=$$?") ; \
	  mv evalgate.yaml.bak evalgate.yaml

baseline:
	cd examples/support-bot && evalgate baseline -s evalgate.yaml --force

clean:
	rm -rf .pytest_cache .ruff_cache build dist **/__pycache__ examples/*/.evalgate

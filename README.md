# Code Module

Lightweight vulnerability scanning and chunking utilities used in this project.

## Files

- `vuln_scanner.py`: scanner implementation.
- `chunking.ipynb`: notebook for chunking experiments.
- `server.py`: local service entrypoint.
- `client.py`: client script for testing the service.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python server.py
python client.py
```

## Notes

- Keep large generated artifacts out of git.

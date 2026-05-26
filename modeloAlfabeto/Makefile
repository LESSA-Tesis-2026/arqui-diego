UV ?= uv
PYTHON ?= 3.11

.PHONY: help setup sync shell check capture process train translate clean

help:
	@echo "Modelo Alfabeto LESSA commands"
	@echo ""
	@echo "  make setup      Create/sync the uv .venv with Python 3.11"
	@echo "  make sync       Alias for setup"
	@echo "  make shell      Open a new shell with .venv activated"
	@echo "  make check      Verify Python, MediaPipe, OpenCV, and TensorFlow"
	@echo "  make capture    Capture raw alphabet images"
	@echo "  make process    Convert captured images to H5 keypoints"
	@echo "  make train      Train the alphabet classifier"
	@echo "  make translate  Run real-time alphabet translation"
	@echo "  make clean      Remove Python caches"

setup:
	$(UV) sync --python $(PYTHON)

sync: setup

shell: setup
	@echo "Opening a new activated shell. Exit it with: exit"
	@. .venv/bin/activate && exec $$SHELL

check: setup
	$(UV) run python -c "import sys, cv2, mediapipe as mp, tensorflow as tf; print('python:', sys.version.split()[0]); print('opencv:', cv2.__version__); print('mediapipe:', mp.__version__, 'solutions:', hasattr(mp, 'solutions')); print('tensorflow:', tf.__version__)"

capture: setup
	$(UV) run python 1a_capturar_alfabeto.py

process: setup
	$(UV) run python 1b_procesar_alfabeto_h5.py

train: setup
	$(UV) run python 2_entrenamiento_alfabeto.py

translate: setup
	$(UV) run python 3_traduccion_alfabeto.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

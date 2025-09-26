
PY=python

.PHONY: setup label convert train infer lint test clean

setup:
	$(PY) -m pip install -r requirements.txt

label:
	$(PY) scripts/cluster_and_label.py --in data/raw --out labels/auto_labels.txt

convert:
	$(PY) scripts/convert_pcd_to_npy.py --in data/raw --out data/processed

train:
	PYTHONPATH=. $(PY) -m pointnet_baseline.train --data data/processed --labels labels/auto_labels.txt --epochs 10 --batch_size 8 --points 4096

infer:
	PYTHONPATH=. $(PY) -m pointnet_baseline.infer --pcd data/processed/example.npy --ckpt runs/best.pth

lint:
	$(PY) -m pip install ruff black
	ruff check . || true
	black --check . || true

test:
	$(PY) - <<'PY'
import importlib; import sys; sys.path.append('.'); importlib.import_module('pointnet_baseline.model'); print('import-ok')
PY

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache

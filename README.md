# Uncertainty-Aware Facial Expression Analysis

CPU-only pipeline for trustworthy facial-expression analysis at the edge: label-noise audit, soft-label training, calibration, conformal prediction, selective prediction, class-level quantification, ONNX/INT8 edge benchmarking and a demo pipeline.

By Utkarsh Gupta.

## Setup
```
python -m venv .venv
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
python -m pytest -q
```

## Layout
- `src/m3/` - library code (data, audit, train, calib, conformal, uncertainty, quantification, edge, explain, demo)
- `scripts/` - run / evaluation drivers
- `configs/` - YAML configs
- `tests/` - unit tests
- `models_ext/` - face detector ONNX model

# Checkpoints

Trained PPM checkpoints are written here.

Example:

```bash
.venv/bin/python tools/train_ppm.py \
  --data data/trajectories.jsonl \
  --output checkpoints/ppm.pt
```

The Streamlit and FastAPI demos accept checkpoint paths such as
`checkpoints/ppm.pt`. If no checkpoint is supplied, the improved system can use
the deterministic verifier scorer from `src/core/scoring.py`.

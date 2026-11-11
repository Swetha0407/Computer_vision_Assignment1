# ITA0520 – Computer Vision | Assignment 1 – CO1 – Set B
## Smart Greenhouse Crop Imaging Station

This is a reproducible Python/OpenCV simulation project for all 10 tasks in Assignment 1.
It creates 30 synthetic greenhouse tray images with controlled variation in illumination,
shadows, glare, leaf size, and missing seedlings, then runs the three alternatives required
for each task and exports measurements, comparison figures, and sample annotations.

### Important scope and honesty note
The input images are **synthetic simulations**, not photographs from a real greenhouse and
not expert-labelled field data. They are suitable for demonstrating reproducible algorithm
comparisons and the required calculations, but they do not establish real-world crop detection
accuracy. Images are generated at 640x480 for practical CPU execution. Task 6 separately
evaluates 640, 1280, and 1920 pixels across the 640 mm tray field. The assignment's 1280x960
camera and 15 fps are treated as system design targets, not claimed as measured hardware results.

### Requirements
- Python 3.9+
- numpy
- opencv-python
- pandas
- matplotlib

Install:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Run:
```bash
python src/run_all.py
```
The script regenerates the synthetic dataset, runs all ten tasks, writes CSV results and
figures, and creates `results/summary.json`.

### Project structure
- `src/run_all.py` – complete reproducible pipeline
- `data/synthetic/` – generated sample images and metadata
- `outputs/` – representative annotated output panels
- `results/` – per-task CSVs, summary JSON and figures
- `report/` – report is supplied separately in the ZIP

### Reproducibility
The random seed is fixed (`2026`). Runtime values are measured on the machine running the
script and will vary with CPU, OpenCV build, image size, and system load. No experimental
scores are fabricated.

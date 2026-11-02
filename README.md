# EV Charger Assignment System — Syllabus Version

## Algorithms used in the project
- Linear Regression and Decision Tree Regressor: compare predictions of installation duration.
- Rule-based ranking: selects only available teams that support the requested charger type.
- Later enhancement: add real coordinates and Haversine/road distance; KNN can be used as a distance-based comparison.

## Setup (Windows PowerShell)
Open this folder in VS Code. In terminal:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_models.py
python assignment_engine.py
```
If `python` does not work, try `py`.

## Experiments folder
Contains beginner-friendly sample codes for the 10 experiments in the provided ML syllabus. Run from the project root, e.g.:
```powershell
python experiments\01_statistics.py
```
For experiment 3, run `python experiments\03_pandas_matplotlib.py`. For experiment 5, sklearn may need internet to download the California housing dataset on first run.

## Important limitations
- The historical installation dataset is synthetic, so model scores are demonstration-only.
- Team charger qualifications, availability, workload, experience and completed jobs are illustrative. Verify them with real operational data.
- Team ranking uses workload and experience only until actual location coordinates are added. No live assignment is made; supervisor approval is required.
- The provided customer work orders contain no names or mobile numbers.

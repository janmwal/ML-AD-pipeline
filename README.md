# Neurodegeneration Classification Pipeline

This repository exposes a light MLOps pipeline for predicting Alzheimer’s disease probability from region-level volumetric CSVs and producing SHAP-based explanations. Two entry points matter:

* `run_classification.py` – loads a pretrained classifier, aligns features, predicts the AD probability, and writes the full set of artefacts needed for downstream inference.
* `make_shap_graphs.py` – regenerates SHAP glass-brain and waterfall plots from the saved prediction folders.

Both scripts support either **region-per-row** CSVs (long format: one row per region) or **subject-per-row** CSVs (wide format) containing a `PR_number` column.

---

## Typical workflow

1. **Create and activate a Python 3.12 virtual environment**
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   python -V                  # should print 3.12.x

   pip install -U pip setuptools wheel
   pip install -r requirements.txt
   ```

2. **Run the classifier**
   ```bash
   python run_classification.py \
     --input_csv data/VOL_GM_unthrs.csv \
     --model lgbm \
     --GM_thrs False \
     --thrs_target youden \
     --output_folder output_pred \
     --overwrite
   ```
   * Region-per-row CSVs (`name`, `index`, `GMvalues_unthresholded`, `GMvalues_thresholded`) generate `output_pred/PRxxxx_ID_<model>_<thrs|unthrs>` automatically (e.g. `sPR04383_NS300459-...` → `PR04383_NS300459_lgbm_unthrs`).
   * Subject-per-row CSVs (wide format) reuse the value in `PR_number` for the folder name. If nothing that looks like `PR12345_ID` is found, the folder becomes `unnamed_<model>_<suffix>`.
   * `--overwrite` clears any existing folder before writing. Without it, suffixes `_0`, `_1`, … are appended to avoid collisions.

3. **Regenerate SHAP plots**
   ```bash
   # single folder
   python make_shap_graphs.py --pred_folder output_pred/PR04383_NS300459_lgbm_unthrs

   # batch over all prediction folders
   python make_shap_graphs.py --pred_folder output_pred
   ```
   Glass-brain and waterfall PNGs land directly in each prediction folder (existing plots are overwritten).

---

## `run_classification.py`

### CLI summary

| Argument | Default | Notes |
|----------|---------|-------|
| `--input_csv` | **required** | Region-per-row CSV **or** subject-per-row CSV. |
| `--model` | `lgbm` | Only LightGBM is currently supported; flag remains for forward compatibility. |
| `--GM_thrs` | `False` | Boolean selector for thresholded GM (`True` → `GMvalues_thresholded`, `False` → `GMvalues_unthresholded`). |
| `--thrs_target` | `youden` | Threshold strategy: `youden`, `sensitivity`, or `f1`. |
| `--output_folder` | `output_pred` | Base directory for subject folders. |
| `--overwrite` | `False` | Clears any existing subject folder before writing when set. |

### Behaviour highlights

* Automatically aligns features and records missing/extra columns inside `prediction.json`.
* Output naming convention: `PRxxxx_ID_<model>_<thrs|unthrs>` derived from the filename / `PR_number`; falls back to `unnamed_...`.
* Artefacts include `prediction.json`, `X_new.csv`, SHAP CSVs (`shap_transformed`, `shap_original`, `shap_original_long`), and a `merged_row.csv` combining raw values + SHAP contributions + probability.
* SHAP outputs are stored in logit space and the waterfall metadata reports logit totals, probabilities, and thresholds.

---

## `make_shap_graphs.py`

### CLI summary

| Argument | Required | Description |
|----------|----------|-------------|
| `--pred_folder` | optional (default `output_pred`) | Accepts a single prediction folder **or** a directory containing multiple subfolders with `shap_original_long.csv`. |
| `--top_k` | optional (default `10`) | Number of top-|SHAP| features shown in the waterfall plot; the remainder are aggregated into "others". |

### Behaviour highlights

* Automatically maps region names back to Neuromorphometrics atlas indices using `data/roi_rename_map.csv` and `data/neuromorphometrics.csv`. Regions lacking a recognised index are skipped with a warning.
* Both plots (`shap_glass_brain.png`, `shap_waterfall.png`) are regenerated in each prediction folder. The waterfall shading marks the logit-transformed decision threshold (left of the line bathed in light blue → CN territory, right in light red → AD territory).
* Works seamlessly across all prediction folders when pointing `--pred_folder` to `output_pred`.

Example outputs:

![Example glass brain](figs/shap_glass_brain.png)
![Example waterfall](figs/shap_waterfall.png)

---

## CSV expectations

* **Region-per-row (long) format**
  | Column | Description |
  |--------|-------------|
  | `name` | Region name (string; spaces allowed) |
  | `index` | Atlas index (int). Optional: will be reconstructed for plotting if missing. |
  | `GMvalues_unthresholded` | Numeric value per region |
  | `GMvalues_thresholded` | Numeric value per region |

* **Subject-per-row (wide) format**
  | Column | Description |
  |--------|-------------|
  | `PR_number` | Subject identifier used to name prediction folders |
  | other columns | Feature names must match the model training schema |

Keep column names exactly as shown. The scripts infer feature alignment automatically.

---

## Notes

* Always run the scripts from the activated virtual environment to ensure LightGBM, SHAP, and dependencies are available.
* `optuna_ml_ad.db` and the `models/` directory hold the tuned artefacts; they are expected to be present before invoking `run_classification.py`.
* Example outputs appear under `output_pred/` after running the workflow, ready for inspection or for the SHAP plotting script.

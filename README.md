# AI‑Driven Diet & Food Safety Recommender

An end‑to‑end prototype that blends food safety risk modeling with nutrition profiling and user preferences to recommend safer, healthier foods. It trains an ensemble model on EU RASFF alerts, scores products for safety, merges with nutrition, and personalizes results by diet, cuisine, and allergies.

---

## What’s inside

- A single notebook: `Diet+FoodSafetyModel.ipynb`
- Datasets (cleaned/ready):
  - `DATASETS/FOOD-SAFETY/rasff_cleaned_ready.csv` (≈30k rows)
  - `DATASETS/NUTRITION/nutrition_cleaned_ready.csv` (≈400k rows)
  - `DATASETS/FOOD-CHOICES/synthetic_users_cleaned_ready.csv` (500 users)
  - `DATASETS/MERGED/master_food_safety_recommender.csv` (≈99k rows; sample “final” master)

Outputs produced by the notebook (when run):
- `food_safety_model_v3.joblib` — TF‑IDF + Ensemble model bundle
- `master_food_safety_recommender_enriched.csv` — products + predicted risk + safety score
- `master_food_safety_recommender_final.csv` — enriched + normalized safety score [0,1]
- `user_food_recommendations.csv` — top‑N personalized recommendations for all users

---

## Why it matters

Most diet recommenders optimize for calories or macros only. Foodborne risks (pathogens, heavy metals, toxins) aren’t considered even though they materially impact health. This project fuses a learned food‑safety signal with nutrition and user context to prioritize options that are both safe and healthy.

---

## Quick start

Requirements
- Python 3.9–3.11
- Recommended packages: xgboost, lightgbm, imbalanced‑learn, scikit‑learn, pandas, numpy, matplotlib, seaborn, tqdm, joblib

Install (Windows bash)
```bash
python -m venv .venv
source .venv/Scripts/activate
pip install --upgrade pip
pip install xgboost lightgbm imbalanced-learn scikit-learn pandas numpy matplotlib seaborn tqdm joblib
```

Open and run the notebook
1. Open `Diet+FoodSafetyModel.ipynb` in VS Code (or Jupyter/Colab).
2. Read the top sections to understand the flow.
3. Execute cells in order: Step 0 → Step 4.

Important: file paths
- The notebook loads CSVs using bare filenames (e.g., `rasff_cleaned_ready.csv`). If running from the repo root, either:
  - Change the working directory to the matching dataset folder before running those steps, or
  - Edit the `pd.read_csv` calls to include relative paths, e.g., `DATASETS/FOOD-SAFETY/rasff_cleaned_ready.csv`, `DATASETS/NUTRITION/nutrition_cleaned_ready.csv`, etc.
- The provided `DATASETS/MERGED/master_food_safety_recommender.csv` can be used directly if you want to skip the training/scoring steps and jump to personalization/visualization.

---

## Project flow (matches the notebook)

1) Train Food Safety Risk Model (Step 1)
- Build TF‑IDF features from RASFF text fields: `PROD_CAT + NOT_COUNTRY + COUNT_ORIGEN + HAZARDS_CAT + HAZARDS`
- Handle class imbalance with oversampling
- Train soft‑voting ensemble: XGBoost + LightGBM + Logistic Regression
- Report accuracy/F1, and derive a continuous `safety_score` from class probabilities
- Save bundle: `food_safety_model_v3.joblib`

2) Score products (Step 2)
- Load model bundle
- Infer `predicted_risk` + `safety_score` for product names in the master list
- Save: `master_food_safety_recommender_enriched.csv`

2.5) Enrich & normalize (Step 2.5)
- Heuristically infer hazard category from product names (domain rules)
- Add geography fields (`NOT_COUNTRY`, `COUNT_ORIGEN`)
- Normalize `safety_score` into `safety_score_normalized` ∈ [0,1]
- Save: `master_food_safety_recommender_final.csv`

3) Personalized recommender (Step 3)
- Merge final safety with nutrition on `product_name`
- Create `health_score`:
  - If `nutrition_score` has variance: invert a scaled version (lower = healthier)
  - Else synthesize from nutrients (fat, sugars, salt, proteins, fiber)
- Blend to `recommendation_score` = 0.6×safety + 0.4×health
- Filter by user diet (vegan/vegetarian/keto), cuisine, and allergies

3.5) Batch export (Step 3.5)
- Generate top‑5 recommendations for each user
- Save: `user_food_recommendations.csv`

4) Visualization (Step 4)
- Safety vs Health scatter
- Mean scores by diet type
- Average recommendation score by diet

---

## Datasets and schema

FOOD‑SAFETY — `rasff_cleaned_ready.csv`
- Columns: `PROD_CAT`, `NOT_COUNTRY`, `COUNT_ORIGEN`, `HAZARDS_CAT`, `HAZARDS`, `RISK_DECISION`
- Labels: `RISK_DECISION ∈ {serious, undecided, not serious}` (mapped to classes)

NUTRITION — `nutrition_cleaned_ready.csv`
- Key columns: `product_name`, `energy-kcal_100g`, `fat_100g`, `saturated-fat_100g`, `carbohydrates_100g`, `sugars_100g`, `fiber_100g`, `proteins_100g`, `salt_100g`, `nutrition-score-fr_100g`
- Normalized to per‑100g; some missing values may be imputed in‑notebook

FOOD‑CHOICES (users) — `synthetic_users_cleaned_ready.csv`
- Columns (excerpt): `user_id`, `age`, `gender`, `country`, `goal`, `diet_type`, `allergies`, `health_condition`, `activity_level`, `daily_calorie_target`, `fav_cuisine`, `preferred_meal_type`, `fruit_intake`, `veggie_intake`, `vitamin_supplements`, `unsafe_food_awareness`

MERGED/FINAL — `master_food_safety_recommender*.csv`
- Core columns: `product_name`, nutrition features, `PROD_CAT`, `safety_score`/`safety_score_normalized`, optionally `health_score`, and personalization fields when exported

Note: The provided `DATASETS/MERGED/master_food_safety_recommender.csv` is a ready‑to‑use sample master with scores; the notebook may also produce `master_food_safety_recommender_enriched.csv` and `master_food_safety_recommender_final.csv` with similar structure.

---

## Model and metrics

- Features: TF‑IDF (5k uni/bi‑grams) over concatenated categorical/text fields
- Classifier: Soft‑voting ensemble (XGBoost + LightGBM + Logistic Regression)
- Reported performance: ~74.4% accuracy, weighted F1 ≈ 0.69 on held‑out data
- Safety scoring: weighted sum over class probabilities (higher = safer)

Weights for final ranking
- `recommendation_score = 0.6 * safety + 0.4 * health`
- You can change these in the notebook to emphasize public‑health risk vs. nutrition/fitness

---

## Reproduce results quickly

If you want to skip model training:
1) Use the existing `DATASETS/MERGED/master_food_safety_recommender.csv`
2) Start at Step 3 in the notebook to merge with nutrition, personalize, and export recommendations

If you want to run end‑to‑end:
1) Ensure dataset paths match your working directory (see “Important: file paths” above)
2) Run Step 1 → Step 4 sequentially

Optional small data sanity checks in Python
```python
import pandas as pd
print(pd.read_csv('DATASETS/FOOD-SAFETY/rasff_cleaned_ready.csv', nrows=3).head())
print(pd.read_csv('DATASETS/NUTRITION/nutrition_cleaned_ready.csv', nrows=3).head())
print(pd.read_csv('DATASETS/FOOD-CHOICES/synthetic_users_cleaned_ready.csv', nrows=3).head())
```

---

## Design choices and notes

- The RASFF dataset is noisy and imbalanced; text features are strong but imperfect proxies for true risk.
- Product names don’t always encode hazards; we add heuristic hazard/category inference during enrichment.
- When `nutrition_score` lacks variance, we synthesize a health metric from macro nutrients to avoid degenerate rankings.
- Large CSVs (≈400k rows) may need more memory; consider chunked loading (`chunksize`) if you hit limits.

---

## Limitations

- Safety labels reflect reporting and may not capture true risk for a specific product batch.
- Heuristic hazard mapping may introduce false positives/negatives.
- Recommendations depend on product name matching for cuisine/diet filtering.
- No real‑time recall/traceability; it’s an offline scoring approach.

---

## Ethics & responsible use

This project is an educational prototype and not a substitute for professional dietary advice or official food safety guidance. Always consult local authorities for recalls and safety alerts, and seek professional medical advice for dietary needs.

---

## Acknowledgements

- RASFF (Rapid Alert System for Food and Feed) for public alert data
- Open Food Facts (nutrition) and related open datasets
- The open‑source Python ecosystem

---


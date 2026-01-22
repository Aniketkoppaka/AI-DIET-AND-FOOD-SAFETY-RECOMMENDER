# AI‑Driven Diet & Food Safety Recommender

An end‑to‑end prototype that blends food safety risk modeling with nutrition profiling and user preferences to recommend safer, healthier foods. It trains an ensemble model on EU RASFF alerts, merges with nutrition data, and personalizes recommendations based on dietary goals, allergies, and preferences.

---

## What's inside

### Notebooks

This repository contains three main notebooks that can be run independently or together:

1. **`FoodSafetyModel.ipynb`** — Food Safety Risk Prediction
   - Trains an ensemble model (XGBoost + LightGBM + Logistic Regression) on RASFF alerts
   - Predicts contamination risk and generates safety scores for food products
   - Uses TF‑IDF features from product categories, hazards, and geographic data
   - Outputs: `food_safety_model_v3.joblib`, safety-scored product datasets

2. **`DietRecommender.ipynb`** — Personalized Diet Recommendations
   - Generates personalized food recommendations based on user profiles
   - Considers dietary restrictions (vegan, vegetarian, keto), allergies, and health goals
   - Combines nutrition data with user preferences and meal planning
   - Outputs: User-specific recommendations, meal plans

3. **`Diet+FoodSafetyModel.ipynb`** — Integrated End-to-End System
   - Combines both food safety modeling and diet recommendations in one workflow
   - Trains the safety model, scores products, and generates personalized recommendations
   - Creates a unified `recommendation_score` blending safety (60%) and health (40%)
   - Outputs: Complete recommendation system with visualizations

### Datasets

Datasets are organized in the `DATASETS/` directory:

- **FOOD-SAFETY**: `rasff_cleaned_ready.csv` (≈30k rows)
  - EU RASFF alerts with risk decisions, hazards, and product categories  

- **NUTRITION**: `nutrition_cleaned_ready.csv` (≈400k rows)
  - Nutritional information per 100g (calories, macros, nutrition scores)
  
- **FOOD-CHOICES**: `synthetic_users_cleaned_ready.csv` (500 users)
  - User profiles with dietary preferences, allergies, health goals, and activity levels  

- **MERGED**: `master_food_safety_recommender.csv` (≈99k rows)
  - Pre-scored master dataset combining safety and nutrition data

### Outputs

When notebooks are run, they produce:

- `food_safety_model_v3.joblib` — TF‑IDF + Ensemble model bundle
- `master_food_safety_recommender_enriched.csv` — Products + predicted risk + safety score
- `master_food_safety_recommender_final.csv` — Enriched + normalized safety score [0,1]
- `user_food_recommendations.csv` — Top‑N personalized recommendations for all users

---

## 🚀 NEW: SOTA Enhanced Analysis Pipeline

### Enhanced Model Comparison (`sota_pipeline.py`)

A comprehensive model comparison pipeline featuring state-of-the-art transformers and advanced ML techniques.

#### Models Included (9 Total)

| Model | Type | Parameters | Accuracy |
|-------|------|------------|----------|
| **DeBERTa-v3-large** | Transformer | 434M | **74.5%** |
| RoBERTa-large | Transformer | 355M | 72.0% |
| DeBERTa-v3-base | Transformer | 184M | 71.7% |
| RoBERTa-base | Transformer | 125M | 68.7% |
| DistilBERT | Transformer | 66M | 68.7% |
| CatBoost | ML | - | 68.2% |
| XGBoost | ML | - | 67.0% |
| LightGBM | ML | - | 66.2% |
| Ensemble | ML | - | 62.8% |

#### Techniques Applied

1. **SMOTE (Synthetic Minority Over-sampling)** - Handles class imbalance
2. **Advanced Feature Engineering** - TF-IDF tri-grams + domain-specific features
3. **Fine-tuning** - Domain-specific transformer training
4. **Hazard Severity Encoding** - Risk-based feature weighting
5. **Country Risk Encoding** - Geographic food safety features

#### Quick Run

```bash
# Navigate to project
cd AI-DIET-AND-FOOD-SAFETY-RECOMMENDER

# Activate environment
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Run SOTA pipeline
python sota_pipeline.py
```

#### Output Tables

**Table 1: Diet Type Analysis**
- Recommendation scores by diet (Vegan, Vegetarian, Keto, Regular)
- Safety scores, health scores, and top suggestions

**Table 2: Model Performance Comparison**
- Accuracy, Precision, Recall, F1-Score for all 9 models
- Best model highlighted

#### Output Files
- `model_comparison_sota.csv` — Performance metrics for all models
- `diet_type_analysis_sota.csv` — Diet-wise recommendation analysis
- `all_recommendations_sota.csv` — Complete user recommendations

---

## Why it matters

Most diet recommenders optimize for calories or macros only. Foodborne risks (pathogens, heavy metals, toxins) aren't considered even though they materially impact health. This project fuses a learned safety model with nutrition data and user profiles to deliver recommendations that are both safe and healthy.

---

## Quick start

### Requirements
- Python 3.9–3.11
- Recommended packages: xgboost, lightgbm, imbalanced‑learn, scikit‑learn, pandas, numpy, matplotlib, seaborn, tqdm, joblib

### Install (Windows bash)
```bash
python -m venv .venv
source .venv/Scripts/activate
pip install --upgrade pip
pip install xgboost lightgbm imbalanced-learn scikit-learn pandas numpy matplotlib seaborn tqdm joblib
```

### Choose your workflow

**Option 1: Run the integrated system (recommended for beginners)**
1. Open `Diet+FoodSafetyModel.ipynb` in VS Code, Jupyter, or Colab
2. Execute cells in order: Step 0 → Step 4
3. Get complete end-to-end results with visualizations

**Option 2: Run notebooks separately**
1. Start with `FoodSafetyModel.ipynb` to train the safety model and score products
2. Then run `DietRecommender.ipynb` to generate personalized recommendations
3. This approach allows you to iterate on each component independently

**Option 3: Skip training and use pre-scored data**
1. Use the existing `DATASETS/MERGED/master_food_safety_recommender.csv`
2. Jump directly to the personalization sections in either `DietRecommender.ipynb` or `Diet+FoodSafetyModel.ipynb`

### Important: file paths
- The notebooks load CSVs using relative paths. Run from the repo root directory.
- If you encounter path errors, verify the `DATASETS/` folder structure matches the expected layout.
- Example paths used: `DATASETS/FOOD-SAFETY/rasff_cleaned_ready.csv`, `DATASETS/NUTRITION/nutrition_cleaned_ready.csv`, etc.

---

## Project flow (integrated notebook)

The `Diet+FoodSafetyModel.ipynb` follows this workflow:

**1) Train Food Safety Risk Model (Step 1)**
- Build TF‑IDF features from RASFF text fields: `PROD_CAT + NOT_COUNTRY + COUNT_ORIGEN + HAZARDS_CAT + HAZARDS`
- Handle class imbalance with oversampling
- Train soft‑voting ensemble: XGBoost + LightGBM + Logistic Regression
- Report accuracy/F1, and derive a continuous `safety_score` from class probabilities
- Save bundle: `food_safety_model_v3.joblib`

**2) Score products (Step 2)**
- Load model bundle
- Infer `predicted_risk` + `safety_score` for product names in the master list
- Save: `master_food_safety_recommender_enriched.csv`

**2.5) Enrich & normalize (Step 2.5)**
- Heuristically infer hazard category from product names (domain rules)
- Add geography fields (`NOT_COUNTRY`, `COUNT_ORIGEN`)
- Normalize `safety_score` into `safety_score_normalized` ∈ [0,1]
- Save: `master_food_safety_recommender_final.csv`

**3) Personalized recommender (Step 3)**
- Merge final safety with nutrition on `product_name`
- Create `health_score`:
  - If `nutrition_score` has variance: invert a scaled version (lower = healthier)
  - Else synthesize from nutrients (fat, sugars, salt, proteins, fiber)
- Blend to `recommendation_score` = 0.6×safety + 0.4×health
- Filter by user diet (vegan/vegetarian/keto), cuisine, and allergies

**3.5) Batch export (Step 3.5)**
- Generate top‑5 recommendations for each user
- Save: `user_food_recommendations.csv`

**4) Visualization (Step 4)**
- Safety vs Health scatter
- Mean scores by diet type
- Average recommendation score by diet

---

## Datasets and schema

### FOOD‑SAFETY — `rasff_cleaned_ready.csv`
- Columns: `PROD_CAT`, `NOT_COUNTRY`, `COUNT_ORIGEN`, `HAZARDS_CAT`, `HAZARDS`, `RISK_DECISION`
- Labels: `RISK_DECISION ∈ {serious, undecided, not serious}` (mapped to classes)

### NUTRITION — `nutrition_cleaned_ready.csv`
- Key columns: `product_name`, `energy-kcal_100g`, `fat_100g`, `saturated-fat_100g`, `carbohydrates_100g`, `sugars_100g`, `fiber_100g`, `proteins_100g`, `salt_100g`, `nutrition-score-fr_100g`
- Normalized to per‑100g; some missing values may be imputed in‑notebook

### FOOD‑CHOICES (users) — `synthetic_users_cleaned_ready.csv`
- Columns (excerpt): `user_id`, `age`, `gender`, `country`, `goal`, `diet_type`, `allergies`, `health_condition`, `activity_level`, `daily_calorie_target`, `fav_cuisine`, `preferred_meal_type`, ...

### MERGED/FINAL — `master_food_safety_recommender*.csv`
- Core columns: `product_name`, nutrition features, `PROD_CAT`, `safety_score`/`safety_score_normalized`, optionally `health_score`, and personalization fields when exported

Note: The provided `DATASETS/MERGED/master_food_safety_recommender.csv` is a ready‑to‑use sample master with scores; the notebook may also produce `master_food_safety_recommender_enriched.csv` and `master_food_safety_recommender_final.csv` during execution.

---

## Model and metrics

### Food Safety Model
- **Features**: TF‑IDF (5k uni/bi‑grams) over concatenated categorical/text fields
- **Classifier**: Soft‑voting ensemble (XGBoost + LightGBM + Logistic Regression)
- **Reported performance**: ~74.4% accuracy, weighted F1 ≈ 0.69 on held‑out data
- **Safety scoring**: Weighted sum over class probabilities (higher = safer)

### Recommendation Scoring
- `recommendation_score = 0.6 * safety + 0.4 * health`
- You can change these weights in the notebooks to emphasize public‑health risk vs. nutrition/fitness

---

## Reproduce results quickly

**If you want to skip model training:**
1. Use the existing `DATASETS/MERGED/master_food_safety_recommender.csv`
2. Start at Step 3 in `Diet+FoodSafetyModel.ipynb` or use `DietRecommender.ipynb` to merge with nutrition, personalize, and export recommendations

**If you want to run end‑to‑end:**
1. Ensure dataset paths match your working directory
2. Run `Diet+FoodSafetyModel.ipynb` cells in order (Step 1 → Step 4), or
3. Run `FoodSafetyModel.ipynb` first, then `DietRecommender.ipynb`

**Optional small data sanity checks in Python:**
```python
import pandas as pd
print(pd.read_csv('DATASETS/FOOD-SAFETY/rasff_cleaned_ready.csv', nrows=3).head())
print(pd.read_csv('DATASETS/NUTRITION/nutrition_cleaned_ready.csv', nrows=3).head())
print(pd.read_csv('DATASETS/FOOD-CHOICES/synthetic_users_cleaned_ready.csv', nrows=3).head())
```

---

## Design choices and notes

- The RASFF dataset is noisy and imbalanced; text features are strong but imperfect proxies for true risk.
- Product names don't always encode hazards; we add heuristic hazard/category inference during enrichment.
- When `nutrition_score` lacks variance, we synthesize a health metric from macro nutrients to avoid degenerate rankings.
- Large CSVs (≈400k rows) may need more memory; consider chunked loading (`chunksize`) if you hit limits.

---

## Limitations

- Safety labels reflect reporting and may not capture true risk for a specific product batch.
- Heuristic hazard mapping may introduce false positives/negatives.
- Recommendations depend on product name matching for cuisine/diet filtering.
- No real‑time recall/traceability; it's an offline scoring approach.

---

## Use cases

This system can be adapted for:
- **Personal meal planning**: Generate safe and healthy meal recommendations based on individual dietary needs
- **Food safety research**: Analyze contamination patterns and risk factors across product categories
- **Public health**: Identify high-risk food products and inform safety interventions
- **Food industry**: Benchmark product safety and nutritional profiles against competitors
- **Education**: Demonstrate real-world applications of ML in food science and public health

---

## Ethics & responsible use

This project is an educational prototype and not a substitute for professional dietary advice or official food safety guidance. Always consult local authorities for recalls and safety alerts, and speak with healthcare providers for personalized diet recommendations.

---

## Acknowledgements

- RASFF (Rapid Alert System for Food and Feed) for public alert data
- Open Food Facts (nutrition) and related open datasets
- The open‑source Python ecosystem

---

## License

This project is provided for educational and research purposes. Please check individual dataset licenses before commercial use.

---

## Contributing

Contributions are welcome! Feel free to:
- Report bugs or suggest features via Issues
- Submit pull requests with improvements
- Share your own use cases or extensions

---

## Contact

For questions or collaboration opportunities, please open an issue on this repository.
